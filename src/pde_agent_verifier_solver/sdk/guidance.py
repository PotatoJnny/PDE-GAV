"""Minimal backend-neutral contract for frozen generator guidance.

Guidance is generator/distribution work.  The controller trains it once at a
round boundary against the previous incumbent, freezes the result for the next
round, and charges training plus guided generation only to campaign budgets.
Neither cost belongs to candidate prepare/query timing.

Development uses the frozen model as an online answer-first oracle with fresh
controller-seeded draws.  The protocol sets no lifetime call or sample limit;
actual wall, concurrency, and compute capacity apply.  A finite replay file may
be retained as an optional cache/reproducibility artifact, never as the access
surface or quota.  Official evaluation instead reuses one controller-generated
common hidden guided cohort for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from time import perf_counter
from types import MappingProxyType
from typing import Callable, ClassVar, Mapping, Protocol, Sequence

from .models import Case, CaseVisibility


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
Clock = Callable[[], float]


class GuidedSamplingPurpose(str, Enum):
    DEVELOPMENT_ONLINE = "development_online"
    OFFICIAL_COMMON_HIDDEN = "official_common_hidden"


class GuidanceCohortKind(str, Enum):
    AVERAGE_BASE = "average_base"
    GUIDED_ADVERSARIAL = "guided_adversarial"


class GuidanceUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateGuidanceSnapshot:
    """Public identity and campaign timing for one frozen guide.

    The training request and artifact locator are controller state. Candidate
    code only needs a stable identity for the frozen distribution it can query.
    """

    snapshot_sha256: str
    campaign_guidance_training_wall_seconds: float

    def __post_init__(self) -> None:
        _sha(self.snapshot_sha256, "candidate guidance snapshot")
        _nonnegative_number(
            self.campaign_guidance_training_wall_seconds,
            "candidate guidance training wall seconds",
        )


@dataclass(frozen=True)
class CandidateGuidedSample:
    """One phase-blind public question keyed by an opaque sample id."""

    sample_id: str
    case: Case
    answer_to_case_derivation_commitment_sha256: str

    def __post_init__(self) -> None:
        _text(self.sample_id, "candidate guided sample_id")
        if not isinstance(self.case, Case):
            raise TypeError("candidate guided sample case must be Case")
        if self.case.visibility is not CaseVisibility.DEVELOPMENT:
            raise ValueError("candidate guided sample must be DEVELOPMENT-visible")
        _sha(
            self.answer_to_case_derivation_commitment_sha256,
            "answer-to-case derivation commitment",
        )


@dataclass(frozen=True)
class CandidateGuidedSampleBatch:
    """Candidate-facing result of one fresh guided inference call.

    It intentionally omits the controller request, seed, nonce, hidden-answer
    fingerprint and answer locator. The settings digest is named as a
    commitment because it binds the caller's public sampling settings; it is
    not a hash of the returned (potentially large) Case payload.
    """

    snapshot_sha256: str
    sample_settings_commitment_sha256: str
    samples: tuple[CandidateGuidedSample, ...]
    campaign_guided_generation_wall_seconds: float

    def __post_init__(self) -> None:
        _sha(self.snapshot_sha256, "candidate guided sample snapshot")
        _sha(
            self.sample_settings_commitment_sha256,
            "guided sample settings commitment",
        )
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("candidate guided sample batch cannot be empty")
        if any(
            not isinstance(sample, CandidateGuidedSample)
            for sample in self.samples
        ):
            raise TypeError(
                "candidate guided samples must be CandidateGuidedSample values"
            )
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("candidate guided sample ids must be unique")
        _nonnegative_number(
            self.campaign_guided_generation_wall_seconds,
            "campaign_guided_generation_wall_seconds",
        )


@dataclass(frozen=True)
class GuidanceCapability:
    """Explicit capability record; absence and legacy booleans fail closed."""

    available: bool = False
    backend_id: str | None = None
    exact_unsupported_reason: str | None = (
        "backend did not declare frozen online guidance support"
    )

    protocol_request_limit: ClassVar[None] = None
    protocol_sample_limit: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("guidance available must be boolean")
        if self.available:
            _text(self.backend_id, "guidance backend_id")
            if self.exact_unsupported_reason is not None:
                raise ValueError("available guidance cannot have an unsupported reason")
        else:
            _text(self.exact_unsupported_reason, "guidance unsupported reason")
            if self.backend_id is not None:
                _text(self.backend_id, "guidance backend_id")

    @classmethod
    def online(cls, backend_id: str) -> GuidanceCapability:
        return cls(True, backend_id, None)

    @classmethod
    def unsupported(
        cls,
        exact_reason: str,
        *,
        backend_id: str | None = None,
    ) -> GuidanceCapability:
        return cls(False, backend_id, exact_reason)



def require_guidance_capability(value: object) -> GuidanceCapability:
    if not isinstance(value, GuidanceCapability):
        raise GuidanceUnsupported(
            "backend did not provide a GuidanceCapability"
        )
    if not value.available:
        assert value.exact_unsupported_reason is not None
        raise GuidanceUnsupported(value.exact_unsupported_reason)
    return value


@dataclass(frozen=True)
class GuidanceTrainingBudget:
    maximum_wall_seconds: float
    compute_capacity: str
    maximum_concurrency: int = 1

    def __post_init__(self) -> None:
        _positive_number(self.maximum_wall_seconds, "training maximum_wall_seconds")
        _text(self.compute_capacity, "training compute_capacity")
        _positive_int(self.maximum_concurrency, "training maximum_concurrency")

    def identity_payload(self) -> dict[str, object]:
        return {
            "maximum_wall_seconds": float(self.maximum_wall_seconds),
            "compute_capacity": self.compute_capacity,
            "maximum_concurrency": self.maximum_concurrency,
        }


@dataclass(frozen=True)
class GuidanceTrainingRequest:
    """One boundary update; source round 0 supports a round-1 guide.

    The controller-retained reward feedback artifact contains public cases and
    scalar incumbent rewards only. It must not expose hidden answers or latent
    generator state.
    """

    source_round_index: int
    target_round_index: int
    base_generator_sha256: str
    target_solver_sha256: str
    reward_matcher_definition_sha256: str
    reward_feedback_uri: str
    reward_feedback_sha256: str
    guidance_config_sha256: str
    controller_training_seed: int
    training_budget: GuidanceTrainingBudget

    cadence: ClassVar[str] = "once_at_round_boundary"

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_round_index, "source_round_index")
        _positive_int(self.target_round_index, "target_round_index")
        if self.target_round_index != self.source_round_index + 1:
            raise ValueError("target round must immediately follow source round")
        for value, label in (
            (self.base_generator_sha256, "base generator"),
            (self.target_solver_sha256, "target solver"),
            (self.reward_matcher_definition_sha256, "reward/matcher definition"),
            (self.reward_feedback_sha256, "reward feedback"),
            (self.guidance_config_sha256, "guidance config"),
        ):
            _sha(value, label)
        _text(self.reward_feedback_uri, "reward feedback uri")
        _nonnegative_int(self.controller_training_seed, "controller_training_seed")
        if not isinstance(self.training_budget, GuidanceTrainingBudget):
            raise TypeError("training_budget must be GuidanceTrainingBudget")

    @property
    def previous_incumbent_sha256(self) -> str:
        return self.target_solver_sha256

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_round_index": self.source_round_index,
            "target_round_index": self.target_round_index,
            "base_generator_sha256": self.base_generator_sha256,
            "target_solver_sha256": self.target_solver_sha256,
            "reward_matcher_definition_sha256": self.reward_matcher_definition_sha256,
            "reward_feedback_uri": self.reward_feedback_uri,
            "reward_feedback_sha256": self.reward_feedback_sha256,
            "guidance_config_sha256": self.guidance_config_sha256,
            "controller_training_seed": self.controller_training_seed,
            "training_budget": self.training_budget.identity_payload(),
            "cadence": self.cadence,
        }


@dataclass(frozen=True)
class FrozenGuidanceSnapshot:
    """Frozen artifact identity; its timing belongs only to the campaign."""

    training_request: GuidanceTrainingRequest
    backend_id: str
    guided_artifact_uri: str
    guided_artifact_sha256: str
    campaign_guidance_training_wall_seconds: float
    snapshot_sha256: str = field(init=False)

    frozen: ClassVar[bool] = True
    training_attempts_per_boundary: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.training_request, GuidanceTrainingRequest):
            raise TypeError("training_request must be GuidanceTrainingRequest")
        _text(self.backend_id, "guidance backend_id")
        _text(self.guided_artifact_uri, "guided artifact uri")
        _sha(self.guided_artifact_sha256, "guided artifact")
        wall = self.campaign_guidance_training_wall_seconds
        _nonnegative_number(wall, "campaign_guidance_training_wall_seconds")
        if wall > self.training_request.training_budget.maximum_wall_seconds:
            raise ValueError("guidance training exceeded its campaign wall budget")
        object.__setattr__(
            self,
            "snapshot_sha256",
            _digest(
                {
                    "backend_id": self.backend_id,
                    "training": self.training_request.identity_payload(),
                    "guided_artifact_sha256": self.guided_artifact_sha256,
                }
            ),
        )


@dataclass(frozen=True)
class GuidanceFailureAudit:
    """Exact failure reason with optional controller-retained artifact."""

    source_round_index: int
    target_round_index: int
    stage: str
    exact_reason: str
    artifact_uri: str | None = None
    artifact_sha256: str | None = None
    campaign_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_round_index, "failure source_round_index")
        _positive_int(self.target_round_index, "failure target_round_index")
        if self.target_round_index != self.source_round_index + 1:
            raise ValueError("failure target round must follow source round")
        _text(self.stage, "guidance failure stage")
        _text(self.exact_reason, "guidance exact failure reason")
        if (self.artifact_uri is None) != (self.artifact_sha256 is None):
            raise ValueError("failure artifact uri and sha256 must appear together")
        if self.artifact_uri is not None:
            _text(self.artifact_uri, "failure artifact uri")
            assert self.artifact_sha256 is not None
            _sha(self.artifact_sha256, "failure artifact")
        if self.campaign_wall_seconds is not None:
            _nonnegative_number(self.campaign_wall_seconds, "failure campaign wall")


@dataclass(frozen=True)
class GuidedSampleSpec:
    """Agent-selected count/settings; it deliberately contains no randomness."""

    sample_count: int
    settings: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )
    settings_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.sample_count, "guided sample_count")
        frozen = _freeze_settings(self.settings)
        object.__setattr__(self, "settings", frozen)
        object.__setattr__(self, "settings_sha256", _digest(dict(frozen)))


@dataclass(frozen=True)
class GuidedSampleRequest:
    """Controller adds a fresh seed/nonce and actual resource allocation."""

    spec: GuidedSampleSpec
    request_id: str
    purpose: GuidedSamplingPurpose
    snapshot_sha256: str
    controller_seed: int
    controller_nonce: str
    inference_wall_budget_seconds: float
    compute_capacity: str

    protocol_request_limit: ClassVar[None] = None
    protocol_sample_limit: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, GuidedSampleSpec):
            raise TypeError("guided request spec must be GuidedSampleSpec")
        _text(self.request_id, "guided request_id")
        if not isinstance(self.purpose, GuidedSamplingPurpose):
            raise TypeError("purpose must be GuidedSamplingPurpose")
        _sha(self.snapshot_sha256, "guided snapshot")
        _nonnegative_int(self.controller_seed, "controller_seed")
        _text(self.controller_nonce, "controller_nonce")
        _positive_number(self.inference_wall_budget_seconds, "inference wall budget")
        _text(self.compute_capacity, "guided compute_capacity")

    @property
    def expected_visibility(self) -> CaseVisibility:
        if self.purpose is GuidedSamplingPurpose.DEVELOPMENT_ONLINE:
            return CaseVisibility.DEVELOPMENT
        return CaseVisibility.HIDDEN


@dataclass(frozen=True)
class GuidedAnswerFirstSample:
    sample_id: str
    snapshot_sha256: str
    case: Case
    reference_answer_sha256: str
    public_case_sha256: str
    answer_to_case_derivation_sha256: str
    answer_access_locator: str | None = None

    def __post_init__(self) -> None:
        _text(self.sample_id, "guided sample_id")
        _sha(self.snapshot_sha256, "guided sample snapshot")
        if not isinstance(self.case, Case):
            raise TypeError("guided sample case must be Case")
        _sha(self.reference_answer_sha256, "reference answer")
        _sha(self.public_case_sha256, "public case")
        _sha(self.answer_to_case_derivation_sha256, "answer-to-case derivation")
        if self.answer_access_locator is not None:
            _text(self.answer_access_locator, "answer access locator")


@dataclass(frozen=True)
class GuidedSampleBatch:
    """Fresh online result with campaign generation timing, never candidate time."""

    request: GuidedSampleRequest
    samples: tuple[GuidedAnswerFirstSample, ...]
    campaign_guided_generation_wall_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.request, GuidedSampleRequest):
            raise TypeError("guided batch request must be GuidedSampleRequest")
        if not isinstance(self.samples, tuple) or any(
            not isinstance(sample, GuidedAnswerFirstSample)
            for sample in self.samples
        ):
            raise TypeError("guided samples must be a tuple of answer-first samples")
        if len(self.samples) != self.request.spec.sample_count:
            raise ValueError("guided batch size differs from Agent-selected count")
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("guided sample ids must be unique")
        for sample in self.samples:
            if sample.snapshot_sha256 != self.request.snapshot_sha256:
                raise ValueError("guided sample uses the wrong frozen snapshot")
            if sample.case.visibility is not self.request.expected_visibility:
                raise ValueError("guided sample visibility does not match request")
            if (
                self.request.purpose
                is GuidedSamplingPurpose.DEVELOPMENT_ONLINE
                and sample.answer_access_locator is None
            ):
                raise ValueError("development sample needs answer access")
            if (
                self.request.purpose
                is GuidedSamplingPurpose.OFFICIAL_COMMON_HIDDEN
                and sample.answer_access_locator is not None
            ):
                raise ValueError("official hidden sample cannot expose answer access")
        _nonnegative_number(
            self.campaign_guided_generation_wall_seconds,
            "campaign_guided_generation_wall_seconds",
        )


class FrozenGuidanceOnlineOracle(Protocol):
    """Frozen inference-only generator oracle; no training method is exposed."""

    @property
    def guidance_capability(self) -> GuidanceCapability: ...

    @property
    def snapshot(self) -> FrozenGuidanceSnapshot: ...

    def generate_answer_first(
        self,
        request: GuidedSampleRequest,
    ) -> Sequence[GuidedAnswerFirstSample]: ...


class GuidanceTrainingBackend(Protocol):
    """Thin generator-layer integration point for train-once then online use."""

    @property
    def guidance_capability(self) -> GuidanceCapability: ...

    def train_guidance(
        self,
        request: GuidanceTrainingRequest,
    ) -> FrozenGuidanceSnapshot: ...

    def open_guidance(
        self,
        snapshot: FrozenGuidanceSnapshot,
    ) -> FrozenGuidanceOnlineOracle: ...


def sample_frozen_guidance(
    oracle: FrozenGuidanceOnlineOracle,
    request: GuidedSampleRequest,
    *,
    clock: Clock = perf_counter,
) -> GuidedSampleBatch:
    """Time one fresh draw as campaign generation, outside candidate clocks."""

    capability = require_guidance_capability(
        getattr(oracle, "guidance_capability", None)
    )
    snapshot = getattr(oracle, "snapshot", None)
    if not isinstance(snapshot, FrozenGuidanceSnapshot):
        raise TypeError("guided oracle must expose FrozenGuidanceSnapshot")
    if capability.backend_id != snapshot.backend_id:
        raise ValueError("guided oracle capability and snapshot backend differ")
    if snapshot.snapshot_sha256 != request.snapshot_sha256:
        raise ValueError("guided request and oracle snapshot differ")
    generate = getattr(oracle, "generate_answer_first", None)
    if not callable(generate):
        raise TypeError("guided oracle must define generate_answer_first")

    started = _clock(clock)
    samples = tuple(generate(request))
    wall_seconds = _elapsed(started, _clock(clock))
    if wall_seconds > request.inference_wall_budget_seconds:
        raise TimeoutError(
            "guided inference exceeded controller wall allocation: "
            f"observed={wall_seconds:.17g}; "
            f"maximum={request.inference_wall_budget_seconds:.17g}"
        )
    return GuidedSampleBatch(request, samples, wall_seconds)


@dataclass(frozen=True)
class OfficialCommonGuidedCohort:
    """One hidden batch reused unchanged across leaderboard/final candidates."""

    cohort_id: str
    batch: GuidedSampleBatch

    def __post_init__(self) -> None:
        _text(self.cohort_id, "official guided cohort_id")
        if not isinstance(self.batch, GuidedSampleBatch):
            raise TypeError("official cohort batch must be GuidedSampleBatch")
        if (
            self.batch.request.purpose
            is not GuidedSamplingPurpose.OFFICIAL_COMMON_HIDDEN
        ):
            raise ValueError("official cohort needs a common hidden draw")

    def samples_for_candidate(
        self,
        candidate_id: str,
    ) -> tuple[GuidedAnswerFirstSample, ...]:
        _text(candidate_id, "candidate_id")
        return self.batch.samples


@dataclass(frozen=True)
class CohortMetrics:
    cohort_id: str
    kind: GuidanceCohortKind
    generator_or_guidance_sha256: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        _text(self.cohort_id, "metric cohort_id")
        if not isinstance(self.kind, GuidanceCohortKind):
            raise TypeError("metric kind must be GuidanceCohortKind")
        _sha(self.generator_or_guidance_sha256, "cohort generator/guidance")
        object.__setattr__(self, "values", _freeze_metrics(self.values))


@dataclass(frozen=True)
class SeparatedGuidanceMetrics:
    """No combined scalar exists until a caller explicitly supplies weights."""

    average_base: CohortMetrics
    guided_adversarial: CohortMetrics

    def __post_init__(self) -> None:
        if self.average_base.kind is not GuidanceCohortKind.AVERAGE_BASE:
            raise ValueError("average_base has the wrong cohort kind")
        if (
            self.guided_adversarial.kind
            is not GuidanceCohortKind.GUIDED_ADVERSARIAL
        ):
            raise ValueError("guided_adversarial has the wrong cohort kind")
        if self.average_base.cohort_id == self.guided_adversarial.cohort_id:
            raise ValueError("base and guided cohorts need distinct ids")

    def explicitly_weighted_metric(
        self,
        name: str,
        *,
        average_base_weight: float,
        guided_adversarial_weight: float,
    ) -> float:
        _text(name, "metric name")
        _nonnegative_number(average_base_weight, "average_base_weight")
        _nonnegative_number(
            guided_adversarial_weight,
            "guided_adversarial_weight",
        )
        total = average_base_weight + guided_adversarial_weight
        if total <= 0.0:
            raise ValueError("explicit weights need a positive total")
        return (
            average_base_weight * self.average_base.values[name]
            + guided_adversarial_weight * self.guided_adversarial.values[name]
        ) / total


def _freeze_settings(
    settings: Mapping[str, str | int | float | bool | None],
) -> Mapping[str, str | int | float | bool | None]:
    if not isinstance(settings, Mapping):
        raise TypeError("guided settings must be a mapping")
    copied: dict[str, str | int | float | bool | None] = {}
    for name, value in settings.items():
        _text(name, "guided setting name")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError("guided settings must contain JSON scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("guided numeric settings must be finite")
        copied[name] = value
    return MappingProxyType(copied)


def _freeze_metrics(values: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("cohort metrics need a nonempty mapping")
    copied: dict[str, float] = {}
    for name, value in values.items():
        _text(name, "metric name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("cohort metric values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("cohort metric values must be finite")
        copied[name] = number
    return MappingProxyType(copied)


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


def _positive_int(value: int, label: str) -> None:
    _nonnegative_int(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")


def _nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _positive_number(value: float, label: str) -> None:
    _nonnegative_number(value, label)
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be positive")


def _nonnegative_number(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")


def _clock(clock: Clock) -> float:
    value = clock()
    _nonnegative_number(value, "clock value")
    return float(value)


def _elapsed(started: float, finished: float) -> float:
    if finished < started:
        raise RuntimeError("wall clock moved backwards")
    return finished - started
