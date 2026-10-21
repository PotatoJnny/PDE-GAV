"""Equation-independent robust selection from standardized per-case errors.

PDE plugins remain responsible for defining and computing macroscopic errors
and gates.  This module only consumes finite, nonnegative errors where lower is
better, preserves their controller-defined case order, groups explicitly
declared case lineage into independent roots, and decides whether a candidate
has enough paired evidence to replace an incumbent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite, sqrt
from statistics import fmean
from typing import Iterable, Sequence


class PairedObjectiveMode(str, Enum):
    """Controller-level use of generated/perturbed answer-first pairs."""

    WEIGHTED_COMPOSITE = "weighted_composite"
    PRIMARY_WITH_PERTURBED_GUARDRAIL = "primary_with_perturbed_guardrail"


@dataclass(frozen=True)
class PairedObjectivePolicy:
    """Equation-independent policy for paired generated/perturbed scores."""

    mode: PairedObjectiveMode = PairedObjectiveMode.WEIGHTED_COMPOSITE
    base_weight: float = 0.5
    perturbed_weight: float = 0.5
    guardrail_relative_margin: float = 0.05
    guardrail_confidence_z: float = 1.6448536269514722

    def __post_init__(self) -> None:
        try:
            mode = PairedObjectiveMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported paired objective mode") from exc
        object.__setattr__(self, "mode", mode)

        weights = (float(self.base_weight), float(self.perturbed_weight))
        if any(not isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("paired objective weights must be finite and nonnegative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one paired objective weight must be positive")
        object.__setattr__(self, "base_weight", weights[0])
        object.__setattr__(self, "perturbed_weight", weights[1])

        margin = float(self.guardrail_relative_margin)
        if not isfinite(margin) or margin < 0.0:
            raise ValueError(
                "guardrail relative margin must be finite and nonnegative"
            )
        object.__setattr__(self, "guardrail_relative_margin", margin)

        confidence_z = float(self.guardrail_confidence_z)
        if not isfinite(confidence_z) or confidence_z < 0.0:
            raise ValueError("guardrail confidence_z must be finite and nonnegative")
        object.__setattr__(self, "guardrail_confidence_z", confidence_z)

    @property
    def normalized_base_weight(self) -> float:
        return self.base_weight / (self.base_weight + self.perturbed_weight)

    @property
    def normalized_perturbed_weight(self) -> float:
        return self.perturbed_weight / (self.base_weight + self.perturbed_weight)


@dataclass(frozen=True)
class MacroCandidateScore:
    """One immutable artifact scored against one private target commitment."""

    artifact_id: str
    artifact_digest: str
    target_sha256: str
    contract_id: str
    case_ids: tuple[str, ...]
    errors: tuple[float, ...]
    valid: bool = True
    gate_failures: tuple[str, ...] = ()
    root_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        case_ids = tuple(self.case_ids)
        errors = tuple(float(value) for value in self.errors)
        gate_failures = tuple(self.gate_failures)
        root_ids = None if self.root_ids is None else tuple(self.root_ids)
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "gate_failures", gate_failures)
        object.__setattr__(self, "root_ids", root_ids)

        if not self.artifact_id.strip() or not self.artifact_digest.strip():
            raise ValueError("artifact id and digest must be nonempty")
        if not self.target_sha256.strip() or not self.contract_id.strip():
            raise ValueError("target digest and macro contract must be nonempty")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        if not case_ids or len(case_ids) != len(errors):
            raise ValueError("case ids and errors need one common nonzero length")
        if any(not case_id.strip() for case_id in case_ids):
            raise ValueError("case ids must be nonempty")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case ids must be unique")
        if any(not isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("macro errors must be finite and nonnegative")
        if any(not failure.strip() for failure in gate_failures):
            raise ValueError("gate failures must be nonempty strings")
        if root_ids is not None:
            if len(root_ids) != len(case_ids):
                raise ValueError("root ids must align one-for-one with case ids")
            if any(
                not isinstance(root_id, str) or not root_id.strip()
                for root_id in root_ids
            ):
                raise ValueError("root ids must be nonempty strings")
            last_position: dict[str, int] = {}
            for position, root_id in enumerate(root_ids):
                previous = last_position.get(root_id)
                if previous is not None and previous != position - 1:
                    raise ValueError(
                        "members of one root must be contiguous in case order"
                    )
                last_position[root_id] = position

    @property
    def eligible(self) -> bool:
        return self.valid and not self.gate_failures


@dataclass(frozen=True)
class PairedScoreSplit:
    """Generated base and perturbed child scores, one member per root."""

    base: MacroCandidateScore
    perturbed: MacroCandidateScore


@dataclass(frozen=True)
class PerturbedGuardrailComparison:
    """One-sided paired non-regression evidence on perturbed children."""

    case_ids: tuple[str, ...]
    root_ids: tuple[str, ...]
    candidate_mean: float
    incumbent_mean: float
    mean_excess: float
    standard_error: float
    confidence_lower: float
    confidence_z: float
    practical_relative_margin: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RobustAggregate:
    """Mean, upper-tail conditional mean, and maximum error."""

    mean: float
    cvar: float
    max: float


@dataclass(frozen=True)
class EvidenceStage:
    """One cumulative paired-evidence promotion gate."""

    case_count: int
    confidence_z: float = 1.96
    min_mean_improvement: float = 0.0
    min_win_rate: float = 0.5
    require_cvar_no_regression: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_count, int)
            or isinstance(self.case_count, bool)
            or self.case_count < 1
        ):
            raise ValueError("evidence case_count must be a positive integer")
        if not isfinite(float(self.confidence_z)) or self.confidence_z < 0.0:
            raise ValueError("confidence_z must be finite and nonnegative")
        if (
            not isfinite(float(self.min_mean_improvement))
            or self.min_mean_improvement < 0.0
        ):
            raise ValueError("min_mean_improvement must be finite and nonnegative")
        if not isfinite(float(self.min_win_rate)) or not 0.0 <= self.min_win_rate <= 1.0:
            raise ValueError("min_win_rate must lie in [0, 1]")
        if not isinstance(self.require_cvar_no_regression, bool):
            raise TypeError("require_cvar_no_regression must be a bool")
        if self.case_count < 2 and float(self.confidence_z) > 0.0:
            raise ValueError(
                "positive-confidence evidence needs at least two paired cases"
            )


@dataclass(frozen=True)
class PairedComparison:
    """Audit record for one candidate at one cumulative evidence stage."""

    candidate_artifact_id: str
    candidate_digest: str
    incumbent_artifact_id: str
    incumbent_digest: str
    target_sha256: str
    contract_id: str
    stage: EvidenceStage
    case_ids: tuple[str, ...]
    candidate_aggregate: RobustAggregate
    incumbent_aggregate: RobustAggregate
    mean_improvement: float
    confidence_lower: float
    confidence_upper: float
    win_rate: float
    passed: bool
    failures: tuple[str, ...]

    confidence_group_size: int = 1
    root_ids: tuple[str, ...] = ()

    @property
    def confidence_sample_count(self) -> int:
        if self.root_ids:
            return len(self.root_ids)
        return len(self.case_ids) // self.confidence_group_size


@dataclass(frozen=True)
class PromotionResult:
    """Deterministic result of comparing candidates with one incumbent."""

    incumbent_artifact_id: str
    incumbent_digest: str
    selected_artifact_id: str
    selected_digest: str
    promoted: bool
    reason: str
    comparisons: tuple[PairedComparison, ...]


def aggregate(
    errors: Iterable[float],
    *,
    cvar_tail_fraction: float = 0.1,
) -> RobustAggregate:
    """Aggregate errors using the worst ``cvar_tail_fraction`` upper tail."""

    values = tuple(float(value) for value in errors)
    if not values:
        raise ValueError("cannot aggregate an empty error sequence")
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("macro errors must be finite and nonnegative")
    if (
        not isfinite(float(cvar_tail_fraction))
        or not 0.0 < cvar_tail_fraction <= 1.0
    ):
        raise ValueError("cvar_tail_fraction must lie in (0, 1]")

    tail_count = max(1, ceil(len(values) * float(cvar_tail_fraction)))
    upper_tail = sorted(values, reverse=True)[:tail_count]
    return RobustAggregate(
        mean=fmean(values),
        cvar=fmean(upper_tail),
        max=max(values),
    )


def aggregate_score(
    score: MacroCandidateScore,
    case_count: int | None = None,
    *,
    cvar_tail_fraction: float = 0.1,
) -> RobustAggregate:
    """Aggregate a raw member prefix using explicit roots when available."""

    if not isinstance(score, MacroCandidateScore):
        raise TypeError("aggregate_score needs a MacroCandidateScore")
    count = len(score.case_ids) if case_count is None else case_count
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        raise ValueError("aggregate case_count must be a positive integer")
    if count > len(score.case_ids):
        raise ValueError("score lacks the requested aggregate prefix")
    if score.root_ids is None:
        values = score.errors[:count]
    else:
        _root_ids, values = _root_errors_for_prefix(score, count)
    return aggregate(values, cvar_tail_fraction=cvar_tail_fraction)


def _validate_confidence_group(
    stage: EvidenceStage,
    confidence_group_size: int,
) -> None:
    _validate_confidence_group_size(confidence_group_size)
    if stage.case_count % confidence_group_size:
        raise ValueError(
            "evidence case_count must be divisible by confidence_group_size"
        )
    if (
        float(stage.confidence_z) > 0.0
        and stage.case_count // confidence_group_size < 2
    ):
        raise ValueError(
            "positive-confidence evidence needs at least two independent groups"
        )


def _validate_confidence_group_size(confidence_group_size: int) -> None:
    if (
        not isinstance(confidence_group_size, int)
        or isinstance(confidence_group_size, bool)
        or confidence_group_size < 1
    ):
        raise ValueError("confidence_group_size must be a positive integer")


def _root_errors_for_prefix(
    score: MacroCandidateScore,
    case_count: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return one equal-weight error per complete explicit lineage root."""

    if score.root_ids is None:
        raise ValueError("explicit root ids are required for root grouping")
    if case_count > len(score.case_ids):
        raise ValueError("candidate lacks the requested root evidence prefix")
    if (
        case_count < len(score.root_ids)
        and score.root_ids[case_count - 1] == score.root_ids[case_count]
    ):
        raise ValueError("evidence case_count cannot split a root group")

    ordered_roots: list[str] = []
    grouped_errors: list[list[float]] = []
    for root_id, error in zip(
        score.root_ids[:case_count],
        score.errors[:case_count],
    ):
        if not ordered_roots or ordered_roots[-1] != root_id:
            ordered_roots.append(root_id)
            grouped_errors.append([])
        grouped_errors[-1].append(error)
    return (
        tuple(ordered_roots),
        tuple(fmean(values) for values in grouped_errors),
    )


def split_paired_score(score: MacroCandidateScore) -> PairedScoreSplit:
    """Split explicit two-member roots ordered as base then perturbed.

    The returned component scores contain one case per root. Missing lineage or
    roots with any arity other than two are rejected rather than guessed.
    """

    if not isinstance(score, MacroCandidateScore):
        raise TypeError("split_paired_score needs a MacroCandidateScore")
    if score.root_ids is None:
        raise ValueError("paired objective needs explicit root lineage")

    ordered_roots: list[str] = []
    grouped_indexes: list[list[int]] = []
    for index, root_id in enumerate(score.root_ids):
        if not ordered_roots or ordered_roots[-1] != root_id:
            ordered_roots.append(root_id)
            grouped_indexes.append([])
        grouped_indexes[-1].append(index)
    if any(len(indexes) != 2 for indexes in grouped_indexes):
        raise ValueError("each paired objective root needs exactly base and perturbed")

    def component(member: int) -> MacroCandidateScore:
        indexes = tuple(indexes[member] for indexes in grouped_indexes)
        return MacroCandidateScore(
            artifact_id=score.artifact_id,
            artifact_digest=score.artifact_digest,
            target_sha256=score.target_sha256,
            contract_id=score.contract_id,
            case_ids=tuple(score.case_ids[index] for index in indexes),
            errors=tuple(score.errors[index] for index in indexes),
            valid=score.valid,
            gate_failures=score.gate_failures,
            root_ids=tuple(ordered_roots),
        )

    return PairedScoreSplit(base=component(0), perturbed=component(1))


def paired_weighted_composite(
    score: MacroCandidateScore,
    *,
    base_weight: float = 0.5,
    perturbed_weight: float = 0.5,
) -> MacroCandidateScore:
    """Collapse each pair to one normalized weighted error per lineage root.

    The default is numerically identical to the legacy per-root fmean.
    The returned score is intentionally one-case-per-root, so callers use root
    counts rather than the original child-member counts.
    """

    policy = PairedObjectivePolicy(
        base_weight=base_weight,
        perturbed_weight=perturbed_weight,
    )
    split = split_paired_score(score)
    if policy.base_weight == policy.perturbed_weight:
        errors = tuple(
            fmean((base, perturbed))
            for base, perturbed in zip(
                split.base.errors,
                split.perturbed.errors,
            )
        )
    else:
        errors = tuple(
            policy.normalized_base_weight * base
            + policy.normalized_perturbed_weight * perturbed
            for base, perturbed in zip(
                split.base.errors,
                split.perturbed.errors,
            )
        )
    return MacroCandidateScore(
        artifact_id=score.artifact_id,
        artifact_digest=score.artifact_digest,
        target_sha256=score.target_sha256,
        contract_id=score.contract_id,
        case_ids=split.base.case_ids,
        errors=errors,
        valid=score.valid,
        gate_failures=score.gate_failures,
        root_ids=split.base.root_ids,
    )


def paired_objective_score(
    score: MacroCandidateScore,
    policy: PairedObjectivePolicy | None = None,
) -> MacroCandidateScore:
    """Return the scalar score used to rank a paired candidate."""

    selected_policy = PairedObjectivePolicy() if policy is None else policy
    if not isinstance(selected_policy, PairedObjectivePolicy):
        raise TypeError("policy must be a PairedObjectivePolicy")
    if selected_policy.mode is PairedObjectiveMode.WEIGHTED_COMPOSITE:
        return paired_weighted_composite(
            score,
            base_weight=selected_policy.base_weight,
            perturbed_weight=selected_policy.perturbed_weight,
        )
    return split_paired_score(score).base


def compare_perturbed_non_regression(
    candidate: MacroCandidateScore,
    incumbent: MacroCandidateScore,
    *,
    policy: PairedObjectivePolicy | None = None,
) -> PerturbedGuardrailComparison:
    """Apply a paired one-sided practical non-regression guardrail.

    For each root, excess is candidate - (1 + margin) * incumbent.
    Regression fails only when the lower one-sided confidence bound is above
    zero. Invalid or gate-failed artifacts and insufficient confidence evidence
    fail closed independently of the numerical bound.
    """

    selected_policy = (
        PairedObjectivePolicy(
            mode=PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
        )
        if policy is None
        else policy
    )
    if not isinstance(selected_policy, PairedObjectivePolicy):
        raise TypeError("policy must be a PairedObjectivePolicy")
    _validate_pairing(candidate, incumbent)
    candidate_perturbed = split_paired_score(candidate).perturbed
    incumbent_perturbed = split_paired_score(incumbent).perturbed

    excesses = tuple(
        candidate_error
        - (1.0 + selected_policy.guardrail_relative_margin) * incumbent_error
        for candidate_error, incumbent_error in zip(
            candidate_perturbed.errors,
            incumbent_perturbed.errors,
        )
    )
    mean_excess = fmean(excesses)
    if len(excesses) == 1:
        standard_error = 0.0
    else:
        sample_variance = sum(
            (value - mean_excess) ** 2 for value in excesses
        ) / (len(excesses) - 1)
        standard_error = sqrt(sample_variance / len(excesses))
    confidence_lower = (
        mean_excess
        - selected_policy.guardrail_confidence_z * standard_error
    )

    failures: list[str] = []
    if not candidate.valid:
        failures.append("candidate_invalid")
    failures.extend(
        f"candidate_gate:{failure}" for failure in candidate.gate_failures
    )
    if not incumbent.valid:
        failures.append("incumbent_invalid")
    failures.extend(
        f"incumbent_gate:{failure}" for failure in incumbent.gate_failures
    )
    if (
        selected_policy.guardrail_confidence_z > 0.0
        and len(excesses) < 2
    ):
        failures.append("insufficient_guardrail_evidence")
    if confidence_lower > 0.0:
        failures.append("perturbed_regression_beyond_margin")

    return PerturbedGuardrailComparison(
        case_ids=candidate_perturbed.case_ids,
        root_ids=candidate_perturbed.root_ids or (),
        candidate_mean=fmean(candidate_perturbed.errors),
        incumbent_mean=fmean(incumbent_perturbed.errors),
        mean_excess=mean_excess,
        standard_error=standard_error,
        confidence_lower=confidence_lower,
        confidence_z=selected_policy.guardrail_confidence_z,
        practical_relative_margin=selected_policy.guardrail_relative_margin,
        passed=not failures,
        failures=tuple(failures),
    )


def paired_compare(
    candidate: MacroCandidateScore,
    incumbent: MacroCandidateScore,
    stage: EvidenceStage,
    *,
    cvar_tail_fraction: float = 0.1,
    confidence_group_size: int = 1,
) -> PairedComparison:
    """Compare exact prefixes using explicit roots as statistical units."""

    _validate_pairing(candidate, incumbent)
    if not incumbent.eligible:
        raise ValueError("incumbent must be valid and pass every gate")
    if len(candidate.case_ids) < stage.case_count:
        raise ValueError("candidate and incumbent lack the requested cumulative evidence")

    case_ids = candidate.case_ids[: stage.case_count]
    root_ids: tuple[str, ...] = ()
    if candidate.root_ids is None:
        _validate_confidence_group(stage, confidence_group_size)
        candidate_errors = candidate.errors[: stage.case_count]
        incumbent_errors = incumbent.errors[: stage.case_count]
    else:
        _validate_confidence_group_size(confidence_group_size)
        root_ids, candidate_errors = _root_errors_for_prefix(
            candidate,
            stage.case_count,
        )
        incumbent_root_ids, incumbent_errors = _root_errors_for_prefix(
            incumbent,
            stage.case_count,
        )
        if root_ids != incumbent_root_ids:
            raise ValueError("candidate and incumbent root order must match exactly")
        if (
            float(stage.confidence_z) > 0.0
            and len(root_ids) < 2
        ):
            raise ValueError(
                "positive-confidence evidence needs at least two independent roots"
            )
    candidate_aggregate = aggregate_score(
        candidate,
        stage.case_count,
        cvar_tail_fraction=cvar_tail_fraction,
    )
    incumbent_aggregate = aggregate_score(
        incumbent,
        stage.case_count,
        cvar_tail_fraction=cvar_tail_fraction,
    )

    improvements = tuple(
        incumbent_error - candidate_error
        for candidate_error, incumbent_error in zip(candidate_errors, incumbent_errors)
    )
    confidence_improvements = (
        improvements
        if root_ids
        else tuple(
            fmean(improvements[start : start + confidence_group_size])
            for start in range(0, len(improvements), confidence_group_size)
        )
    )
    mean_improvement = fmean(improvements)
    confidence_mean = fmean(confidence_improvements)
    if len(confidence_improvements) == 1:
        standard_error = 0.0
    else:
        squared_deviations = sum(
            (value - confidence_mean) ** 2
            for value in confidence_improvements
        )
        sample_variance = squared_deviations / (
            len(confidence_improvements) - 1
        )
        standard_error = sqrt(
            sample_variance / len(confidence_improvements)
        )
    radius = float(stage.confidence_z) * standard_error
    confidence_lower = mean_improvement - radius
    confidence_upper = mean_improvement + radius
    win_rate = sum(value > 0.0 for value in improvements) / len(improvements)

    failures: list[str] = []
    if not candidate.valid:
        failures.append("candidate_invalid")
    failures.extend(f"candidate_gate:{failure}" for failure in candidate.gate_failures)
    if confidence_lower <= float(stage.min_mean_improvement):
        failures.append("mean_improvement_not_supported")
    if win_rate < float(stage.min_win_rate):
        failures.append("win_rate_below_minimum")
    if (
        stage.require_cvar_no_regression
        and candidate_aggregate.cvar > incumbent_aggregate.cvar
    ):
        failures.append("cvar_regression")

    return PairedComparison(
        candidate_artifact_id=candidate.artifact_id,
        candidate_digest=candidate.artifact_digest,
        incumbent_artifact_id=incumbent.artifact_id,
        incumbent_digest=incumbent.artifact_digest,
        target_sha256=candidate.target_sha256,
        contract_id=candidate.contract_id,
        stage=stage,
        case_ids=case_ids,
        candidate_aggregate=candidate_aggregate,
        incumbent_aggregate=incumbent_aggregate,
        mean_improvement=mean_improvement,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        win_rate=win_rate,
        passed=not failures,
        failures=tuple(failures),
        confidence_group_size=confidence_group_size,
        root_ids=root_ids,
    )


def select_or_keep(
    incumbent: MacroCandidateScore,
    candidate: MacroCandidateScore,
    stages: Sequence[EvidenceStage],
    *,
    cvar_tail_fraction: float = 0.1,
    confidence_group_size: int = 1,
) -> PromotionResult:
    """Promote one pre-selected challenger only after every stage passes.

    Selecting a finalist and testing promotion are deliberately separate. The
    controller ranks any number of finalists on a selection cohort, then calls
    this function once with exactly one challenger on an independent promotion
    cohort. This keeps the promotion gate free of hidden multiple-comparison
    behavior. A tie always retains the incumbent.
    """

    ordered_stages = _validate_stages(stages)
    if not incumbent.eligible:
        raise ValueError("incumbent must be valid and pass every gate")
    if not isinstance(candidate, MacroCandidateScore):
        raise TypeError("promotion accepts exactly one MacroCandidateScore challenger")
    if incumbent.artifact_id == candidate.artifact_id:
        raise ValueError("a candidate cannot reuse the incumbent artifact id")
    if incumbent.artifact_digest == candidate.artifact_digest:
        raise ValueError("a candidate cannot reuse the incumbent artifact digest")
    _validate_pairing(candidate, incumbent)
    if candidate.root_ids is None:
        for stage in ordered_stages:
            _validate_confidence_group(
                stage,
                confidence_group_size,
            )
    else:
        _validate_confidence_group_size(confidence_group_size)

    comparisons: list[PairedComparison] = []
    for stage in ordered_stages:
        if stage.case_count > len(candidate.case_ids):
            return PromotionResult(
                incumbent_artifact_id=incumbent.artifact_id,
                incumbent_digest=incumbent.artifact_digest,
                selected_artifact_id=incumbent.artifact_id,
                selected_digest=incumbent.artifact_digest,
                promoted=False,
                reason="insufficient_evidence",
                comparisons=tuple(comparisons),
            )
        comparison = paired_compare(
            candidate,
            incumbent,
            stage,
            cvar_tail_fraction=cvar_tail_fraction,
            confidence_group_size=confidence_group_size,
        )
        comparisons.append(comparison)
        if not comparison.passed:
            break

    if len(comparisons) != len(ordered_stages) or not comparisons[-1].passed:
        return PromotionResult(
            incumbent_artifact_id=incumbent.artifact_id,
            incumbent_digest=incumbent.artifact_digest,
            selected_artifact_id=incumbent.artifact_id,
            selected_digest=incumbent.artifact_digest,
            promoted=False,
            reason="challenger_did_not_clear_evidence",
            comparisons=tuple(comparisons),
        )

    return PromotionResult(
        incumbent_artifact_id=incumbent.artifact_id,
        incumbent_digest=incumbent.artifact_digest,
        selected_artifact_id=candidate.artifact_id,
        selected_digest=candidate.artifact_digest,
        promoted=True,
        reason="candidate_cleared_all_evidence_stages",
        comparisons=tuple(comparisons),
    )


def _validate_pairing(
    candidate: MacroCandidateScore,
    incumbent: MacroCandidateScore,
) -> None:
    if candidate.target_sha256 != incumbent.target_sha256:
        raise ValueError("candidate and incumbent target digests must match exactly")
    if candidate.contract_id != incumbent.contract_id:
        raise ValueError("candidate and incumbent macro contracts must match exactly")
    if candidate.case_ids != incumbent.case_ids:
        raise ValueError("candidate and incumbent case order must match exactly")
    if len(candidate.errors) != len(incumbent.errors):
        raise ValueError("candidate and incumbent error counts must match exactly")
    if candidate.root_ids != incumbent.root_ids:
        raise ValueError("candidate and incumbent root lineages must match exactly")


def _validate_stages(stages: Sequence[EvidenceStage]) -> tuple[EvidenceStage, ...]:
    materialized = tuple(stages)
    if not materialized:
        raise ValueError("at least one evidence stage is required")
    counts = tuple(stage.case_count for stage in materialized)
    if any(current <= previous for previous, current in zip(counts, counts[1:])):
        raise ValueError("evidence stage case counts must be strictly increasing")
    return materialized
