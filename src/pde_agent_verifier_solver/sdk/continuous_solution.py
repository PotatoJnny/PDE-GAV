"""Backend-neutral two-phase ABI for continuously accessible solutions.

The candidate first prepares a solution without receiving any scoring query::

    handle = candidate.prepare_solution(case, solve_budget)

Only after that call has completed may the controller reveal a point query::

    answer = handle.evaluate(point_query)

The handle owns projection from any candidate-selected grid, basis, surrogate,
or other internal representation to the requested physical points.  This ABI
therefore contains no equation-specific dimensions and no protocol-selected
candidate grid.

``measure_prepare_solution`` and ``measure_point_query`` deliberately time only
the candidate's per-case solve and point projection, respectively.  Campaign
search, guidance training, generator work, and case sampling use separate
campaign/generator budgets and happen outside both clocks.  Controller truth
materialization and matching form a third, controller-owned phase; along with
serialization and reporting, they are absent from candidate timing records.
Trusted controller queue wait is recorded separately and deducted from elapsed
wall time; candidate work and public-oracle computation remain charged.
Hard process deadlines remain the trusted runner's responsibility; these
helpers provide attribution and eligibility records, not process isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from time import perf_counter
from types import MappingProxyType
from typing import Callable, ClassVar, Protocol, Sequence, Mapping

from ._candidate_timing import controller_queue_wait_seconds
from .models import AnswerQuery, AnswerView, Case, QueryKind
from .solve_budget import candidate_solve_clock


__all__ = [
    "BatchSolutionProgram",
    "measure_prepare_solutions",
    "ContinuousSolution",
    "LegacyOneShotAdapter",
    "PointQueryLatencyEligibility",
    "PointQueryLatencyEligibilityContract",
    "PointQueryResult",
    "PreparedSolutionResult",
    "SolveWallEligibility",
    "SolveWallEligibilityContract",
    "TimingAssurance",
    "TwoPhaseEligibility",
    "TwoPhaseEligibilityContract",
    "TwoPhaseSolutionProgram",
    "measure_point_query",
    "measure_prepare_solution",
]


Clock = Callable[[], float]
LegacyOneShotCall = Callable[[Case, AnswerQuery, float | None], AnswerView]


class ContinuousSolution(Protocol):
    """Candidate-owned solution that can answer arbitrary physical points."""

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        """Evaluate a ``QueryKind.POINTS`` query revealed after preparation."""


class TwoPhaseSolutionProgram(Protocol):
    """Candidate ABI whose preparation phase cannot receive scoring points."""

    def prepare_solution(
        self,
        case: Case,
        solve_budget: float | None = None,
    ) -> ContinuousSolution:
        """Build a continuous handle using a candidate-selected representation."""


class BatchSolutionProgram(Protocol):
    """Optional cross-Case preparation; queries stay hidden until all handles exist."""
    def prepare_solutions(self, cases: Sequence[Case], solve_budget: float | None = None
                          ) -> Mapping[str, ContinuousSolution]: ...


def measure_prepare_solutions(program, cases, *, solve_budget=None):
    """Return Case-keyed handles and shared preparation time (never divide by N)."""
    cases = tuple(cases)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("batch preparation needs distinct Cases")
    before = controller_queue_wait_seconds()
    started = perf_counter()
    prepare = getattr(program, "prepare_solutions", None)
    if not callable(prepare):
        raise TypeError("batch solver must define prepare_solutions(cases, solve_budget=None)")
    handles = prepare(cases, solve_budget=solve_budget)
    if not isinstance(handles, Mapping) or set(handles) != {case.case_id for case in cases}:
        raise ValueError("prepare_solutions must return exactly one handle per Case id")
    for handle in handles.values():
        _require_callable_evaluate(handle)
    elapsed = max(0., perf_counter() - started - (controller_queue_wait_seconds() - before))
    return dict(handles), elapsed


class TimingAssurance(str, Enum):
    """Whether solve/query attribution follows the genuine two-phase boundary.

    This describes timing attribution only.  It is not an execution-isolation
    or hidden-data security attestation.
    """

    TWO_PHASE_METERED = "two_phase_metered"
    LEGACY_ONE_SHOT_UNASSURED = "legacy_one_shot_unassured"


@dataclass(frozen=True)
class PreparedSolutionResult:
    """A prepared handle and elapsed solve wall time excluding trusted queue wait."""

    handle: ContinuousSolution
    candidate_solve_wall_seconds: float
    solve_budget_seconds: float | None
    timing_assurance: TimingAssurance = TimingAssurance.TWO_PHASE_METERED
    candidate_solve_queue_wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        _require_callable_evaluate(self.handle)
        _require_nonnegative_seconds(
            self.candidate_solve_wall_seconds,
            "candidate_solve_wall_seconds",
        )
        _require_nonnegative_seconds(
            self.candidate_solve_queue_wait_seconds,
            "candidate_solve_queue_wait_seconds",
        )
        _require_optional_positive_seconds(
            self.solve_budget_seconds,
            "solve_budget_seconds",
        )
        if not isinstance(self.timing_assurance, TimingAssurance):
            raise TypeError("timing_assurance must be a TimingAssurance")

    @property
    def formal_timing_assurance(self) -> bool:
        """Whether this preparation can enter formal timing eligibility."""

        return self.timing_assurance is TimingAssurance.TWO_PHASE_METERED


@dataclass(frozen=True)
class PointQueryResult:
    """One finite answer and elapsed query wall time excluding trusted queue wait."""

    answer: AnswerView
    candidate_query_wall_seconds: float
    point_count: int
    candidate_query_queue_wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.answer, AnswerView):
            raise TypeError("answer must be an AnswerView")
        _require_nonnegative_seconds(
            self.candidate_query_wall_seconds,
            "candidate_query_wall_seconds",
        )
        _require_nonnegative_seconds(
            self.candidate_query_queue_wait_seconds,
            "candidate_query_queue_wait_seconds",
        )
        if isinstance(self.point_count, bool) or not isinstance(self.point_count, int):
            raise TypeError("point_count must be an integer")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")

    @property
    def candidate_query_wall_seconds_per_point(self) -> float:
        """Latency normalized by requested physical points, not grid size."""

        return self.candidate_query_wall_seconds / self.point_count


@dataclass(frozen=True)
class SolveWallEligibility:
    """Solve-wall assessment kept separate from query latency."""

    eligible: bool
    candidate_solve_wall_seconds: float
    maximum_candidate_solve_wall_seconds: float


@dataclass(frozen=True)
class PointQueryLatencyEligibility:
    """Aggregate point-normalized query-latency assessment."""

    eligible: bool
    candidate_query_wall_seconds: float
    point_count: int
    candidate_query_wall_seconds_per_point: float
    maximum_candidate_query_wall_seconds_per_point: float


@dataclass(frozen=True)
class SolveWallEligibilityContract:
    """Independent eligibility limit for candidate preparation."""

    maximum_candidate_solve_wall_seconds: float

    def __post_init__(self) -> None:
        _require_positive_seconds(
            self.maximum_candidate_solve_wall_seconds,
            "maximum_candidate_solve_wall_seconds",
        )

    def assess(self, prepared: PreparedSolutionResult) -> SolveWallEligibility:
        if not isinstance(prepared, PreparedSolutionResult):
            raise TypeError("prepared must be a PreparedSolutionResult")
        observed = prepared.candidate_solve_wall_seconds
        limit = self.maximum_candidate_solve_wall_seconds
        return SolveWallEligibility(observed <= limit, observed, limit)


@dataclass(frozen=True)
class PointQueryLatencyEligibilityContract:
    """Independent latency limit normalized over all requested points.

    Aggregating wall time and point count before division prevents the result
    from depending on how the controller batches one fixed scoring cohort.
    """

    maximum_candidate_query_wall_seconds_per_point: float

    def __post_init__(self) -> None:
        _require_positive_seconds(
            self.maximum_candidate_query_wall_seconds_per_point,
            "maximum_candidate_query_wall_seconds_per_point",
        )

    def assess(
        self,
        queries: Sequence[PointQueryResult],
    ) -> PointQueryLatencyEligibility:
        results = tuple(queries)
        if not results:
            raise ValueError("query latency eligibility needs at least one query")
        if any(not isinstance(result, PointQueryResult) for result in results):
            raise TypeError("queries must contain only PointQueryResult values")
        wall_seconds = math.fsum(
            result.candidate_query_wall_seconds for result in results
        )
        point_count = sum(result.point_count for result in results)
        seconds_per_point = wall_seconds / point_count
        limit = self.maximum_candidate_query_wall_seconds_per_point
        return PointQueryLatencyEligibility(
            eligible=seconds_per_point <= limit,
            candidate_query_wall_seconds=wall_seconds,
            point_count=point_count,
            candidate_query_wall_seconds_per_point=seconds_per_point,
            maximum_candidate_query_wall_seconds_per_point=limit,
        )


@dataclass(frozen=True)
class TwoPhaseEligibility:
    """Combined result that preserves each independent eligibility decision."""

    solve: SolveWallEligibility
    query: PointQueryLatencyEligibility
    timing_assurance: TimingAssurance

    @property
    def formal_timing_assurance(self) -> bool:
        return self.timing_assurance is TimingAssurance.TWO_PHASE_METERED

    @property
    def eligible(self) -> bool:
        return self.formal_timing_assurance and self.solve.eligible and self.query.eligible


@dataclass(frozen=True)
class TwoPhaseEligibilityContract:
    """Compose solve and point-query limits without merging their clocks."""

    solve: SolveWallEligibilityContract
    query: PointQueryLatencyEligibilityContract

    def assess(
        self,
        prepared: PreparedSolutionResult,
        queries: Sequence[PointQueryResult],
    ) -> TwoPhaseEligibility:
        return TwoPhaseEligibility(
            solve=self.solve.assess(prepared),
            query=self.query.assess(queries),
            timing_assurance=prepared.timing_assurance,
        )


def measure_prepare_solution(
    program: TwoPhaseSolutionProgram,
    case: Case,
    solve_budget: float | None = None,
    *,
    clock: Clock = perf_counter,
) -> PreparedSolutionResult:
    """Time candidate ABI resolution and ``prepare_solution``.

    Dynamic resolution of both ``prepare_solution`` and the returned handle's
    ``evaluate`` callable is candidate work and remains inside this clock.

    No ``AnswerQuery`` is accepted by this function, so a caller following the
    ABI cannot reveal hidden scoring coordinates during preparation.
    ``solve_budget`` is a per-case candidate budget; campaign search, guidance
    training, generator execution, and case sampling are not charged here.
    The default raw clock has trusted Controller queue wait subtracted once.
    Supplying ``sdk.solve_budget.candidate_solve_clock`` directly is also
    supported: that clock already excludes the same wait, so this helper does
    not subtract it again. Other injected clocks must measure raw elapsed
    time. Oracle numerical execution remains charged. Runner safety limits
    are enforced separately.
    """

    if not isinstance(case, Case):
        raise TypeError("case must be a Case")
    _require_optional_positive_seconds(solve_budget, "solve_budget")

    queue_wait_started = controller_queue_wait_seconds()
    started = _clock_value(clock)
    prepare = getattr(program, "prepare_solution", None)
    if not callable(prepare):
        raise TypeError("program must define callable prepare_solution")
    handle = prepare(case, solve_budget)
    evaluate = _require_callable_evaluate(handle)
    resolved_handle = _ResolvedContinuousSolution(evaluate)
    finished = _clock_value(clock)
    queue_wait_seconds = max(
        0.0, controller_queue_wait_seconds() - queue_wait_started
    )
    # This is elapsed wall time, with only trusted controller queue wait removed.
    wait_to_subtract = 0.0 if clock is candidate_solve_clock else queue_wait_seconds
    wall_seconds = max(0.0, _elapsed_seconds(started, finished) - wait_to_subtract)
    assurance = (
        TimingAssurance.LEGACY_ONE_SHOT_UNASSURED
        if isinstance(program, LegacyOneShotAdapter)
        else TimingAssurance.TWO_PHASE_METERED
    )
    return PreparedSolutionResult(
        handle=resolved_handle,
        candidate_solve_wall_seconds=wall_seconds,
        solve_budget_seconds=solve_budget,
        timing_assurance=assurance,
        candidate_solve_queue_wait_seconds=queue_wait_seconds,
    )


def measure_point_query(
    handle: ContinuousSolution,
    query: AnswerQuery,
    *,
    clock: Clock = perf_counter,
) -> PointQueryResult:
    """Time only candidate projection at explicit physical points.

    Query validation happens before the clock starts. Candidate callable
    resolution, projection, and canonical materialization of the complete
    finite response all happen inside the candidate query clock. Truth
    generation and matching are intentionally absent; a controller performs
    those after this function returns.
    """

    _require_point_query(query)
    queue_wait_started = controller_queue_wait_seconds()
    started = _clock_value(clock)
    evaluate = _require_callable_evaluate(handle)
    answer = _materialize_point_answer(query, evaluate(query))
    finished = _clock_value(clock)
    queue_wait_seconds = max(
        0.0, controller_queue_wait_seconds() - queue_wait_started
    )
    # This is elapsed wall time, with only trusted controller queue wait removed.
    wait_to_subtract = 0.0 if clock is candidate_solve_clock else queue_wait_seconds
    wall_seconds = max(0.0, _elapsed_seconds(started, finished) - wait_to_subtract)
    return PointQueryResult(
        answer=answer,
        candidate_query_wall_seconds=wall_seconds,
        point_count=len(query.points),
        candidate_query_queue_wait_seconds=queue_wait_seconds,
    )


@dataclass(frozen=True)
class LegacyOneShotAdapter:
    """Compatibility shim for a query-aware one-shot callable.

    Preparation only stores inputs.  The complete legacy solve runs later in
    ``evaluate``, after the query is visible, and repeated queries repeat that
    solve.  Consequently this adapter is useful for migration and development
    comparisons but **never** satisfies formal two-phase timing assurance.
    """

    one_shot: LegacyOneShotCall

    formal_timing_assurance: ClassVar[bool] = False
    timing_assurance: ClassVar[TimingAssurance] = (
        TimingAssurance.LEGACY_ONE_SHOT_UNASSURED
    )

    def __post_init__(self) -> None:
        if not callable(self.one_shot):
            raise TypeError("one_shot must be callable")

    def prepare_solution(
        self,
        case: Case,
        solve_budget: float | None = None,
    ) -> ContinuousSolution:
        _require_optional_positive_seconds(solve_budget, "solve_budget")
        return _LegacyOneShotHandle(self.one_shot, case, solve_budget)


@dataclass(frozen=True)
class _LegacyOneShotHandle:
    one_shot: LegacyOneShotCall
    case: Case
    solve_budget: float | None

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        _require_point_query(query)
        return self.one_shot(self.case, query, self.solve_budget)



@dataclass(frozen=True)
class _ResolvedContinuousSolution:
    """Controller-owned wrapper around the callable resolved during solve time."""

    evaluate_callable: Callable[[AnswerQuery], AnswerView]

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        return self.evaluate_callable(query)

def _require_point_query(query: AnswerQuery) -> None:
    if not isinstance(query, AnswerQuery):
        raise TypeError("query must be an AnswerQuery")
    if query.kind is not QueryKind.POINTS:
        raise ValueError("continuous solutions accept only QueryKind.POINTS queries")
    if len(set(query.fields)) != len(query.fields):
        raise ValueError("point-query fields must be unique")


def _materialize_point_answer(query: AnswerQuery, answer: object) -> AnswerView:
    """Copy a candidate response into finite controller-owned tuples."""

    if not isinstance(answer, AnswerView):
        raise TypeError("continuous solution evaluate() must return an AnswerView")
    if answer.kind is not QueryKind.POINTS:
        raise ValueError("continuous solution must return a POINTS AnswerView")
    candidate_points = tuple(answer.points)
    if candidate_points != query.points:
        raise ValueError("continuous solution must echo the requested physical points")
    field_names = tuple(answer.fields)
    if set(field_names) != set(query.fields) or len(field_names) != len(query.fields):
        raise ValueError("continuous solution must return exactly the requested fields")
    shape = tuple(answer.shape)
    point_count = len(query.points)
    if not shape or shape[0] != point_count:
        raise ValueError("point-answer shape must start with the requested point count")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in shape
    ):
        raise ValueError("point-answer shape dimensions must be positive integers")
    flattened_size = math.prod(shape)
    materialized_fields: dict[str, tuple[float, ...]] = {}
    for field_name in query.fields:
        try:
            values = tuple(float(value) for value in answer.fields[field_name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"point-answer field {field_name!r} must be a finite numerical sequence"
            ) from exc
        if len(values) != flattened_size:
            raise ValueError(
                "each returned field must have the flattened point-answer shape"
            )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("point-answer fields must contain only finite values")
        materialized_fields[field_name] = values
    return AnswerView(
        kind=QueryKind.POINTS,
        fields=MappingProxyType(materialized_fields),
        points=query.points,
        shape=shape,
    )


def _require_callable_evaluate(handle: object) -> Callable[[AnswerQuery], AnswerView]:
    evaluate = getattr(handle, "evaluate", None)
    if not callable(evaluate):
        raise TypeError("prepared solution handle must define callable evaluate")
    return evaluate


def _clock_value(clock: Clock) -> float:
    value = clock()
    _require_nonnegative_seconds(value, "clock value")
    return float(value)


def _elapsed_seconds(started: float, finished: float) -> float:
    elapsed = finished - started
    if elapsed < 0.0:
        raise RuntimeError("wall clock moved backwards")
    return elapsed


def _require_optional_positive_seconds(value: float | None, label: str) -> None:
    if value is not None:
        _require_positive_seconds(value, label)


def _require_positive_seconds(value: float, label: str) -> None:
    _require_nonnegative_seconds(value, label)
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be positive")


def _require_nonnegative_seconds(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
