"""Per-case solve clock excluding only Controller-metered platform waits.

Oracle numerical execution, candidate sleep, and arbitrary network waits remain
charged. The execution service owns the meter; a candidate can read its clock
but must not supply timing claims. This is accounting, not a hostile-code
security boundary. The subprocess runner independently enforces the upstream
charged-wall policy using its live Controller-owned wait ledger; task clocks
and response-side credit cannot extend that limit by themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from threading import Lock
from time import monotonic, perf_counter
from typing import Iterator, Mapping

from ._candidate_timing import bind_queue_wait_reader

ORACLE_WAIT_TIMING_SCHEMA = "canonical-candidate-oracle-wait-v1"


def _seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite nonnegative seconds")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return result


class CandidateSolveBudgetMeter:
    """An execution-owned ledger for one synchronous Candidate/Case call."""

    def __init__(
        self, *, budget_seconds: float | None = None,
        started_at_monotonic: float | None = None,
        batch_mode: bool = False,
    ) -> None:
        if budget_seconds is not None:
            budget_seconds = _seconds(budget_seconds, "budget_seconds")
            if budget_seconds <= 0.0:
                raise ValueError("budget_seconds must be positive")
        if started_at_monotonic is not None:
            started_at_monotonic = _seconds(started_at_monotonic, "started_at_monotonic")
            if budget_seconds is None:
                raise ValueError("started_at_monotonic requires budget_seconds")
        self._batch_mode = batch_mode
        self._budget_seconds = budget_seconds
        self._started_at_monotonic = (
            monotonic() if started_at_monotonic is None else started_at_monotonic
        )
        self._lock = Lock()
        self._queue_wait = 0.0
        self._lock_wait = 0.0
        self._requests: set[str] = set()

    def record_oracle_response(
        self,
        request_id: str,
        timing: Mapping[str, object],
        *,
        rpc_wall_seconds: float,
    ) -> None:
        """Validate Controller response timing, then debit its wait exactly once."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("oracle timing request_id must be nonempty")
        group_wait = timing.get("group_wait_seconds", 0.) if isinstance(timing, Mapping) else 0.
        if self._batch_mode and isinstance(timing, Mapping):
            timing = {key: value for key, value in timing.items() if key != "group_wait_seconds"}
        if not isinstance(timing, Mapping) or set(timing) != {
            "schema", "queue_wait_seconds", "lock_wait_seconds"
        } or timing.get("schema") != ORACLE_WAIT_TIMING_SCHEMA:
            raise ValueError("Candidate oracle wait timing is malformed")
        queue_wait = _seconds(timing["queue_wait_seconds"], "queue_wait_seconds")
        lock_wait = _seconds(timing["lock_wait_seconds"], "lock_wait_seconds")
        rpc_wall = _seconds(rpc_wall_seconds, "rpc_wall_seconds")
        # The clocks live in different processes, but only elapsed durations
        # cross the boundary. Never allow a response to erase local work.
        if queue_wait + lock_wait > rpc_wall:
            raise ValueError("Candidate oracle wait exceeds observed RPC wall")
        with self._lock:
            if request_id in self._requests:
                raise ValueError("Candidate oracle wait timing was already recorded")
            self._requests.add(request_id)
            if self._batch_mode:
                group_wait = _seconds(group_wait, "group_wait_seconds")
                if group_wait > monotonic() - self._started_at_monotonic:
                    raise ValueError("batch wait exceeds execution wall")
                self._queue_wait = max(self._queue_wait, group_wait)
            else:
                self._queue_wait += queue_wait
                self._lock_wait += lock_wait

    def remaining_seconds(self) -> float | None:
        """Read the upstream worker's remaining charged time, never reset it."""
        if self._budget_seconds is None:
            return None
        elapsed = monotonic() - self._started_at_monotonic
        with self._lock:
            charged = max(0.0, elapsed - self._queue_wait - self._lock_wait)
        return max(0.0, self._budget_seconds - charged)

    def snapshot(self) -> Mapping[str, float]:
        with self._lock:
            return {
                "candidate_oracle_queue_wait_seconds": self._queue_wait,
                "candidate_oracle_lock_wait_seconds": self._lock_wait,
                "candidate_platform_wait_seconds": self._queue_wait + self._lock_wait,
                "candidate_oracle_timed_requests": float(len(self._requests)),
            }


_SOLVE_METER: ContextVar[CandidateSolveBudgetMeter | None] = ContextVar(
    "candidate_solve_budget_meter", default=None
)


@contextmanager
def bind_candidate_solve_budget(
    *, new_case: bool = False, budget_seconds: float | None = None,
    started_at_monotonic: float | None = None, batch_mode: bool = False,
) -> Iterator[CandidateSolveBudgetMeter]:
    """Bind one case's meter, reusing nested Task scopes unless ``new_case``.

    Execution services set ``new_case=True`` at the per-case boundary. Tasks
    may bind defensively without resetting an already active worker meter.
    Optional execution-owned limits use the parent's same-host monotonic
    start, so Candidate imports and preparation setup do not receive a fresh
    budget. Nested scopes cannot replace that deadline.
    """

    existing = _SOLVE_METER.get()
    if existing is not None and not new_case:
        yield existing
        return
    meter = CandidateSolveBudgetMeter(
        budget_seconds=budget_seconds, started_at_monotonic=started_at_monotonic, batch_mode=batch_mode,
    )
    token = _SOLVE_METER.set(meter)
    try:
        # The common SDK helpers and cooperative clock consume the same wait
        # total. Nested Task scopes reuse this binding, never sum two meters.
        with bind_queue_wait_reader(
            lambda: meter.snapshot()["candidate_platform_wait_seconds"]
        ):
            yield meter
    finally:
        _SOLVE_METER.reset(token)


def candidate_solve_clock() -> float:
    """Read the cooperative solve clock; outside an execution use real time.

    Use differences of this clock exclusively, never mix it with perf_counter.
    Wait credit arrives synchronously with each validated oracle response.
    The clock binding is local to the execution context: a Candidate-created
    thread must inherit that context explicitly to read this cooperative clock.
    The trusted Oracle client separately retains the worker's meter so responses
    received in a child thread still credit preparation on the main thread.
    The Controller's independent enforced deadline remains authoritative.
    """

    return perf_counter() - candidate_solve_timing()["candidate_platform_wait_seconds"]


def candidate_solve_remaining_seconds() -> float | None:
    """Remaining upstream per-Case budget, or None for an unbounded local call.

    This includes startup, preparation, query, and result serialization in one
    limit. It is a cooperative hint: only the Controller can enforce the
    deadline. A Task may choose a smaller preparation budget but must leave
    time for the remaining phases. Controller wait credit arrives with each
    completed validated Oracle response.
    """

    meter = _SOLVE_METER.get()
    return None if meter is None else meter.remaining_seconds()


def candidate_solve_timing() -> Mapping[str, float]:
    """Read accumulated per-case attribution without exposing a mutable meter."""

    meter = _SOLVE_METER.get()
    return (meter if meter is not None else CandidateSolveBudgetMeter()).snapshot()


def _current_candidate_solve_budget_meter() -> CandidateSolveBudgetMeter | None:
    """Capture the trusted worker meter for an Oracle client, before dispatch."""

    return _SOLVE_METER.get()


def _record_oracle_response_wait(
    request_id: str,
    timing: Mapping[str, object],
    *,
    rpc_wall_seconds: float,
    meter: CandidateSolveBudgetMeter | None = None,
) -> None:
    # A trusted client may receive a serialized Case's RPC on a Candidate-made
    # thread, which does not inherit ContextVars. Retain the same execution
    # meter across that handoff; never create a new Case budget for the thread.
    # Concurrent overlapping same-Case RPCs are not supported by this protocol.
    if meter is not None and not isinstance(meter, CandidateSolveBudgetMeter):
        raise TypeError("Oracle response meter must be execution-owned")
    if meter is None:
        meter = _SOLVE_METER.get()
    # Validate even without a bound candidate (e.g. diagnostic client calls).
    (meter if meter is not None else CandidateSolveBudgetMeter()).record_oracle_response(
        request_id, timing, rpc_wall_seconds=rpc_wall_seconds
    )


@dataclass
class _OracleWait:
    lock_wait_seconds: float = 0.0


_ORACLE_WAIT: ContextVar[_OracleWait | None] = ContextVar(
    "controller_oracle_wait", default=None
)


@contextmanager
def _capture_oracle_wait() -> Iterator[_OracleWait]:
    """Controller-only collector scoped to one allowlisted public RPC."""

    wait = _OracleWait()
    token = _ORACLE_WAIT.set(wait)
    try:
        yield wait
    finally:
        _ORACLE_WAIT.reset(token)


def _record_oracle_lock_wait(seconds: float) -> None:
    wait = _ORACLE_WAIT.get()
    if wait is not None:
        wait.lock_wait_seconds += _seconds(seconds, "oracle lock wait")


__all__ = [
    "bind_candidate_solve_budget",
    "candidate_solve_clock",
    "candidate_solve_remaining_seconds",
    "candidate_solve_timing",
]
