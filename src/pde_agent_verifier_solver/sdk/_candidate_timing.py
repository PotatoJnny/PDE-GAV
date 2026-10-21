"""Private timing hooks bound by trusted candidate execution services.

Only waiting for a controller-owned queue is excluded. Candidate computation,
public-oracle computation, and ordinary I/O remain elapsed wall time. These
hooks provide attribution; the runner remains responsible for enforcement.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator


_queue_wait_observer: ContextVar[Callable[[bool], None] | None] = ContextVar(
    "candidate_controller_queue_wait_observer", default=None
)
_queue_wait_cancel_check: ContextVar[Callable[[], None] | None] = ContextVar(
    "candidate_controller_queue_cancel_check", default=None
)
_queue_wait_reader: ContextVar[Callable[[], float] | None] = ContextVar(
    "candidate_controller_queue_wait_reader", default=None
)


@contextmanager
def bind_queue_wait_observer(
    callback: Callable[[bool], None],
    *, check_cancelled: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Bind queue begin/end notifications to this execution context."""

    token = _queue_wait_observer.set(callback)
    cancel_token = _queue_wait_cancel_check.set(check_cancelled)
    try:
        yield
    finally:
        _queue_wait_cancel_check.reset(cancel_token)
        _queue_wait_observer.reset(token)


@contextmanager
def controller_queue_wait() -> Iterator[Callable[[], None]]:
    """Mark only acquisition of a trusted controller lock, not work under it."""

    observer = _queue_wait_observer.get()
    check_cancelled = _queue_wait_cancel_check.get() or (lambda: None)
    check_cancelled()
    if observer is not None:
        observer(True)
    try:
        yield check_cancelled
    finally:
        if observer is not None:
            observer(False)


@contextmanager
def bind_queue_wait_reader(reader: Callable[[], float]) -> Iterator[None]:
    """Bind a trusted cumulative wait counter, restoring any outer binding."""

    token = _queue_wait_reader.set(reader)
    try:
        yield
    finally:
        _queue_wait_reader.reset(token)


def controller_queue_wait_seconds() -> float:
    """Return cumulative trusted queue wait, or zero outside a bound runner."""

    reader = _queue_wait_reader.get()
    return float(reader()) if reader is not None else 0.0
