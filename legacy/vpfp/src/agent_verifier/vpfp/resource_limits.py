from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class SolverWallTimeLimitExceeded(TimeoutError):
    pass


@contextmanager
def wall_time_limit(seconds: float | None) -> Iterator[None]:
    """Interrupt one in-process solver call after a broad wall-time limit."""

    if seconds is None:
        yield
        return
    limit = float(seconds)
    if limit <= 0.0:
        raise ValueError("wall-time limit must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("wall-time limits require evaluation on the main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def _expired(_signum, _frame) -> None:
        raise SolverWallTimeLimitExceeded(
            f"solver exceeded {limit:.3f} wall-time seconds"
        )

    signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0.0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_remaining - elapsed, 1.0e-9),
                previous_interval,
            )
