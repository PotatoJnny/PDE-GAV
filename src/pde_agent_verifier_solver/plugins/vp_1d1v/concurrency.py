"""One process-local guard for Artemis's forward-AD numerical runtime.

The installed torch JVP/dual-level bookkeeping is process-global, not protected
by the GIL. Every local entry to the physical runtime must use the same guard,
including private off-grid truth which does not hold the engine Generator lock.
The engine may acquire its own lock first; this module never acquires it. Spawned
hidden-evaluation workers retain independent guards and GPU-batched execution.
"""

from contextlib import contextmanager
from functools import wraps
from threading import RLock
from time import perf_counter

from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait
from pde_agent_verifier_solver.sdk.solve_budget import _record_oracle_lock_wait


# Runtime synchronization only: never attach a lock to scientific checkpoints.
_AD_COMPUTE_LOCK = RLock()


@contextmanager
def artemis_ad_compute():
    """Serialize local numerics; refund only contention, never the computation."""
    acquired = _AD_COMPUTE_LOCK.acquire(blocking=False)
    try:
        if not acquired:
            # Reentrant/uncontended calls take the fast path and publish no wait.
            # Reuse the Controller's live wait ledger so its worker watchdog and
            # the Candidate's RPC budget see the same excluded interval.
            with controller_queue_wait() as check_cancelled:
                started = perf_counter()
                try:
                    while not acquired:
                        acquired = _AD_COMPUTE_LOCK.acquire(timeout=0.1)
                        check_cancelled()
                finally:
                    _record_oracle_lock_wait(perf_counter() - started)
        yield
    finally:
        if acquired:
            _AD_COMPUTE_LOCK.release()


def serialized_artemis_compute(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with artemis_ad_compute():
            return function(*args, **kwargs)
    return guarded


class LocalArtemisSession:
    """Keep the physical session surface, guarding calls into local numerics.

    This wrapper is never applied to a process-pool session: remote GPU lanes
    must remain concurrent. The SDK still enforces its existing method allowlist.
    """

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        value = getattr(self._session, name)
        return serialized_artemis_compute(value) if callable(value) else value
