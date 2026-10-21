"""Independent safe public sessions overlap without weakening plugin isolation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
import time

import pytest

from pde_agent_verifier_solver.engine.evaluator import _PersistentPublicCaseSession
from tests.test_candidate_execution import _case
from tests.test_canonical_candidate_boundary import _generator_state


def _session(index, shared_lock, opener):
    case = _case(f"parallel-{index}", value=float(index))
    return _PersistentPublicCaseSession(
        opener=opener, generator_state=_generator_state(),
        controller_case=case, candidate_case=case, methods=("query",),
        operation_lock=shared_lock,
    )


def test_safe_independent_sessions_overlap_but_open_under_generator_lock():
    shared = Lock()
    rendezvous = Barrier(2)
    closed = []

    class SafeSession:
        public_calls_thread_safe = True

        def query(self, value):
            rendezvous.wait(timeout=3.0)
            return value + 1

        def close(self):
            closed.append(self)

    def opener(*_args):
        assert not shared.acquire(blocking=False), "session creation escaped Generator lock"
        return SafeSession()

    sessions = [_session(i, shared, opener) for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(s.invoke, "query", (i,), {}) for i, s in enumerate(sessions)]
        assert [f.result(timeout=4.0) for f in pending] == [1, 2]
    shared.acquire()
    try:
        for s in sessions:
            s.close()
        assert len(closed) == 2, "safe session cleanup should not wait on unrelated work"
    finally:
        shared.release()


@pytest.mark.parametrize("capability", [None, False, 1, "true"])
def test_unknown_or_nonboolean_capability_retains_shared_serialization(capability):
    shared = Lock()
    counter_lock = Lock()
    active = peak = 0

    class Session:
        public_calls_thread_safe = capability

        def query(self, value):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(active, peak)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
            return value

    sessions = [_session(i, shared, lambda *_: Session()) for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(s.invoke, "query", (i,), {}) for i, s in enumerate(sessions)]
        assert [f.result(timeout=3) for f in pending] == [0, 1, 2]
    assert peak == 1
    for s in sessions:
        s.close()


def test_one_safe_session_remains_serialized_and_closes_after_inflight_call():
    entered, release, closed = Event(), Event(), Event()
    calls = []

    class Session:
        public_calls_thread_safe = True

        def query(self, value):
            calls.append(value)
            entered.set()
            assert release.wait(3.0)
            return value

        def close(self):
            closed.set()

    session = _session(0, Lock(), lambda *_: Session())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(session.invoke, "query", (1,), {})
        assert entered.wait(2)
        second = pool.submit(session.invoke, "query", (2,), {})
        session.close()
        assert not closed.is_set()
        release.set()
        assert first.result(timeout=3) == 1
        with pytest.raises(RuntimeError, match="no longer active"):
            second.result(timeout=3)
    assert calls == [1]
    assert closed.wait(1)
