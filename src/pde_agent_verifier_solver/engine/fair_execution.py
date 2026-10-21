"""Small round-robin queues for the existing candidate and Oracle services."""
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition, Thread, current_thread


class FairExecutor:
    """Bound running work; rotate between evaluation requests and their Cases."""
    def __init__(self, max_workers, *, thread_name_prefix="candidate-case"):
        self.max_workers = max_workers
        self._condition = Condition()
        self._queues = {}
        self._order = deque()
        self._closed = False
        self._threads = []
        self._prefix = thread_name_prefix

    def submit_for(self, owner, function, *args, **kwargs):
        future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("execution service is closed")
            if owner not in self._queues:
                self._queues[owner] = deque()
                self._order.append(owner)
            self._queues[owner].append((future, function, args, kwargs))
            if len(self._threads) < self.max_workers:
                thread = Thread(target=self._work, name=f"{self._prefix}-{len(self._threads)}", daemon=True)
                self._threads.append(thread)
                thread.start()
            self._condition.notify()
        return future

    def _work(self):
        while True:
            with self._condition:
                while not self._order and not self._closed:
                    self._condition.wait(.5)
                    if not self._order and not self._closed:
                        self._threads.remove(current_thread())
                        return
                if not self._order:
                    return
                owner = self._order.popleft()
                queue = self._queues[owner]
                future, function, args, kwargs = queue.popleft()
                if queue:
                    self._order.append(owner)
                else:
                    del self._queues[owner]
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._condition:
            self._closed = True
            if cancel_futures:
                for queue in self._queues.values():
                    for future, *_ in queue:
                        future.cancel()
                self._queues.clear()
                self._order.clear()
            self._condition.notify_all()
        if wait:
            for thread in tuple(self._threads):
                thread.join()


class FairAdmission:
    """Round-robin admission shared by all Agent Oracle hosts in one service."""
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("concurrency must be a positive integer")
        self.capacity = capacity
        self._condition = Condition()
        self._queues = {}
        self._order = deque()
        self._active = 0

    @contextmanager
    def enter(self, owner):
        from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait
        ticket = object()
        acquired = False
        with self._condition:
            if owner not in self._queues:
                self._queues[owner] = deque()
                self._order.append(owner)
            self._queues[owner].append(ticket)
            try:
                with controller_queue_wait() as check:
                    while (self._active >= self.capacity or self._order[0] != owner
                           or self._queues[owner][0] is not ticket):
                        check()
                        self._condition.wait(.05)
                    check()
                self._active += 1
                acquired = True
            finally:
                queue = self._queues[owner]
                queue.remove(ticket)
                if acquired:
                    self._order.remove(owner)
                    if queue:
                        self._order.append(owner)
                if not queue:
                    del self._queues[owner]
                    if owner in self._order:
                        self._order.remove(owner)
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


_ORACLE_POLICY = ContextVar("oracle_execution_policy", default=("legacy", 1))


@contextmanager
def bind_oracle_policy(owner, concurrency):
    token = _ORACLE_POLICY.set((owner, concurrency))
    try:
        yield
    finally:
        _ORACLE_POLICY.reset(token)


def current_oracle_policy():
    return _ORACLE_POLICY.get()
