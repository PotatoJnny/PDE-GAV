"""Backend-neutral reuse of an allocation's idle GPUs for hidden evaluation."""

from contextlib import contextmanager
from threading import Lock
from time import perf_counter


_SCOPE_LOCKS_LOCK = Lock()


class HiddenEvaluationResourceScope:
    """Temporarily bind a Backend's numerical pool to healthy allocated devices.

    The Controller calls this only at a hidden-evaluation phase boundary, once
    Agent leases have settled. No resources are requested from a scheduler.
    ``pool_factory(cases=..., devices=...)`` returns an unstarted pool exposing
    ``start()``, ``close()`` and ``stats()``; the latter must report
    ``cleanup_complete=True`` after successful cleanup. ``activate(pool)`` and
    ``deactivate()`` only install/remove the Backend's invocation binding.

    Sampling, plans, truth, scoring and promotion remain Controller-owned.
    Different scope objects sharing an execution service cannot overlap.
    """

    def __init__(self, candidate_execution, *, devices, pool_factory, activate,
                 deactivate, events=None):
        self.execution = candidate_execution
        self.devices = tuple(devices)
        if (not self.devices or any(not isinstance(device, str) or not device for device in self.devices)
                or len(set(self.devices)) != len(self.devices)):
            raise ValueError("evaluation GPU devices must be nonempty and distinct")
        if not all(callable(callback) for callback in (pool_factory, activate, deactivate)):
            raise TypeError("evaluation pool factory and binding callbacks must be callable")
        self.pool_factory = pool_factory
        self._activate, self._deactivate = activate, deactivate
        self.events = events
        self.last_report = None
        # Composition is normally serial, but sharing this lock also prevents
        # independently configured Backends from racing the same host count.
        with _SCOPE_LOCKS_LOCK:
            lock = getattr(candidate_execution, "_hidden_evaluation_resource_lock", None)
            if lock is None:
                lock = Lock()
                candidate_execution._hidden_evaluation_resource_lock = lock
            self._scope_lock = lock

    def _emit(self, event_type, message, payload):
        if self.events is not None:
            self.events.emit(event_type, message, payload=payload)

    @contextmanager
    def __call__(self, *, cases, unavailable_gpu_devices=()):
        if not self._scope_lock.acquire(blocking=False):
            raise RuntimeError("hidden GPU evaluation scopes cannot overlap")
        try:
            blocked = set(unavailable_gpu_devices)
            devices = tuple(device for device in self.devices if device not in blocked)
            if not devices:
                raise RuntimeError("no healthy GPU is available for hidden evaluation")
            cases = tuple(cases)
            if not cases:
                yield None
                return
            previous_count = self.execution.oracle_host_count
            pool = self.pool_factory(cases=cases, devices=devices)
            started = perf_counter()
            primary_error = None
            try:
                self._emit("evaluation_gpus_starting", "Reusing idle GPUs for hidden evaluation",
                           {"gpu_devices": devices, "case_count": len(cases), "new_allocation": False})
                pool.start()
                self._activate(pool)
                self.execution.oracle_host_count = len(devices)
                self._emit("evaluation_gpus_ready", "Parallel evaluation Oracle workers are ready",
                           {"gpu_devices": devices, "oracle_workers": len(devices),
                            "startup_wall_seconds": perf_counter() - started})
                yield pool
            except BaseException as error:
                primary_error = error
            finally:
                cleanup_errors = []
                # Always attempt all three steps; a failed pool or binding
                # cleanup must not leave a stale host count in the next phase.
                for cleanup in (
                    pool.close,
                    self._deactivate,
                    lambda: setattr(self.execution, "oracle_host_count", previous_count),
                ):
                    try:
                        cleanup()
                    except BaseException as error:
                        cleanup_errors.append(error)
                try:
                    snapshot = dict(pool.stats())
                except BaseException as error:
                    cleanup_errors.append(error)
                    snapshot = {"cleanup_complete": False}
                if snapshot.get("cleanup_complete") is not True:
                    cleanup_errors.append(RuntimeError("evaluation GPU cleanup was not confirmed"))
                self.last_report = {"gpu_devices": devices,
                                    "wall_seconds": perf_counter() - started,
                                    "pool": snapshot}
                released = not cleanup_errors
                try:
                    self._emit("evaluation_gpus_released" if released else "evaluation_gpu_cleanup_failed",
                               "Evaluation GPUs released" if released else "Evaluation GPU cleanup failed",
                               self.last_report)
                except BaseException as error:
                    cleanup_errors.append(error)
                for error in cleanup_errors:
                    if primary_error is None:
                        primary_error = error
                    else:
                        add_note = getattr(primary_error, "add_note", None)
                        if callable(add_note):
                            add_note(f"Evaluation resource cleanup also failed: {type(error).__name__}: {error}")
            if primary_error is not None:
                raise primary_error
        finally:
            self._scope_lock.release()
