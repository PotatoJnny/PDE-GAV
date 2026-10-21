"""GPU numerical admission preserves fairness, cancellation and device overlap."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSCUDAQueryAdmission
from pde_agent_verifier_solver.sdk._candidate_timing import bind_queue_wait_observer


def test_waiting_gpu_calls_enter_in_order_and_other_device_runs():
    admission = _NSCUDAQueryAdmission()
    first_queued, second_queued = Event(), Event()
    entered = []
    waits = []

    def query(label, queued):
        def observed(waiting):
            waits.append((label, waiting))
            if waiting:
                queued.set()
        with bind_queue_wait_observer(observed):
            with admission.for_device(0):
                entered.append(label)

    with ThreadPoolExecutor(max_workers=3) as workers:
        with admission.for_device(0):
            first = workers.submit(query, "first", first_queued)
            assert first_queued.wait(3)
            second = workers.submit(query, "second", second_queued)
            assert second_queued.wait(3)
            with admission.for_device(1):
                entered.append("other-device")
            assert entered == ["other-device"]
        first.result(timeout=3)
        second.result(timeout=3)
    assert entered == ["other-device", "first", "second"]
    assert waits == [("first", True), ("second", True), ("first", False), ("second", False)]
    assert not admission._active and not admission._queues


def test_cancelled_waiter_is_removed_without_losing_later_waiters():
    admission = _NSCUDAQueryAdmission()
    queued, cancel, later_queued = Event(), Event(), Event()
    waits = []

    def cancelled():
        def observe(waiting):
            waits.append(waiting)
            if waiting:
                queued.set()
        def check():
            if cancel.is_set():
                raise RuntimeError("Case revoked")
        with bind_queue_wait_observer(observe, check_cancelled=check):
            with admission.for_device(0):
                pytest.fail("cancelled call ran")

    def later():
        with bind_queue_wait_observer(lambda waiting: later_queued.set() if waiting else None):
            with admission.for_device(0):
                return "ran"

    with ThreadPoolExecutor(max_workers=2) as workers:
        with admission.for_device(0):
            future = workers.submit(cancelled)
            assert queued.wait(3)
            cancel.set()
            with pytest.raises(RuntimeError, match="revoked"):
                future.result(timeout=3)
            next_call = workers.submit(later)
            assert later_queued.wait(3)
        assert next_call.result(timeout=3) == "ran"
    assert waits == [True, False]
    assert not admission._active and not admission._queues


def test_failed_numerical_work_releases_device():
    admission = _NSCUDAQueryAdmission()
    with pytest.raises(ValueError, match="scientific failure"):
        with admission.for_device(0):
            raise ValueError("scientific failure")
    with admission.for_device(0):
        pass
    assert not admission._active and not admission._queues


def test_sampler_uses_the_same_device_admission(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires scheduled CUDA allocation")
    from threading import Lock
    from types import SimpleNamespace
    import numpy as np
    from pde_agent_verifier_solver.plugins.ns import structured_flow_runtime as flow

    admission = _NSCUDAQueryAdmission()
    monkeypatch.setattr(flow, "_NS_CUDA_QUERY_ADMISSION", admission)
    sampled, queued = Event(), Event()
    def sample(count, *, initial_values, **kwargs):
        sampled.set()
        return initial_values.clone()
    sampler = flow.LegacyStructuredNSFlowSampler.__new__(flow.LegacyStructuredNSFlowSampler)
    sampler._torch, sampler._device = torch, torch.device("cuda")
    sampler._lock, sampler._cuda_graph_sampling = Lock(), False
    sampler._steps, sampler._method = 1, "euler"
    sampler._model = SimpleNamespace(sample=sample)
    sampler.pca = SimpleNamespace(rank=3, decode=lambda scores: scores)
    sampler._field_from_coefficients = lambda coefficients: coefficients.detach().cpu().numpy()

    def run():
        with bind_queue_wait_observer(lambda waiting: queued.set() if waiting else None):
            return sampler.sample_many((101, 102))

    with ThreadPoolExecutor(max_workers=1) as workers:
        with admission.for_device(torch.cuda.current_device()):
            result = workers.submit(run)
            assert queued.wait(3)
            assert not sampled.is_set()
        values = result.result(timeout=10)
    assert sampled.is_set() and len(values) == 2
    for value, repeated in zip(values, sampler.sample_many((101, 102)), strict=True):
        np.testing.assert_array_equal(value, repeated)
    assert not admission._active and not admission._queues


@pytest.mark.parametrize("cancelled", [False, True])
def test_device_wait_reaches_controller_deadline_ledger_and_drains(cancelled):
    """Exercise the production science-thread observer, not a test-only meter."""
    from time import monotonic, sleep
    from types import SimpleNamespace
    from pde_agent_verifier_solver.engine.candidate_execution import _CandidateOracleState

    admission = _NSCUDAQueryAdmission()
    entered = Event()
    def invoke(case_id, method, args, kwargs):
        entered.set()
        with admission.for_device(0):
            return 7
    codec = SimpleNamespace(encode=lambda value: value, decode=lambda value: value)
    state = _CandidateOracleState(codec, ("marker",), invoke, {"token": "case"})
    try:
        with admission.for_device(0):
            state.submit("token", "call", {"case_id": "case", "method": "marker",
                                             "args": (), "kwargs": {}})
            assert entered.wait(3)
            deadline = monotonic() + 3
            while state.queue_wait_seconds("case") < .05 and monotonic() < deadline:
                sleep(.002)
            credited = state.queue_wait_seconds("case")
            assert credited >= .05
            assert state.execution_metrics("case")["oracle_service_wall_seconds"] < credited / 2
            if cancelled:
                state.revoke_case("case")
                assert state.wait_case_drained("case", 3)
        assert state.wait_all_drained(3)
        if not cancelled:
            assert state.poll("token", "call")["value"] == 7
            assert state.queue_wait_seconds("case") >= credited
    finally:
        state.revoke_all()
        assert state.stop_worker(3)
    assert not admission._active and not admission._queues


def test_unknown_field_capture_lock_wait_is_credited_and_cancellable():
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSCUDAWorkGuard

    guard = _NSCUDAWorkGuard()
    queued, cancel = Event(), Event()
    observed = []
    def writer():
        def observe(waiting):
            observed.append(waiting)
            if waiting:
                queued.set()
        def check():
            if cancel.is_set():
                raise RuntimeError("revoked")
        with bind_queue_wait_observer(observe, check_cancelled=check):
            with guard:
                pytest.fail("revoked writer acquired capture lock")
    with ThreadPoolExecutor(max_workers=1) as workers:
        with guard.shared_query():
            result = workers.submit(writer)
            assert queued.wait(3)
            cancel.set()
            with pytest.raises(RuntimeError, match="revoked"):
                result.result(timeout=3)
    assert observed == [True, False]
    assert not guard._waiting_writers
    assert guard.acquire(False)
    guard.release()


def test_capture_lock_timeout_releases_wait_priority():
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSCUDAWorkGuard

    guard = _NSCUDAWorkGuard()
    with ThreadPoolExecutor(max_workers=1) as workers:
        with guard.shared_query():
            assert workers.submit(guard.acquire, True, .01).result(timeout=3) is False
            assert workers.submit(guard.acquire, False).result(timeout=3) is False
    assert not guard._waiting_writers
    assert guard.acquire(False)
    guard.release()
