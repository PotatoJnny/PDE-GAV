"""Backend-neutral resource policy tested with software-only numerical pools."""

from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import HiddenEvaluationResourceScope


class FakePool:
    def __init__(self, *, cases, devices, fail=None):
        self.cases, self.devices, self.fail = cases, devices, fail
        self.started = self.closed = False

    def start(self):
        self.started = True
        if self.fail == "start":
            raise RuntimeError("pool start failed")

    def close(self):
        self.closed = True
        if self.fail == "close":
            raise RuntimeError("pool close failed")

    def stats(self):
        return {"started": self.started, "closed": self.closed,
                "cleanup_complete": self.closed and self.fail != "unconfirmed"}


def make_scope(execution=None, *, backend_name="fake", fail=None):
    execution = execution or SimpleNamespace(oracle_host_count=1)
    backend = SimpleNamespace(name=backend_name, pool=None)
    events, created = [], []

    def factory(**kwargs):
        pool = FakePool(**kwargs, fail=fail)
        created.append(pool)
        return pool

    def activate(pool):
        backend.pool = pool
        if fail == "activate":
            raise RuntimeError("binding activate failed")

    def deactivate():
        backend.pool = None
        if fail == "deactivate":
            raise RuntimeError("binding deactivate failed")

    scope = HiddenEvaluationResourceScope(
        execution, devices=("2", "0", "1"), pool_factory=factory,
        activate=activate, deactivate=deactivate,
        events=SimpleNamespace(emit=lambda name, message, **kwargs: events.append((name, kwargs["payload"]))),
    )
    return scope, execution, backend, events, created


@pytest.mark.parametrize("backend_name", ["fluid-fake", "kinetic-fake", "reduced-model-fake"])
def test_distinct_backends_share_the_same_resource_policy(backend_name):
    scope, execution, backend, events, created = make_scope(backend_name=backend_name)
    execution.oracle_host_count = 4
    cases = (object(), object())
    with scope(cases=cases, unavailable_gpu_devices=("0",)) as pool:
        assert backend.pool is pool
        assert pool.cases == cases and pool.devices == ("2", "1")
        assert pool.started and execution.oracle_host_count == 2
    assert backend.pool is None and execution.oracle_host_count == 4
    assert created == [pool] and pool.closed
    assert [event[0] for event in events] == [
        "evaluation_gpus_starting", "evaluation_gpus_ready", "evaluation_gpus_released",
    ]
    assert events[0][1] == {"gpu_devices": ("2", "1"), "case_count": 2, "new_allocation": False}
    assert scope.last_report["pool"]["cleanup_complete"] is True
    assert events[-1][1] is scope.last_report


@pytest.mark.parametrize("failure", ["start", "activate", "body", "close", "deactivate", "unconfirmed"])
def test_failed_pool_or_candidate_restores_the_execution_boundary(failure):
    scope, execution, backend, events, created = make_scope(fail=failure)
    with pytest.raises(RuntimeError):
        with scope(cases=(object(),)):
            if failure == "body":
                raise RuntimeError("candidate failed")
    assert backend.pool is None and execution.oracle_host_count == 1
    assert created[0].closed
    assert not scope._scope_lock.locked()
    cleanup_failed = failure in {"close", "deactivate", "unconfirmed"}
    assert events[-1][0] == (
        "evaluation_gpu_cleanup_failed" if cleanup_failed else "evaluation_gpus_released"
    )


def test_separate_backend_scopes_cannot_overlap_one_execution_service():
    execution = SimpleNamespace(oracle_host_count=1)
    first, _, _, _, _ = make_scope(execution, backend_name="one")
    second, _, second_backend, _, second_pools = make_scope(execution, backend_name="two")
    with first(cases=(object(),)):
        with pytest.raises(RuntimeError, match="overlap"):
            with second(cases=(object(),)):
                pytest.fail("overlapping pool entered")
        assert second_pools == [] and second_backend.pool is None
        assert execution.oracle_host_count == 3
    with second(cases=(object(),)):
        assert execution.oracle_host_count == 3
    assert execution.oracle_host_count == 1


def test_quarantined_or_empty_work_never_starts_a_pool():
    scope, execution, _, events, created = make_scope()
    with pytest.raises(RuntimeError, match="no healthy GPU"):
        with scope(cases=(object(),), unavailable_gpu_devices=("0", "1", "2")):
            pytest.fail("unavailable GPUs admitted")
    with scope(cases=()) as pool:
        assert pool is None
    assert created == [] and events == [] and execution.oracle_host_count == 1


def test_cleanup_error_does_not_hide_candidate_failure():
    scope, execution, backend, events, _ = make_scope(fail="close")
    with pytest.raises(LookupError, match="candidate evidence") as caught:
        with scope(cases=(object(),)):
            raise LookupError("candidate evidence")
    assert any("pool close failed" in note for note in caught.value.__notes__)
    assert backend.pool is None and execution.oracle_host_count == 1
    assert events[-1][0] == "evaluation_gpu_cleanup_failed"


def test_cleanup_preserves_primary_exception_when_add_note_is_unavailable():
    class LegacyError(LookupError):
        add_note = None  # Model Python 3.10's BaseException interface.

    scope, execution, backend, events, _ = make_scope(fail="close")
    original = LegacyError("candidate evidence")
    with pytest.raises(LegacyError) as caught:
        with scope(cases=(object(),)):
            raise original
    assert caught.value is original
    assert backend.pool is None and execution.oracle_host_count == 1
    assert not scope._scope_lock.locked()
    assert events[-1][0] == "evaluation_gpu_cleanup_failed"
