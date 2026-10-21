"""Small CPU-only real-spawn/private-pipe tests, not GPU performance evidence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vp_1d1v.parallel_oracle import (
    ArtemisOracleWorkerError, ArtemisParallelOraclePool, _case_descriptor, _export_case,
)
from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility


def _case(index, *, visibility=CaseVisibility.HIDDEN):
    return Case(str(index), CaseSource.GENERATED, visibility, {
        "equation": {"domain": {"time_interval": [0., 1.]},
                     "oracle_protocol": {"limits": {"max_points_per_request": 4096}}},
        "arrays": {"source": np.zeros((2, 2, 2))},
    })


class _Source:
    def __init__(self, cases):
        self.cases = {case.case_id: case for case in cases}

    def _active_issued(self, case, *, label):
        original = self.cases.get(case.case_id)
        if original is None or case.visibility != original.visibility:
            raise ValueError("inactive or modified issued case")
        return SimpleNamespace(case=original), case is not original


def _fake_export(source, case):
    source._active_issued(case, label="test")
    return {"case": _case_descriptor(case)}


class _FakeResident:
    def __init__(self, settings, device):
        assert device == "cuda:0"
        assert "guidance_adapter_factory" not in settings
        assert "reward_guidance_settings" not in settings
        assert "device" not in settings
        if settings.get("fail_startup"):
            raise ValueError("startup failed")
        self.settings, self.cases = settings, {}

    def register(self, descriptor):
        if self.settings.get("fail_register"):
            raise ValueError("register failed")
        case_id = descriptor["case"]["case_id"]
        assert case_id not in self.cases
        self.cases[case_id] = descriptor

    def open(self, case_id, presented):
        assert case_id in self.cases
        return _FakeSession(self.settings, presented)

    def close(self):
        self.cases.clear()


class _FakeSession:
    def __init__(self, settings, case):
        self.settings = settings
        self.interval = case["public_data"]["equation"]["domain"]["time_interval"]
        self.calls, self.closed = 0, False

    def _query(self, points, width):
        assert not self.closed
        if self.settings.get("exit_during_query"):
            os._exit(17)
        if self.settings.get("delay"):
            import time
            time.sleep(self.settings["delay"])
        if points.ndim != 2 or points.shape[1] != width or not np.isfinite(points).all():
            raise ValueError("invalid points")
        if len(points) > 4096:
            raise ValueError("too many points")
        if width == 3 and np.any(points[:, 0] > self.interval[1]):
            raise ValueError("outside causal interval")
        if self.settings.get("barrier") is not None:
            self.settings["barrier"].wait(timeout=10)
        self.calls += 1
        return points.sum(axis=1)

    def query_source(self, points):
        return self._query(points, 3)

    def query_initial_distribution(self, points):
        return self._query(points, 2)

    @property
    def usage_snapshot(self):
        return {"query_calls": self.calls, "physical_device": os.environ["CUDA_VISIBLE_DEVICES"]}

    def close(self):
        self.closed = True


def _pool(cases, devices=("2", "5"), settings=None, **kwargs):
    return ArtemisParallelOraclePool(_Source(cases), cases=cases, devices=devices,
        backend_settings=settings or {}, _resident_factory=_FakeResident,
        _exporter=_fake_export, startup_timeout_seconds=30, **kwargs)


def test_spawned_memory_requests_affinity_usage_and_cleanup():
    cases = tuple(_case(index) for index in range(4))
    pool = _pool(cases, settings={"device": "ignored", "reward_guidance_settings": {}})
    assert not pool.stats()["started"]
    try:
        assert pool.start() is pool
        assert pool.start() is pool
        assert pool.operation_lock("0") is pool.operation_lock("2")
        assert pool.operation_lock("0") is not pool.operation_lock("1")
        points = np.arange(4096 * 3, dtype=np.float64).reshape(4096, 3) / 20000
        initial = points[:, 1:]
        first, second = pool.session(cases[0]), pool.session(cases[1])
        with pool.operation_lock("0"):
            np.testing.assert_array_equal(first.query_source(points), points.sum(axis=1))
        np.testing.assert_array_equal(second.query_initial_distribution(initial), initial.sum(axis=1))
        np.testing.assert_array_equal(first.query_source(points[::-1]), points[::-1].sum(axis=1))
        assert first.usage_snapshot == {"query_calls": 2, "physical_device": "2"}
        assert second.usage_snapshot == {"query_calls": 1, "physical_device": "5"}
        fresh = pool.session(cases[0])
        assert fresh.usage_snapshot["query_calls"] == 0
        fresh.close()
        first.close()
        first.close()
        with pytest.raises(RuntimeError, match="closed"):
            first.query_source(points)
        stats = pool.stats()
        assert stats["case_assignments"] == {"0": "2", "1": "5", "2": "2", "3": "5"}
        assert [lane["request_count"] for lane in stats["lanes"]] == [2, 1]
        assert [lane["source_calls"] for lane in stats["lanes"]] == [2, 0]
        assert [lane["distribution_calls"] for lane in stats["lanes"]] == [0, 1]
        assert all(lane["pid"] != os.getpid() for lane in stats["lanes"])
    finally:
        pool.close()
    pool.close()
    assert pool.stats()["cleanup_complete"]
    assert all(not lane["alive"] for lane in pool.stats()["lanes"])
    with pytest.raises(RuntimeError, match="closed"):
        second.query_initial_distribution(initial)
    with pytest.raises(RuntimeError, match="closed"):
        pool.start()


def test_distinct_lanes_enter_scientific_callback_concurrently():
    barrier = multiprocessing.get_context("spawn").Barrier(2)
    cases = (_case(0), _case(1))
    pool = _pool(cases, settings={"barrier": barrier}).start()
    try:
        sessions = [pool.session(case) for case in cases]
        with ThreadPoolExecutor(max_workers=2) as threads:
            outputs = list(threads.map(lambda session: session.query_source(np.ones((8, 3))), sessions))
        for output in outputs:
            np.testing.assert_array_equal(output, np.full(8, 3.))
    finally:
        pool.close()


def test_one_lane_keeps_complete_rpc_pairs_together():
    cases = (_case(0), _case(1))
    pool = _pool(cases, devices=("7",), settings={"delay": .005}).start()
    try:
        sessions = [pool.session(case) for case in cases]
        def query(index):
            return sessions[index % 2].query_source(np.full((8, 3), index / 30))
        with ThreadPoolExecutor(max_workers=4) as threads:
            results = list(threads.map(query, range(20)))
        for index, output in enumerate(results):
            np.testing.assert_allclose(output, index / 10)
        assert [session.usage_snapshot["query_calls"] for session in sessions] == [10, 10]
    finally:
        pool.close()


def test_errors_causal_projection_and_case_scope_do_not_poison_worker():
    case = _case(0)
    pool = _pool((case,), devices=("3",)).start()
    try:
        with pytest.raises(PermissionError, match="outside"):
            pool.session(_case(99))
        projected = replace(case, public_data={"equation": {"domain": {"time_interval": [0., .5]}}})
        session = pool.session(projected)
        for points in (np.ones((1, 3)), np.zeros((4097, 3)), np.ones((2, 4)), np.full((1, 3), np.nan)):
            with pytest.raises(ValueError):
                session.query_source(points)
        np.testing.assert_array_equal(session.query_source(np.zeros((1, 3))), [0.])
        assert session.usage_snapshot["query_calls"] == 1
        assert not pool.stats()["lanes"][0]["failed"]
    finally:
        pool.close()


@pytest.mark.parametrize("settings, message", [({"fail_startup": True}, "startup"),
                                               ({"fail_register": True}, "register")])
def test_startup_failure_cleans_all_own_workers(settings, message):
    pool = _pool((_case(0), _case(1)), settings=settings)
    with pytest.raises(ValueError, match=message):
        pool.start()
    assert pool.stats()["closed"]
    assert pool.stats()["cleanup_complete"]


def test_worker_exit_fails_closed_without_replay():
    pool = _pool((_case(0),), devices=("4",), settings={"exit_during_query": True}).start()
    try:
        session = pool.session(_case(0))
        with pytest.raises(ArtemisOracleWorkerError, match="disconnected"):
            session.query_source(np.zeros((1, 3)))
        with pytest.raises(ArtemisOracleWorkerError, match="unavailable"):
            session.query_source(np.zeros((1, 3)))
        assert pool.stats()["lanes"][0]["request_count"] == 1
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"]


def test_scope_validation_before_any_worker():
    with pytest.raises(PermissionError, match="hidden"):
        _pool((_case(0, visibility=CaseVisibility.DEVELOPMENT),))
    with pytest.raises(ValueError, match="distinct issued"):
        _pool((_case(0), _case(0)))
    with pytest.raises(ValueError, match="distinct physical"):
        _pool((_case(0),), devices=("1", "1"))


@dataclass
class _Phase:
    numerator: int = 2
    denominator: int = 7
    translation: float = .125
    lineage_sha256: str = "phase"


@dataclass
class _Audit:
    state_sha256: str = "original-state"


class _TinyTensor:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float64)

    def detach(self):
        return self

    def to(self, *, device):
        assert device == "cpu"
        return self

    def numpy(self):
        return self.values


def test_export_only_original_compact_cpu_field_no_truth_or_sample():
    case = _case(0)
    field = SimpleNamespace(_condition_scores=_TinyTensor([.1, .2]), _latent=_TinyTensor([.7]))
    handle = SimpleNamespace(field=field, curvatures=_TinyTensor([.3]), spatial_phase=_Phase())
    manufactured = SimpleNamespace(audit=_Audit())
    issued = SimpleNamespace(case=case, manufactured=manufactured)
    def resolve(value):
        assert value is manufactured
        return handle
    source = SimpleNamespace(_active_issued=lambda *args, **kwargs: (issued, False),
                             runtime=SimpleNamespace(_exact_query_handle=resolve), artifact_sha256="same-runtime")
    descriptor = _export_case(source, case)
    assert "arrays" not in descriptor["case"]["public_data"]
    assert set(descriptor) == {"case", "condition_scores", "latent", "curvatures", "spatial_phase", "audit", "artifact_sha256"}
    assert descriptor["spatial_phase"] == vars(_Phase())
    np.testing.assert_array_equal(descriptor["latent"], [.7])
    assert not np.shares_memory(descriptor["latent"], field._latent.values)


def test_cleanup_attempts_remaining_workers_when_one_connection_close_fails():
    pool = _pool((_case(0), _case(1)))
    calls = []
    class FakeConnection:
        def __init__(self, index):
            self.index = index
        def send(self, value):
            calls.append((self.index, "send"))
        def close(self):
            calls.append((self.index, "close"))
            if self.index == 0:
                raise OSError("close error")
    class FakeProcess:
        pid = 1
        def __init__(self, index):
            self.index, self.alive = index, True
        def is_alive(self):
            return self.alive
        def join(self, timeout):
            calls.append((self.index, "join"))
        def terminate(self):
            calls.append((self.index, "terminate"))
            self.alive = False
        def kill(self):
            self.alive = False
    for index, lane in enumerate(pool._lanes):
        lane.connection, lane.process = FakeConnection(index), FakeProcess(index)
    with pytest.raises(ArtemisOracleWorkerError, match="cleanup"):
        pool.close()
    assert (1, "terminate") in calls
    assert all(not lane["alive"] for lane in pool.stats()["lanes"])
    assert not pool.stats()["cleanup_complete"]
    assert pool.stats()["cleanup_failed"]


def test_cleanup_never_claims_success_if_owned_worker_cannot_be_stopped():
    pool = _pool((_case(0),), devices=("1",))
    lane = pool._lanes[0]
    lane.process = SimpleNamespace(pid=1, is_alive=lambda: True,
        join=lambda **kwargs: None, terminate=lambda: None, kill=lambda: None)
    with pytest.raises(ArtemisOracleWorkerError, match="cleanup"):
        pool.close()
    assert pool.stats()["lanes"][0]["alive"]
    assert not pool.stats()["cleanup_complete"]
