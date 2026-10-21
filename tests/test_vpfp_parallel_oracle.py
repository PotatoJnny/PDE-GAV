"""Tiny CPU spawn/private-pipe checks; no actual assets or GPU speed claims."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.parallel_oracle import (
    VPFPOracleWorkerError, VPFPParallelOraclePool, _ResidentVPFP,
    _case_descriptor, _export_case,
)
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _equivalent_vpfp_public_snapshot
from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
from tests.test_vpfp_batched_source import decoder, _queries
from tests.test_vpfp_derivative_features import tiny
from tests.test_vpfp_canonical_demand_oracle import physical


def _case(index, visibility=CaseVisibility.HIDDEN):
    return Case(str(index), CaseSource.GENERATED, visibility,
                {"equation": "VPFP", "arrays": {"preview": np.zeros((2,))}})


class _Runtime:
    oracle_kernel = "structured_reuse_v1"
    def __init__(self, cases):
        self.cases = {case.case_id: case for case in cases}
    def _active_resident(self, case):
        original = self.cases.get(case.case_id)
        if original is None or not _equivalent_vpfp_public_snapshot(original, case):
            raise ValueError("inactive or changed case")
        return SimpleNamespace(case=original)
    def _resident_for_id(self,case_id):
        if case_id not in self.cases:
            raise ValueError("resident answer is no longer active")
        return SimpleNamespace(case=self.cases[case_id])


def _fake_export(runtime, case):
    runtime._active_resident(case)
    return {"case": _case_descriptor(case), "latent": np.ones(2),
            "equation_condition": np.array([3., .5])}


class _FakeSession:
    def __init__(self, settings):
        self.settings = settings
        self.calls = self.failed = 0
    def _query(self, points, width):
        if self.settings.get("exit_query"):
            os._exit(17)
        if points.ndim != 2 or points.shape[1] != width:
            self.failed += 1
            raise ValueError("wrong width")
        self.calls += 1
        return points.sum(axis=1)
    def source(self, points):
        return self._query(points, 5)
    def rho0(self, points):
        return self._query(points, 2)
    def g0(self, points):
        return self._query(points, 4)
    @property
    def usage_snapshot(self):
        return {"calls": self.calls, "failed": self.failed,
                "physical_device": os.environ["CUDA_VISIBLE_DEVICES"], "process_id": os.getpid()}
    def close(self):
        pass


class _FakeResident:
    def __init__(self, settings, device):
        assert device == "cuda:0"
        if settings.get("fail_start"):
            raise ValueError("startup failed")
        self.settings, self.cases = settings, {}
    def register(self, descriptor):
        if self.settings.get("fail_register"):
            raise ValueError("register failed")
        self.cases[descriptor["case"]["case_id"]] = descriptor
    def open(self, case_id):
        assert case_id in self.cases
        return _FakeSession(self.settings)
    def stats(self):
        return {"case_count": len(self.cases)}
    def close(self):
        self.cases.clear()


def _pool(cases, *, devices=("2", "5"), settings=None, **kwargs):
    return VPFPParallelOraclePool(_Runtime(cases), cases=cases, devices=devices,
        _resident_factory=_FakeResident, _exporter=_fake_export, _settings=settings or {},
        startup_timeout_seconds=30., request_timeout_seconds=10., **kwargs)


def test_spawn_private_pipe_affinity_exact_queries_and_cleanup():
    cases = tuple(_case(i) for i in range(4))
    pool = _pool(cases)
    assert not pool.stats()["started"]
    try:
        assert pool.start() is pool and pool.start() is pool
        assert pool.operation_lock("0") is pool.operation_lock("2")
        assert pool.operation_lock("0") is not pool.operation_lock("1")
        query = np.array([[.2,.4,.3,.1,-.2],[.7,.1,.8,-.6,.5],[.2,.4,.3,.1,-.2]])
        for case, device in zip(cases, ("2", "5", "2", "5")):
            session = pool.session(case)
            np.testing.assert_array_equal(session.source(query), query.sum(1))
            np.testing.assert_array_equal(session.rho0(query[:,1:3]), query[:,1:3].sum(1))
            np.testing.assert_array_equal(session.g0(query[:,1:]), query[:,1:].sum(1))
            assert session.usage_snapshot["calls"] == 3
            assert session.usage_snapshot["physical_device"] == device
            session.close(); session.close()
            with pytest.raises(RuntimeError, match="closed"):
                session.source(query)
        info = pool.stats()
        assert [lane["registered_cases"] for lane in info["lanes"]] == [2,2]
        assert sum(lane["request_count"] for lane in info["lanes"]) == 12
        assert info["transferred_latent_bytes"] == 64
        assert info["gpu_device_count"] == info["process_count"] == 2
        assert info["workers_per_gpu"] == 1
        assert info["maximum_prepared_cases_per_gpu"] == 16
    finally:
        pool.close(); pool.close()
    assert pool.stats()["cleanup_complete"] is True
    assert all(not lane["alive"] for lane in pool.stats()["lanes"])
    with pytest.raises(RuntimeError, match="closed"):
        pool.start()


@pytest.mark.parametrize("devices, workers", [(("2",), 2), (("2",), 4), (("2", "5", "8"), 2)])
def test_multiple_processes_per_gpu_have_fixed_single_case_owners(devices, workers):
    lane_count = len(devices) * workers
    cases = tuple(_case(i) for i in range(2 * lane_count))
    pool = _pool(cases, devices=devices, workers_per_gpu=workers)
    try:
        pool.start()
        info = pool.stats()
        assert info["gpu_device_count"] == len(devices)
        assert info["process_count"] == lane_count
        assert info["workers_per_gpu"] == workers
        assert info["worker_layout"] == "fixed_case_owner_v1"
        assert info["maximum_prepared_cases_per_process"] == 16
        assert info["maximum_prepared_cases_per_gpu"] == 16 * workers
        assert len({lane["pid"] for lane in info["lanes"]}) == lane_count
        assert [lane["device"] for lane in info["lanes"]] == list(devices) * workers
        assert [lane["worker_index"] for lane in info["lanes"]] == [i // len(devices) for i in range(lane_count)]
        for index, case in enumerate(cases):
            owner = index % lane_count
            assert info["case_lane_assignments"][case.case_id] == owner
            assert info["case_assignments"][case.case_id] == devices[index % len(devices)]
            assert pool.operation_lock(case.case_id) is pool.operation_lock(str(owner))
            session = pool.session(case)
            np.testing.assert_array_equal(session.source(np.ones((2, 5))), [5., 5.])
            assert session.usage_snapshot["process_id"] == info["lanes"][owner]["pid"]
            assert session.usage_snapshot["calls"] == 1
            session.close()
        assert [lane["registered_cases"] for lane in pool.stats()["lanes"]] == [2] * lane_count
        assert pool.stats()["transferred_latent_bytes"] == len(cases) * 16
        for lane in pool._lanes:
            assert lane.rpc({"kind": "stats"}, 10.)["result"]["case_count"] == 2
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"] is True
    assert all(not lane["alive"] for lane in pool.stats()["lanes"])


def test_same_gpu_lanes_do_not_share_query_queue_locks():
    cases = (_case(0), _case(1))
    pool = _pool(cases, devices=("2",), workers_per_gpu=2)
    try:
        pool.start()
        first, second = (pool.session(case) for case in cases)
        assert pool.operation_lock("0") is not pool.operation_lock("1")
        with ThreadPoolExecutor(max_workers=2) as executor:
            with pool.operation_lock("0"):
                blocked = executor.submit(first.source, np.ones((1, 5)))
                ready = executor.submit(second.source, np.ones((1, 5)))
                np.testing.assert_array_equal(ready.result(timeout=5.), [5.])
                assert not blocked.done()
            np.testing.assert_array_equal(blocked.result(timeout=5.), [5.])
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"] is True


def test_failed_process_is_not_replayed_to_other_process_on_same_gpu():
    cases = (_case(0), _case(1))
    pool = _pool(cases, devices=("2",), workers_per_gpu=2, settings={"exit_query": True})
    try:
        pool.start()
        session = pool.session(cases[0])
        with pytest.raises(VPFPOracleWorkerError):
            session.source(np.zeros((1, 5)))
        with pytest.raises(VPFPOracleWorkerError, match="unavailable"):
            session.source(np.zeros((1, 5)))
        info = pool.stats()
        assert info["case_lane_assignments"] == {"0": 0, "1": 1}
        assert [lane["request_count"] for lane in info["lanes"]] == [1, 0]
        assert info["lanes"][1]["alive"] is True
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"] is True


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2", None])
def test_invalid_workers_per_gpu_rejected_before_spawn(value):
    with pytest.raises((TypeError, ValueError), match="workers_per_gpu"):
        _pool((_case(0),), workers_per_gpu=value)


def test_query_errors_snapshot_scope_and_separate_sessions():
    cases = (_case(0), _case(1))
    pool = _pool(cases)
    try:
        pool.start()
        first, second = pool.session(cases[0]), pool.session(cases[0])
        with pytest.raises(ValueError, match="wrong width"):
            first.source(np.zeros((2,4)))
        assert first.usage_snapshot["failed"] == 1
        assert second.usage_snapshot["failed"] == 0
        np.testing.assert_array_equal(first.source(np.zeros((2,5))), [0.,0.])
        with pytest.raises(PermissionError, match="outside"):
            pool.session(_case(9))
        with pytest.raises(ValueError, match="changed"):
            pool.session(replace(cases[0], public_data={"equation":"tampered"}))
        assert not hasattr(first, "released_macros")
        assert not hasattr(first, "latent")
        # Even a malformed private pipe operation cannot ask a replica for
        # truth/development macros or change to an undeclared public method.
        reply = pool._lanes[0].rpc({"kind":"query", "session_id":first._session_id,
                                   "method":"released_macros", "points":np.zeros((1,3))},10.)
        assert not reply["ok"] and reply["error_type"] == "PermissionError"
        pool.runtime.cases.pop(cases[0].case_id)
        with pytest.raises(ValueError,match="no longer active"):
            first.source(np.zeros((1,5)))
    finally:
        pool.close()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("setting, message", [("fail_start","startup failed"),("fail_register","register failed")])
def test_startup_or_registration_failure_reclaims_all_owned_workers(setting, message, workers):
    pool = _pool((_case(0), _case(1)), settings={setting:True}, workers_per_gpu=workers)
    with pytest.raises(ValueError, match=message):
        pool.start()
    assert pool.stats()["cleanup_complete"] is True


def test_worker_exit_fails_closed_without_changing_lane():
    pool = _pool((_case(0),), devices=("2",), settings={"exit_query":True})
    try:
        pool.start()
        session = pool.session(_case(0))
        with pytest.raises(VPFPOracleWorkerError):
            session.source(np.zeros((1,5)))
        with pytest.raises(VPFPOracleWorkerError, match="unavailable"):
            session.source(np.zeros((1,5)))
        assert pool.stats()["lanes"][0]["request_count"] == 1
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"] is True


@pytest.mark.parametrize("devices", [(),("2","2"),("",),("1,2",)])
def test_bad_devices_do_not_start_processes(devices):
    with pytest.raises(ValueError, match="GPU tokens"):
        _pool((_case(0),), devices=devices)


def test_nonhidden_or_duplicate_case_scope_rejected():
    with pytest.raises(PermissionError, match="hidden"):
        _pool((_case(0, CaseVisibility.DEVELOPMENT),))
    with pytest.raises(ValueError, match="distinct issued"):
        _pool((_case(0),_case(0)))


class _CPUResident(_ResidentVPFP):
    def __init__(self, settings, device):
        assert device == "cuda:0"
        super().__init__(settings, "cpu")


@pytest.mark.parametrize("workers", [1, 2])
def test_real_tiny_decoder_private_replica_equals_parent_queries(physical, workers):
    runtime = physical.runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    record = runtime.record("real-tiny", visibility=CaseVisibility.HIDDEN, seed=1)
    query = _queries("sparse_shuffled_duplicates")
    expected = {"source":runtime.evaluate_source(record.case.case_id,query),
                "rho0":runtime.evaluate_rho0(record.case.case_id,query[:,1:3]),
                "g0":runtime.evaluate_g0(record.case.case_id,query[:,1:])}
    descriptor = _export_case(runtime,record.case)
    assert set(descriptor) == {"case","latent","equation_condition"}
    np.testing.assert_array_equal(descriptor["latent"],runtime._resident_for_id(record.case.case_id).latent)
    assert descriptor["latent"] is not runtime._resident_for_id(record.case.case_id).latent
    pool = VPFPParallelOraclePool(runtime,cases=(record.case,),devices=("2",),
                                 workers_per_gpu=workers,
                                 _resident_factory=_CPUResident,startup_timeout_seconds=30.)
    try:
        pool.start()
        session = pool.session(record.case)
        for _ in range(2):
            for method,points in (("source",query),("rho0",query[:,1:3]),("g0",query[:,1:])):
                np.testing.assert_array_equal(getattr(session,method)(points),expected[method])
        assert session.usage_snapshot["query_calls"] == 6
        lane = pool._lanes[0]
        cache = lane.rpc({"kind":"stats"},10.)["result"]
        assert cache["case_count"] == 1 and cache["max_cases"] == 16
        assert cache["cases"][record.case.case_id]["cached_final_source_arrays"] is False
        runtime.release_cases((record.case.case_id,))
        with pytest.raises(ValueError,match="no longer active"):
            session.source(query)
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"] is True


def test_export_uses_existing_latent_and_never_touches_truth(physical):
    record = physical.runtime.record("export",visibility=CaseVisibility.HIDDEN,seed=1)
    descriptor = _export_case(physical.runtime,record.case)
    assert not any(name in descriptor for name in ("truth","generator","guidance","source"))
    descriptor["latent"].fill(0.)
    assert np.any(np.asarray(physical.runtime._resident_for_id("export").latent) != 0.)


class _OwnedProcess:
    pid=123  # This fake never invokes an operating-system process operation.
    def __init__(self, unstoppable=False):
        self.alive,self.unstoppable,self.stops=True,unstoppable,[]
    def is_alive(self):return self.alive
    def join(self,timeout):pass
    def terminate(self):
        self.stops.append("terminate")
        if not self.unstoppable:self.alive=False
    def kill(self):
        self.stops.append("kill")
        if not self.unstoppable:self.alive=False


class _OwnedConnection:
    def __init__(self,fail=False):self.closed=False;self.fail=fail
    def send(self,request):pass
    def close(self):
        self.closed=True
        if self.fail:raise OSError("connection close failed")


@pytest.mark.parametrize("workers", [1, 2])
def test_cleanup_failure_still_attempts_every_owned_lane(workers):
    pool=_pool((_case(0),_case(1)),workers_per_gpu=workers)
    for index,lane in enumerate(pool._lanes):
        lane.process=_OwnedProcess()
        lane.connection=_OwnedConnection(fail=index==0)
    with pytest.raises(VPFPOracleWorkerError,match="cleanup failed"):
        pool.close()
    assert all(lane.connection.closed for lane in pool._lanes)
    assert all(not lane.process.is_alive() for lane in pool._lanes)
    assert pool.stats()["cleanup_complete"] is False


def test_unstoppable_owned_worker_cannot_report_successful_handoff():
    pool=_pool((_case(0),),devices=("2",))
    pool._lanes[0].process=_OwnedProcess(unstoppable=True)
    pool._lanes[0].connection=_OwnedConnection()
    with pytest.raises(VPFPOracleWorkerError,match="cleanup failed"):
        pool.close()
    assert pool._lanes[0].process.stops==["terminate","kill"]
    assert pool.stats()["cleanup_complete"] is False
