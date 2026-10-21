"""Synthetic mechanics only; production throughput is a separate GPU check."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Barrier, Event, RLock, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.evaluator import _CaseSessionBindings
from pde_agent_verifier_solver.sdk import CaseVisibility, EvaluationPhase
from pde_agent_verifier_solver.plugins.vpfp import weak_parallel_oracle as parallel
from pde_agent_verifier_solver.plugins.vpfp.moving_frame_backend import _Resident
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _snapshot_vpfp_public_case
from tests.test_vpfp_weak_parallel_oracle import runtime, wrapper, FakePool
from tests.test_vpfp_moving_frame import points


def development_cases(runtime):
    for identifier, row in tuple(runtime._resident_cases.items()):
        case = replace(row.case, visibility=CaseVisibility.DEVELOPMENT)
        runtime._resident_cases[identifier] = replace(row, case=case,
            public_snapshot=_snapshot_vpfp_public_case(case))
    return tuple(row.case for row in runtime._resident_cases.values())


def add_case(runtime, original, identifier, visibility=CaseVisibility.DEVELOPMENT):
    resident = runtime._resident_for_id(original.case_id)
    case = replace(original, case_id=identifier, visibility=visibility)
    runtime._resident_cases[identifier] = _Resident(case, _snapshot_vpfp_public_case(case),
        {key: value.clone() for key, value in resident.carrier.items()},
        resident.condition.clone(), resident.answer_sha256)
    return case


class CPUDevelopmentResident(parallel._DevelopmentResidentWeakVPFP):
    def __init__(self, settings, device):
        assert device == "cuda:0"
        super().__init__(settings, "cpu")


def test_development_dynamic_replica_exactness_revocation_and_cleanup(runtime):
    original = development_cases(runtime)
    hidden = add_case(runtime, original[0], "hidden-forbidden", CaseVisibility.HIDDEN)
    pool = parallel.WeakVPFPDevelopmentPool(runtime, cases=original[:1], devices=("test-controller",),
        workers_per_gpu=2, _resident_factory=CPUDevelopmentResident, startup_timeout_seconds=60.)
    query = points().numpy()[[2, 0, 2, 1]]
    try:
        pool.start()
        pool.register_cases(original)
        assert len({lane["pid"] for lane in pool.stats()["lanes"]}) == 2
        assert pool.operation_lock(original[0].case_id) is not pool.operation_lock(original[1].case_id)
        with pytest.raises(PermissionError):
            pool.register_cases((hidden,))
        later = add_case(runtime, original[0], "later-guided")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(pool.register_cases, (later,)) for _ in range(2)]
            for future in futures:
                future.result(timeout=15.)
        assert sum(lane["registered_cases"] for lane in pool.stats()["lanes"]) == 3
        session = pool.session(later)
        ordinary = runtime.public_session_factory(later)
        try:
            for name, argument in (("source", query), ("g0", query[:, 1:]), ("rho0", query[:, 1:3])):
                np.testing.assert_array_equal(getattr(session, name)(argument), getattr(ordinary, name)(argument))
            with pytest.raises(Exception, match="open session"):
                pool.release_cases((later.case_id,))
        finally:
            session.close()
            ordinary.close()
        pool.release_cases((later.case_id,))
        assert later.case_id not in pool.stats()["case_lane_assignments"]
        with pytest.raises(PermissionError):
            pool.session(later)
        runtime.release_cases((original[0].case_id,))
        with pytest.raises(ValueError, match="no longer active"):
            pool.session(original[0])
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"]


class FakeDevelopmentPool:
    instances = []

    def __init__(self, runtime, *, cases, devices, workers_per_gpu):
        self.runtime, self.devices = runtime, tuple(devices)
        self.workers_per_gpu = workers_per_gpu
        self.locks = [RLock() for _ in range(workers_per_gpu)]
        self._assignments = {}
        self.closed, self.started = False, False
        self.rendezvous = Barrier(2)
        self.register_cases(cases)
        self.instances.append(self)

    def start(self):
        self.started = True
        return self

    def register_cases(self, cases):
        for case in cases:
            assert case.visibility is CaseVisibility.DEVELOPMENT
            self.runtime._active(case)
            if case.case_id not in self._assignments:
                self._assignments[case.case_id] = len(self._assignments) % self.workers_per_gpu

    def operation_lock(self, case_id):
        return self.locks[self._assignments[case_id]]

    def session(self, case):
        self.runtime._active(case)
        def source(query):
            self.rendezvous.wait(timeout=5.)
            return np.asarray(query).sum(1)
        return SimpleNamespace(source=source, close=lambda: None)

    def release_cases(self, identifiers):
        for identifier in identifiers:
            self._assignments.pop(identifier, None)

    def close(self):
        self.closed = True

    def stats(self):
        return dict(cleanup_complete=self.closed, gpu_devices=self.devices,
            process_count=self.workers_per_gpu, started=self.started)


def test_development_lane_dispatch_is_parallel_and_provisioning_precedes_candidate_budget(runtime):
    original = development_cases(runtime)
    binding, backend = wrapper(runtime)
    called = []
    service = SimpleNamespace(oracle_host_count=1, oracle_max_workers=1,
        case_hard_wall_seconds=900, evaluation_settings={"candidate_budget_mode": "ns_remaining_case_v1"})

    def execute(artifact, cases, plans, evaluation, **kwargs):
        assert manager._active_batches == 1
        assert manager._pool.started
        assert cases[0].case_id == "opaque-candidate-0"
        called.append("candidate_process_start")
        return ("finished",)

    service.execute_cases = execute
    manager = parallel.install_development_oracle(binding, service, controller_gpu="controller-only",
        workers_per_gpu=2, pool_factory=FakeDevelopmentPool)
    sessions = _CaseSessionBindings(binding.plugin.generator, SimpleNamespace())
    active = []
    try:
        active = [sessions.persistent_public_session(case, replace(case, case_id=f"opaque-candidate-{i}"))
            for i, case in enumerate(original)]
        assert manager._pool.started and not called
        assert manager._pool.devices == ("controller-only",)
        assert active[0]._operation_lock is not active[1]._operation_lock
        assert active[0]._operation_lock is not sessions._operation_lock
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(item.invoke, "source", (np.ones((1, 5)),), {}) for item in active]
            for future in futures:
                np.testing.assert_array_equal(future.result(timeout=6.), [5.])
        assert service.execute_cases(None, [replace(original[0], case_id="opaque-candidate-0")],
            [SimpleNamespace(phase=EvaluationPhase.DEVELOPMENT)], None) == ("finished",)
        assert service.case_hard_wall_seconds == 900
        assert service.evaluation_settings == {"candidate_budget_mode": "ns_remaining_case_v1"}
        # Shared admission capacity is fixed by the service constructor, not
        # mutated when a PDE-specific development pool is attached.
        assert service.oracle_host_count == 1 and service.oracle_max_workers == 1
    finally:
        for item in active:
            item.close()
        manager.shutdown()
    assert manager.reports[-1]["cleanup_complete"]


def test_development_formal_boundary_cleanup_dynamic_release_and_restart(runtime):
    original = development_cases(runtime)
    hidden = add_case(runtime, original[0], "official", CaseVisibility.HIDDEN)
    binding, backend = wrapper(runtime)
    execution = SimpleNamespace(oracle_host_count=1, oracle_max_workers=1, execute_cases=lambda *a, **k: ())
    manager = parallel.install_development_oracle(binding, execution, controller_gpu="0",
        workers_per_gpu=2, pool_factory=FakeDevelopmentPool)
    scope = parallel.WeakVPFPHiddenEvaluationScope(binding, execution,
        devices=("0", "1", "2"), workers_per_gpu=2, pool_factory=FakePool)
    manager.ensure_cases(original)
    first = manager._pool
    with manager.batch():
        with pytest.raises(RuntimeError, match="active development"):
            with scope(cases=(hidden,)):
                pass
    with scope(cases=(hidden,)):
        assert first.closed and manager._pool is None
        assert backend._parallel_oracle_pool is not None
    assert backend._parallel_oracle_pool is None
    manager.ensure_cases(original)
    assert manager._pool is not first
    with pytest.raises(PermissionError):
        manager.ensure_cases((hidden,))
    backend.release_cases((original[0].case_id,))
    assert original[0].case_id not in manager._pool._assignments
    assert original[0].case_id not in runtime._resident_cases
    manager.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        manager.ensure_cases(original[1:])


def test_development_failed_start_and_shutdown_close_all_owned_workers(runtime):
    original = development_cases(runtime)
    class FailedStart(FakeDevelopmentPool):
        def start(self):
            raise LookupError("test startup failure")
    manager = parallel.WeakVPFPDevelopmentOracle(runtime, controller_gpu="0",
        pool_factory=FailedStart)
    with pytest.raises(LookupError, match="startup failure"):
        manager.ensure_cases(original)
    assert FailedStart.instances[-1].closed
    assert manager._pool is None
    manager.shutdown()


def test_development_registration_cannot_deadlock_lazy_session_open(runtime):
    original = development_cases(runtime)
    later = add_case(runtime, original[0], "registered-during-active-query")
    query_has_lane, registration_waiting = Event(), Event()
    query_done, registration_done = Event(), Event()
    errors = []

    class BlockingRegistration(FakeDevelopmentPool):
        block_new = False

        def register_cases(self, cases):
            if not self.block_new:
                return super().register_cases(cases)
            registration_waiting.set()
            # Match real lane.rpc: manager registration lock -> lane lock.
            with self.locks[0]:
                return super().register_cases(cases)

    manager = parallel.WeakVPFPDevelopmentOracle(runtime, controller_gpu="0",
        pool_factory=BlockingRegistration)
    manager.ensure_cases(original)
    manager._pool.block_new = True

    def query():
        try:
            with manager._pool.operation_lock(original[0].case_id):
                query_has_lane.set()
                assert registration_waiting.wait(3.)
                # This is the real Evaluator ordering: lane lock -> factory.
                manager.session(original[0]).close()
            query_done.set()
        except BaseException as error:
            errors.append(error)

    def register():
        try:
            assert query_has_lane.wait(3.)
            # A direct public probe of a newly issued Case also provisions;
            # the registered query must remain a lock-free manager fastpath.
            manager.session(later).close()
            registration_done.set()
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=query, daemon=True), Thread(target=register, daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=4.)
    assert not any(thread.is_alive() for thread in threads), "lane/registration lock inversion"
    assert not errors and query_done.is_set() and registration_done.is_set()
    manager.shutdown()


def test_direct_agent_oracle_provisions_first_and_later_cases_via_evaluator(runtime):
    original = development_cases(runtime)
    binding, backend = wrapper(runtime)
    execution = SimpleNamespace(oracle_max_workers=1, execute_cases=lambda *a, **k: ())

    class ImmediatePool(FakeDevelopmentPool):
        def session(self, case):
            self.runtime._active(case)
            return SimpleNamespace(source=lambda values: np.asarray(values).sum(1),
                g0=lambda values: np.asarray(values).sum(1), close=lambda: None)

    manager = parallel.install_development_oracle(binding, execution, controller_gpu="0",
        pool_factory=ImmediatePool)
    bindings = _CaseSessionBindings(binding.plugin.generator, SimpleNamespace())
    assert manager._pool is None
    try:
        result = bindings.invoke_public(original[0], replace(original[0], case_id="opaque-first"),
            EvaluationPhase.DEVELOPMENT, "g0", (np.ones((2, 4)),), {})
        np.testing.assert_array_equal(result, [4., 4.])
        assert original[0].case_id in manager._pool._assignments
        late = add_case(runtime, original[1], "later-direct-probe")
        result = bindings.invoke_public(late, replace(late, case_id="opaque-later"),
            EvaluationPhase.DEVELOPMENT, "source", (np.ones((1, 5)),), {})
        np.testing.assert_array_equal(result, [5.])
        assert late.case_id in manager._pool._assignments
    finally:
        manager.shutdown()
