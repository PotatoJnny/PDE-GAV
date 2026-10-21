"""Tiny synthetic mechanics; real assets and GPU throughput are separate checks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import RLock, Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
from pde_agent_verifier_solver.engine.evaluator import _CaseSessionBindings
from pde_agent_verifier_solver.plugins.vpfp import weak_parallel_oracle as parallel
from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameDecoder
from pde_agent_verifier_solver.plugins.vpfp.moving_frame_backend import MovingFrameBackend, _Resident
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _snapshot_vpfp_public_case
from pde_agent_verifier_solver.plugins.vpfp.weak_iteration_backend import WeakIterationBackend, WeakIterationGenerator
from pde_agent_verifier_solver.plugins.vpfp.weak_portable_base import _state_sha
from tests.test_vpfp_moving_frame import tiny, constant_macro, points


@pytest.fixture
def runtime(tmp_path):
    original_threads, original_dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    base, value, macro, z, condition = tiny(dtype="float32")
    decoder = MovingFrameDecoder(base, macro, value_config=value).eval().requires_grad_(False)
    result = MovingFrameBackend(decoder, SimpleNamespace(identity={"synthetic": True}),
        output_root=tmp_path / "native", adaptive_source_only=True, source_chunk=32, field_chunk=32)
    for index in range(2):
        case = Case(f"native-{index}", CaseSource.GENERATED, CaseVisibility.HIDDEN,
            {"arrays": {"preview": np.zeros(2)}}, round_index=2, tags=("base",))
        carrier = {"m": constant_macro(macro).float(), "z": z.clone()}
        result._resident_cases[case.case_id] = _Resident(case, _snapshot_vpfp_public_case(case),
            carrier, condition.clone(), "a" * 64)
    try:
        yield result
    finally:
        result.release_cases(tuple(result._resident_cases))
        torch.set_num_threads(original_threads)
        torch.set_default_dtype(original_dtype)


def cases(runtime):
    return tuple(row.case for row in runtime._resident_cases.values())


class CPUWeakResident(parallel._ResidentWeakVPFP):
    def __init__(self, settings, device):
        assert device == "cuda:0"
        super().__init__(settings, "cpu")


def test_decoder_only_rebuild_preserves_original_bytes_and_all_public_queries(runtime):
    case = cases(runtime)[0]
    before = _state_sha(runtime.decoder.state_dict())
    descriptor = parallel._export_weak_case(runtime, case)
    settings = parallel._descriptor_settings(runtime)
    assert "sampler" not in settings and "guidance" not in settings and "truth" not in descriptor
    replica = parallel._ResidentWeakVPFP(settings, "cpu")
    try:
        replica.register(descriptor)
        with pytest.raises(ValueError, match="twice"):
            replica.register(descriptor)
        assert _state_sha(replica.runtime.decoder.state_dict()) == before
        for key in ("m", "z"):
            actual = replica.runtime._resident_for_id(case.case_id).carrier[key]
            assert torch.equal(actual, runtime._resident_for_id(case.case_id).carrier[key])
            assert not np.shares_memory(actual.numpy(), descriptor["carrier"][key])
        query = points().numpy()[[2, 0, 2, 1]]
        session = replica.open(case.case_id)
        original = runtime.public_session_factory(case)
        try:
            for method, argument in (("source", query), ("rho0", query[:, 1:3]), ("g0", query[:, 1:])):
                np.testing.assert_array_equal(getattr(session, method)(argument), getattr(original, method)(argument))
            assert session.usage_snapshot["attempted_calls"] == 3
        finally:
            session.close(); original.close()
        assert replica.stats()["generated_networks_loaded"] is False
        assert _state_sha(runtime.decoder.state_dict()) == before
    finally:
        replica.close()
    assert replica.stats()["resident_case_count"] == 0


@pytest.mark.parametrize("mutation", ["state", "identity", "kernel", "shape", "visibility"])
def test_changed_replica_inputs_fail_closed(runtime, mutation):
    settings = parallel._descriptor_settings(runtime)
    descriptor = parallel._export_weak_case(runtime, cases(runtime)[0])
    if mutation == "state":
        first = next(value for value in settings["decoder_state"].values() if value.dtype.kind == "f" and value.size)
        first.flat[0] += 1.
    elif mutation == "identity": settings["decoder_identity"]["schema"] = "wrong"
    elif mutation == "kernel": settings["source_implementation"]["name"] = "structured_reuse_v1"
    elif mutation == "shape": descriptor["carrier"]["m"] = descriptor["carrier"]["m"][:, :-1]
    else: descriptor["case"]["visibility"] = CaseVisibility.DEVELOPMENT.value
    replica = None
    try:
        with pytest.raises((ValueError, PermissionError)):
            replica = parallel._ResidentWeakVPFP(settings, "cpu")
            replica.register(descriptor)
    finally:
        if replica is not None:
            replica.close()


def test_hidden_perturbed_carrier_is_not_reinterpreted_as_a_generated_draw(runtime):
    descriptor = parallel._export_weak_case(runtime, cases(runtime)[0])
    descriptor["case"].update(source=CaseSource.PERTURBED.value, parent_case_id="anchor")
    replica = parallel._ResidentWeakVPFP(parallel._descriptor_settings(runtime), "cpu")
    try:
        replica.register(descriptor)
        assert replica.runtime._resident_for_id(descriptor["case"]["case_id"]).case.source is CaseSource.PERTURBED
        assert not hasattr(replica.runtime, "sampler") and not hasattr(replica.runtime, "generator")
    finally:
        replica.close()


def test_real_weak_carrier_spawn_affinity_snapshot_revocation_and_cleanup(runtime):
    original = cases(runtime)
    pool = parallel.WeakVPFPParallelOraclePool(runtime, cases=original, devices=("test-GPU",),
        workers_per_gpu=2, _resident_factory=CPUWeakResident, startup_timeout_seconds=60.)
    query = points().numpy()[[1, 2, 1]]
    expected = runtime.evaluate_source(original[0].case_id, query)
    try:
        pool.start()
        assert pool.operation_lock(original[0].case_id) is not pool.operation_lock(original[1].case_id)
        session = pool.session(original[0])
        np.testing.assert_array_equal(session.source(query), expected)
        assert pool.stats()["oracle_kernel"] == "adaptive_exact_v1"
        assert pool.stats()["process_count"] == 2
        assert all(not row["device_metadata"]["generated_networks_loaded"] for row in pool.stats()["lanes"])
        changed = deepcopy(original[0]); changed.public_data["arrays"]["preview"][0] = 1.
        with pytest.raises(ValueError, match="snapshot"):
            pool.session(changed)
        runtime.release_cases((original[0].case_id,))
        with pytest.raises(ValueError, match="no longer active"):
            session.source(query)
        session.close()
    finally:
        pool.close()
    assert pool.stats()["cleanup_complete"]
    assert all(not lane["alive"] for lane in pool.stats()["lanes"])


class FakePool:
    def __init__(self, runtime, *, cases, devices, workers_per_gpu):
        self.runtime, self.cases, self.devices = runtime, tuple(cases), tuple(devices)
        self.workers_per_gpu, self.closed = workers_per_gpu, False
        self.locks = {case.case_id: RLock() for case in cases}
        self.rendezvous = Barrier(2)

    def start(self): return self
    def close(self): self.closed = True
    def operation_lock(self, case_id): return self.locks[case_id]
    def stats(self): return dict(cleanup_complete=self.closed)
    def session(self, case):
        self.runtime._active(case)
        def query(points):
            self.rendezvous.wait(timeout=5.)
            return np.asarray(points).sum(1)
        return SimpleNamespace(source=query, close=lambda: None)


def wrapper(runtime):
    capture = SimpleNamespace(base=SimpleNamespace(assert_frozen=lambda: None))
    backend = WeakIterationBackend(runtime, capture, output_root=runtime.output_root / "guide",
        trainer=None, config={})
    generator = WeakIterationGenerator(backend, {"binding_sha256": "b" * 64})
    return SimpleNamespace(plugin=SimpleNamespace(generator=generator)), backend


@pytest.mark.parametrize("fails", [False, True])
def test_full_evaluator_generator_backend_lane_lock_chain_and_scope_restoration(runtime, fails):
    binding, backend = wrapper(runtime)
    execution = SimpleNamespace(oracle_host_count=1)
    events = []
    scope = parallel.WeakVPFPHiddenEvaluationScope(binding, execution, devices=("0", "1", "2"),
        workers_per_gpu=2, pool_factory=FakePool,
        events=SimpleNamespace(emit=lambda kind, message, *, payload: events.append((kind, payload))))
    generator = binding.plugin.generator
    sessions = _CaseSessionBindings(generator, SimpleNamespace())
    original = cases(runtime)
    assert generator.public_session_operation_lock(None, original[0]) is None
    default = sessions.persistent_public_session(original[0], deepcopy(original[0]))
    assert default._operation_lock is sessions._operation_lock
    default.close()
    try:
        with scope(cases=original, unavailable_gpu_devices=("2",)) as pool:
            assert execution.oracle_host_count == 4
            assert backend._parallel_oracle_pool is pool and pool.devices == ("0", "1")
            active = [sessions.persistent_public_session(case, deepcopy(case)) for case in original]
            assert all(item._operation_lock is pool.operation_lock(case.case_id)
                for item, case in zip(active, original))
            assert active[0]._operation_lock is not active[1]._operation_lock
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = [executor.submit(item.invoke, "source", (np.ones((1, 5)),), {}) for item in active]
                for future in pending:
                    np.testing.assert_array_equal(future.result(timeout=6.), [5.])
            for item in active: item.close()
            if fails: raise LookupError("candidate failed")
    except LookupError:
        assert fails
    assert pool.closed and backend._parallel_oracle_pool is None and execution.oracle_host_count == 1
    assert scope.last_report["pool"]["cleanup_complete"]
    assert generator.public_session_operation_lock(None, original[0]) is None
    ready = next(row for kind, row in events if kind == "evaluation_gpus_ready")
    assert ready["oracle_workers"] == ready["oracle_process_count"] == 4
    assert ready["candidate_transport"] == "unchanged_existing_wire_codec"


def test_scope_rejects_wrong_numerical_kernel_before_worker_start(runtime):
    binding, _ = wrapper(runtime)
    runtime.adaptive_source_only = False
    with pytest.raises(ValueError, match="unchanged adaptive-only"):
        parallel.WeakVPFPHiddenEvaluationScope(binding, SimpleNamespace(oracle_host_count=1), devices=("0",))
