"""Reuse the real decoder/kernel while testing scoped per-device lifetimes."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib
import gc
from threading import RLock
import weakref

import numpy as np
import pytest

from tests.test_vpfp_batched_source import decoder, tiny, _queries
from tests.test_vpfp_canonical_demand_oracle import _TinyResidentRuntime
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import VPFPLegacyStructuredRuntime
from pde_agent_verifier_solver.plugins.vpfp import evaluation_devices as devices
from pde_agent_verifier_solver.sdk.evaluation_devices import (
    EvaluationDevicePool, bind_evaluation_device_pool, register_cuda_workspace_cleanup,
)


@pytest.fixture(autouse=True)
def vpfp_test_package_path(monkeypatch):
    # NS and VPFP vendor the same top-level legacy package. These mixed-plugin
    # tests explicitly expose both subpackages without changing runtime loaders.
    root = Path(__file__).resolve().parents[1] / "legacy/vpfp/src/agent_verifier"
    try:
        package = importlib.import_module("agent_verifier")
    except ModuleNotFoundError:
        monkeypatch.syspath_prepend(str(root.parent))
    else:
        monkeypatch.setattr(package, "__path__", [*package.__path__, str(root)])


@pytest.fixture
def physical(decoder, tmp_path):
    sampled = _TinyResidentRuntime(decoder, tmp_path / "runtime")
    record = sampled.record("evaluation-case")
    # Formal binding uses the exact base runtime; fixture sampling above only
    # supplies tiny deterministic resident data, never replaces its Oracle math.
    runtime = object.__new__(VPFPLegacyStructuredRuntime)
    runtime.__dict__.update(vars(sampled))
    return runtime, record.case


@pytest.fixture
def fake_cuda(monkeypatch, decoder):
    actions = []
    torch = decoder.api.torch
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: actions.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: actions.append(("empty", None)))
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: actions.append(("blas-clear", None)))
    return actions


def _three_queries(session):
    q = _queries("sparse_shuffled_duplicates")
    return session.source(q), session.rho0(q[:, 1:3]), session.g0(q[:, 1:])


def test_cpu_and_unknown_runtime_keep_original_session(physical, monkeypatch):
    runtime, case = physical
    def unexpected(*args):
        raise AssertionError("CPU runtime allocated an evaluation replica")
    monkeypatch.setattr(devices, "_VPFPDeviceRuntime", unexpected)
    with EvaluationDevicePool(("cuda:0",)) as pool, bind_evaluation_device_pool(pool):
        session = runtime.public_session_factory(case)
        assert session._runtime is runtime
        assert not getattr(session, "public_calls_thread_safe", False)
        session.close()
    assert case.case_id in runtime._resident_cases


def test_real_model_copy_exact_queries_rng_and_session_release(physical, decoder, fake_cuda):
    runtime, case = physical
    torch = runtime.api.torch
    rng = torch.get_rng_state().clone()
    original = runtime.public_session_factory(case)
    reference = _three_queries(original)
    resource = devices._VPFPDeviceRuntime(runtime, "cpu")
    replica = resource.runtime
    model_reference = weakref.ref(replica.expanded_model)
    assert replica._runtime_lock is not runtime._runtime_lock
    assert replica.expanded_model is not runtime.expanded_model
    for name, tensor in runtime.expanded_model.state_dict().items():
        copied = replica.expanded_model.state_dict()[name]
        torch.testing.assert_close(tensor, copied, rtol=0, atol=0)
        if tensor.numel():
            assert tensor.data_ptr() != copied.data_ptr()
    resident = runtime._active_resident(case)
    first, second = resource.session(resident), resource.session(resident)
    assert first.public_calls_thread_safe and second.public_calls_thread_safe
    for actual, expected in zip(_three_queries(first), reference):
        np.testing.assert_array_equal(actual, expected)
    first.close()
    assert case.case_id in replica._resident_cases
    second.close()
    assert not replica._resident_cases
    still_open = resource.session(resident)
    resource.close()
    assert still_open._runtime is None and resource.runtime is None
    with pytest.raises(RuntimeError, match="closed"):
        still_open.source(_queries("sparse_shuffled_duplicates"))
    gc.collect()
    assert model_reference() is None
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert case.case_id in runtime._resident_cases
    assert ("sync", "cpu") in fake_cuda and ("empty", None) in fake_cuda


def test_pool_routes_and_closes_after_replica_queries(physical, monkeypatch, fake_cuda):
    runtime, case = physical
    original_copy = devices._copy_runtime
    targets = []
    def copy_on_cpu(original, device):
        targets.append(device)
        return original_copy(original, "cpu")
    monkeypatch.setattr(devices, "_supports_replication", lambda model: True)
    monkeypatch.setattr(devices, "_copy_runtime", copy_on_cpu)
    pool = EvaluationDevicePool(("cuda:0", "cuda:1", "cuda:2"))
    with pytest.raises(ValueError, match="stop evaluation"):
        with pool, bind_evaluation_device_pool(pool):
            session = runtime.public_session_factory(case)
            another = runtime.public_session_factory(case)
            assert session._runtime is another._runtime
            assert targets == [pool.device_for(case.case_id)]
            _three_queries(session)
            raise ValueError("stop evaluation")
    assert pool.cleanup_confirmed
    assert session._runtime is None and another._runtime is None


def test_cuda_replica_all_devices_queries_and_memory(physical, decoder):
    runtime, case = physical
    torch = runtime.api.torch
    if not torch.cuda.is_available():
        pytest.skip("requires scheduled CUDA allocation")
    runtime.expanded_model.to("cuda:0")
    cpu_resident = runtime._active_resident(case)
    reference = _three_queries(runtime.public_session_factory(case))
    targets = tuple(f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 3)))
    # Normalize pre-existing library workspaces before the allocation baseline.
    # The after-scope assertion below relies solely on production cleanup.
    for target in targets:
        torch.cuda.synchronize(target)
    torch._C._cuda_clearCublasWorkspaces()
    for target in targets:
        with torch.cuda.device(target):
            torch.cuda.empty_cache()
    before = {target: torch.cuda.memory_allocated(target) for target in targets}
    rng = torch.get_rng_state().clone()
    cuda_rng = [torch.cuda.get_rng_state(target).clone() for target in targets]
    resources = []
    pool = EvaluationDevicePool(targets)
    try:
        with pool, bind_evaluation_device_pool(pool):
            register_cuda_workspace_cleanup(pool, torch)
            # Exercise each device even if this single Case hashes to one slot.
            pool.get_resource(("vpfp-coordinate-precision", id(runtime)),
                              lambda: devices._CoordinatePrecision(runtime))
            sessions = []
            for target in targets:
                resource = pool.get_resource(("vpfp-test", target),
                    lambda target=target: devices._VPFPDeviceRuntime(runtime, target))
                resources.append(resource)
                session = resource.session(cpu_resident)
                assert str(next(session._runtime.expanded_model.parameters()).device) == target
                sessions.append(session)
            # Independent per-device locks must safely admit concurrent kernels.
            with ThreadPoolExecutor(max_workers=len(targets)) as workers:
                actual_results = list(workers.map(_three_queries, sessions))
            for session, actual_result in zip(sessions, actual_results):
                for actual, expected in zip(actual_result, reference):
                    np.testing.assert_allclose(actual, expected, rtol=decoder.rtol, atol=decoder.atol)
                session.close()
            routed = runtime.public_session_factory(case)
            assert str(next(routed._runtime.expanded_model.parameters()).device) == pool.device_for(case.case_id)
            _three_queries(routed)
        gc.collect()
        assert all(resource.runtime is None for resource in resources)
        for target in targets:
            assert torch.cuda.memory_allocated(target) == before[target]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for target, expected in zip(targets, cuda_rng):
            torch.testing.assert_close(torch.cuda.get_rng_state(target), expected, rtol=0, atol=0)
    finally:
        runtime.expanded_model.to("cpu")


def test_failed_latent_copy_cannot_confirm_gpu_release(physical, monkeypatch, fake_cuda):
    from pde_agent_verifier_solver.plugins.vpfp import perturbation_backend
    runtime, case = physical
    resource = devices._VPFPDeviceRuntime(runtime, "cpu")
    pool = EvaluationDevicePool(("cuda:0",))
    pool.get_resource("vpfp", lambda: resource)
    def fail_latent(*args):
        partial_tensor = runtime.api.torch.as_tensor(runtime._active_resident(case).latent).clone()
        raise ValueError("latent copy failed")
    monkeypatch.setattr(perturbation_backend, "_latent_for_model", fail_latent)
    with pytest.raises(ValueError, match="latent copy failed"):
        resource.session(runtime._active_resident(case))
    with pytest.raises(RuntimeError, match="cleanup"):
        pool.close()
    assert resource.runtime is None
    assert not pool.cleanup_confirmed
    with pytest.raises(RuntimeError, match="cleanup"):
        resource.close()
