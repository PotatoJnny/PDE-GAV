"""GPU policy tests use mocks; actual CUDA qualification runs on Slurm only."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver import sdk
from pde_agent_verifier_solver.engine import candidate_execution as execution
from pde_agent_verifier_solver.engine.gpu_execution import (
    GpuDevicePool, initialize_cuda_device, synchronize_cuda_device,
    validate_cuda_metadata, validated_gpu_devices, validated_memory_fraction,
)
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from tests.test_candidate_execution import _artifact, _case, _query, _evaluation_settings


def _metadata(token="0"):
    return {"schema": "candidate-cuda-execution-v1", "device": "cuda:0",
            "cuda_visible_devices": token, "uuid": "1234", "name": "Fake GPU",
            "device_count": 1, "readiness_kernel_passed": True, "synchronized": True,
            "peak_memory_allocated_bytes": 128, "peak_memory_reserved_bytes": 256}


@pytest.mark.parametrize("value", ["0", [""], ["0,1"], [" 0"], ["-1"], ["0", "0"], [True]])
def test_device_tokens_are_explicit_distinct_and_not_environment_strings(value):
    with pytest.raises((ValueError, TypeError)):
        validated_gpu_devices(value)


def test_device_token_forms():
    assert validated_gpu_devices(("0", "GPU-abc-123", "MIG-GPU-abc/1/0")) == (
        "0", "GPU-abc-123", "MIG-GPU-abc/1/0")


@pytest.mark.parametrize("value", [True, 0, -0.1, 1.1, float("inf"), float("nan"), "0.5"])
def test_allocator_fraction_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        validated_memory_fraction(value)


def test_gpu_pool_releases_and_quarantines_exact_leases():
    pool = GpuDevicePool(("2", "3"))
    assert pool.acquire(timeout=0) == "2"
    assert pool.acquire(timeout=0) == "3"
    assert pool.acquire(timeout=0) is None
    pool.release("2")
    assert pool.acquire(timeout=0) == "2"
    pool.release("2", healthy=False)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire(timeout=0, devices=("2",))
    pool.release("3")
    assert pool.snapshot() == {"devices": ["2", "3"], "leased": [],
                               "quarantined": ["2"], "slots_per_device": 1}
    with pytest.raises(ValueError, match="no active"):
        pool.release("3")


def test_gpu_pool_has_one_global_slot_across_callers():
    pool = GpuDevicePool(("0",))
    assert pool.acquire(timeout=0) == "0"
    waiting = Event()

    def acquire():
        waiting.set()
        return pool.acquire(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(acquire)
        assert waiting.wait(1)
        assert not future.done()
        pool.release("0")
        assert future.result(timeout=2) == "0"
    pool.release("0")


def _service(tmp_path, **kwargs):
    return execution.SubprocessCandidateExecutionService(
        tmp_path, task=execution.PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings={}, max_workers=2, case_hard_wall_seconds=900, **kwargs)


def test_default_cpu_policy_never_inherits_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    service = _service(tmp_path)
    assert service.gpu_devices == () and service.gpu_device_pool is None
    assert service._worker_environment(tmp_path)["CUDA_VISIBLE_DEVICES"] == ""


def test_explicit_shared_pool_requires_explicit_device_subset(tmp_path):
    pool = GpuDevicePool(("0", "1"))
    with pytest.raises(ValueError, match="explicitly select"):
        _service(tmp_path / "implicit", gpu_device_pool=pool)
    with pytest.raises(ValueError, match="explicitly select"):
        _service(tmp_path / "foreign", gpu_devices=("2",), gpu_device_pool=pool)
    service = _service(tmp_path / "valid", gpu_devices=("1",), gpu_device_pool=pool)
    assert service.gpu_device_pool is pool
    # Visibility is assigned only after an exclusive lease, never pre-exposed.
    assert service._worker_environment(tmp_path)["CUDA_VISIBLE_DEVICES"] == ""


def test_cancelled_gpu_queue_never_starts_or_charges_case_worker(tmp_path, monkeypatch):
    waiting = Event()

    class ObservedPool(GpuDevicePool):
        def acquire(self, **kwargs):
            waiting.set()
            return super().acquire(**kwargs)

    pool = ObservedPool(("0",))
    assert pool.acquire(timeout=0) == "0"
    waiting.clear()
    cancel = Event()
    query = _query()
    settings = _evaluation_settings(query)
    service = execution.SubprocessCandidateExecutionService(
        tmp_path / "execution", task=execution.PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=1, case_hard_wall_seconds=900,
        gpu_devices=("0",), gpu_device_pool=pool)
    plan = sdk.EvaluationPlan(phase=sdk.EvaluationPhase.DEVELOPMENT,
                              query=query, seed=0, settings=settings)

    def forbidden_process(*args, **kwargs):
        pytest.fail("A queued GPU Case must not launch a worker")

    monkeypatch.setattr(execution.subprocess, "Popen", forbidden_process)
    artifact = _artifact(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.execute_cases, artifact, (_case("one", value=1),),
                                 (plan,), NSSolverTask().create_evaluation(settings),
                                 cancellation_event=cancel)
        assert waiting.wait(2)
        cancel.set()
        with pytest.raises(execution.CandidateExecutionCancelled):
            future.result(timeout=3)
    assert len(service.audits) == 1
    audit = service.audits[0]
    assert audit.execution_started is False and audit.charged_wall_seconds == 0
    assert audit.gpu_queue_wait_seconds >= 0 and audit.gpu_device is None
    assert pool.snapshot()["leased"] == ["0"]  # The original owner's lease remains.
    pool.release("0")


class _Tensor:
    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 4096.0


def _fake_torch(monkeypatch, *, available=True, count=1, uuid="1234"):
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: available, device_count=lambda: count,
        set_device=lambda device: calls.append(("set", device)),
        set_per_process_memory_fraction=lambda fraction, device: calls.append(("fraction", fraction)),
        synchronize=lambda device: calls.append(("sync", device)),
        get_device_properties=lambda device: SimpleNamespace(uuid=uuid, name="Fake GPU"),
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        max_memory_allocated=lambda device: 128, max_memory_reserved=lambda device: 256)
    torch = SimpleNamespace(cuda=cuda, ones=lambda *args, **kwargs: _Tensor())
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_readiness_requires_real_kernel_identity_and_final_synchronization(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    calls = _fake_torch(monkeypatch)
    metadata = initialize_cuda_device("GPU-1234", memory_fraction=.5)
    assert metadata["synchronized"] is False
    assert ("fraction", .5) in calls and ("sync", 0) in calls
    synchronize_cuda_device(metadata)
    assert validate_cuda_metadata(metadata, "GPU-1234")["peak_memory_allocated_bytes"] == 128
    assert calls.count(("sync", 0)) == 2


@pytest.mark.parametrize("available,count,uuid", [(False, 1, "1234"), (True, 2, "1234"), (True, 1, ""), (True, 1, "9999")])
def test_readiness_never_falls_back_to_cpu(monkeypatch, available, count, uuid):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    _fake_torch(monkeypatch, available=available, count=count, uuid=uuid)
    with pytest.raises(RuntimeError):
        initialize_cuda_device("GPU-1234")


@pytest.mark.parametrize("changed", [{"synchronized": False}, {"uuid": ""},
                                    {"readiness_kernel_passed": False},
                                    {"cuda_visible_devices": "1"},
                                    {"peak_memory_allocated_bytes": -1}])
def test_controller_requires_gpu_evidence(changed):
    with pytest.raises(ValueError):
        validate_cuda_metadata({**_metadata(), **changed}, "0")


@pytest.mark.parametrize("failure", [None, "candidate", "readiness", "sync", "candidate_and_sync"])
def test_worker_gpu_lifecycle_synchronizes_even_after_candidate_failure(tmp_path, monkeypatch, failure):
    case = sdk.Case("one", sdk.CaseSource.GENERATED, sdk.CaseVisibility.DEVELOPMENT, {})
    calls = []
    payload = {
        "schema": execution.CANDIDATE_WORKER_REQUEST_SCHEMA, "execution_id": "gpu-test",
        "case_index": 0, "sidecar_root": str(tmp_path / "arrays"),
        "evaluation_program_id": "gpu-test", "public_oracle": {},
        "device_policy": {"schema": "candidate-device-policy-v1", "device": "cuda",
                          "assigned_gpu": "0", "memory_fraction": None}}

    def initialize(*args, **kwargs):
        calls.append("readiness")
        if failure == "readiness":
            raise RuntimeError("CUDA unavailable")
        return _metadata()

    def synchronize(metadata):
        calls.append("sync")
        if failure in ("sync", "candidate_and_sync"):
            raise RuntimeError("CUDA failure")

    def execute(*args):
        calls.append("execute")
        if failure in ("candidate", "candidate_and_sync"):
            raise ValueError("candidate failure")
        return object()

    evaluation = SimpleNamespace(program_id="gpu-test", supports_split_candidate_execution=True,
                                 execute_candidate=execute)
    monkeypatch.setattr(execution, "_read_bounded_json", lambda *args: payload)
    monkeypatch.setattr(execution, "_task_from_payload", lambda *args: object())
    monkeypatch.setattr(execution, "_settings_from_payload", lambda *args: {})
    monkeypatch.setattr(execution, "_artifact_from_payload", lambda *args: object())
    monkeypatch.setattr(execution, "_case_from_payload", lambda *args: case)
    monkeypatch.setattr(execution, "_plan_from_payload", lambda *args: object())
    monkeypatch.setattr(sdk, "resolve_evaluation_program", lambda *args: evaluation)
    monkeypatch.setattr(execution, "CandidatePublicSessionClient", lambda *args: SimpleNamespace(
        session_factory=None, available=False, methods=()))
    monkeypatch.setattr(execution, "_require_candidate_output", lambda value: value)
    monkeypatch.setattr(execution, "_candidate_output_to_payload", lambda *args: {})
    monkeypatch.setattr(execution, "initialize_cuda_device", initialize)
    monkeypatch.setattr(execution, "synchronize_cuda_device", synchronize)
    response = tmp_path / "response.json"
    result = execution.run_candidate_worker(tmp_path / "request.json", response)
    value = json.loads(response.read_text())
    assert result == (0 if failure is None else 1)
    assert calls == (["readiness"] if failure == "readiness" else ["readiness", "execute", "sync"])
    if failure in ("readiness", "sync"):
        assert value["error_code"] == "candidate_execution_infrastructure_failure"
    elif failure in ("candidate", "candidate_and_sync"):
        assert value["error_type"] == "ValueError" and value["device_execution"]["uuid"] == "1234"
        if failure == "candidate_and_sync":
            assert "GPU synchronization also failed: CUDA failure" in value["traceback"]
    else:
        assert value["device_execution"]["readiness_kernel_passed"] is True
