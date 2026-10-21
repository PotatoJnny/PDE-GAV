from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import importlib.util
import os
import time

import pytest

from pde_agent_verifier_solver.engine import (
    CandidateExecutionCancelled, CandidateExecutionError, PythonCandidateCodec,
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan, TaskKind
from tests.test_candidate_execution import _candidate_bundle, _case, _evaluation_settings, _query


def _artifact(root):
    bundle = _candidate_bundle(root / 'source')
    source = bundle / 'submission.py'
    text = source.read_text().replace('    del solve_budget\n', '''    del solve_budget
    import os
    from pathlib import Path
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible == case.public_data["expected_gpu"]
    marker = case.public_data.get("started_marker")
    if marker:
        Path(marker).write_text("started")
    lock = case.public_data.get("device_lock")
    if lock:
        # A second simultaneous worker on this GPU would fail O_EXCL.
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            time.sleep(0.15)
        finally:
            os.unlink(lock)
    if case.public_data.get("real_cuda"):
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        gpu_values = torch.full((32,), 0.25, device="cuda:0", dtype=torch.float64)
        assert gpu_values.sum().item() == 8.0
''')
    source.write_text(text)
    return PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        bundle, storage_root=root / 'sealed', artifact_id='candidate', name='GPU candidate',
        summary='Checks GPU visibility and executes a real CUDA operation when requested.',
        usage='prepare then evaluate',
    )


def _service(root, devices=(), *, environment=None):
    return SubprocessCandidateExecutionService(
        root, task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=_evaluation_settings(_query()), max_workers=3,
        case_hard_wall_seconds=60, gpu_devices=devices, environment=environment,
    )


@pytest.fixture(autouse=True)
def _mock_worker_cuda_for_cpu_tests(tmp_path, monkeypatch, request):
    if request.node.name.startswith("test_real_cuda"):
        return
    mock_root = tmp_path / "mock_torch"
    mock_root.mkdir()
    (mock_root / "torch.py").write_text("""from types import SimpleNamespace
class Tensor:
    def __matmul__(self, other): return self
    def sum(self): return self
    def item(self): return 4096.0
def ones(*a, **kw): return Tensor()
cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
    set_device=lambda *a: None, synchronize=lambda *a: None,
    get_device_properties=lambda *a: SimpleNamespace(uuid='test-only', name='Mock CUDA'),
    reset_peak_memory_stats=lambda *a: None, max_memory_allocated=lambda *a: 0,
    max_memory_reserved=lambda *a: 0)
""")
    original = _service
    def mocked(root, devices=()):
        environment = {"PYTHONPATH": str(mock_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        return original(root, devices, environment=environment)
    monkeypatch.setattr(__import__(__name__, fromlist=["_service"]), "_service", mocked)


def _run(service, artifact, cases, phase=EvaluationPhase.OFFICIAL, cancellation=None):
    settings = _evaluation_settings(_query())
    plans = tuple(EvaluationPlan(phase=phase, query=_query(), seed=i, settings=settings)
                  for i in range(len(cases)))
    return service.execute_cases(
        artifact, cases, plans, NSSolverTask().create_evaluation(settings),
        cancellation_event=cancellation,
    )


@pytest.mark.parametrize('devices', ['0', ('0', '0'), ('bad',), (True,), ('',)])
def test_rejects_invalid_or_duplicate_device_assignments(tmp_path, devices):
    with pytest.raises((ValueError, TypeError)):
        _service(tmp_path, devices)


def test_never_expands_cuda_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(ValueError, match='already be CUDA-visible'):
        _service(tmp_path, ('0',))


def test_default_cpu_remains_cpu_even_when_parent_has_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    service = _service(tmp_path / 'execution')
    try:
        output = _run(service, _artifact(tmp_path), (_case('cpu', expected_gpu='', value=3),))
        assert output[0].prediction.fields['velocity_x'] == (3.0, 3.0)
        assert service.audits[0].gpu_device is None
    finally:
        service._executor.shutdown(wait=True, cancel_futures=True)


def test_one_worker_per_gpu_across_concurrent_batches_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5')
    service = _service(tmp_path / 'execution', ('4',))
    artifact = _artifact(tmp_path)
    cases = [_case(f'gpu-{i}', expected_gpu='4', value=i, device_lock=str(tmp_path / 'gpu.lock'))
             for i in (1, 2)]
    try:
        with ThreadPoolExecutor(2) as executor:
            futures = [executor.submit(_run, service, artifact, (case,)) for case in cases]
            values = [future.result(timeout=45)[0].prediction.fields['velocity_x'][0] for future in futures]
        assert values == [1.0, 2.0]
        with pytest.raises(CandidateExecutionError):
            _run(service, artifact, (_case('failed', expected_gpu='4', value=0, fail=True),))
        output = _run(service, artifact, (_case('recovered', expected_gpu='4', value=7),))
        assert output[0].prediction.fields['velocity_x'] == (7.0, 7.0)
        assert all(a.gpu_device == '4' and a.process_group_cleaned for a in service.audits)
    finally:
        service._executor.shutdown(wait=True, cancel_futures=True)


def test_waiting_for_gpu_is_cancellable_without_interrupting_owner(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    service = _service(tmp_path / 'execution', ('4',))
    artifact = _artifact(tmp_path)
    started = tmp_path / 'started'
    cancellation = Event()
    try:
        with ThreadPoolExecutor(2) as executor:
            owner = executor.submit(_run, service, artifact,
                (_case('owner', expected_gpu='4', value=2, sleep_seconds=1.5, started_marker=str(started)),))
            deadline = time.monotonic() + 30
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert started.exists()
            queued = executor.submit(_run, service, artifact,
                (_case('queued', expected_gpu='4', value=3),), EvaluationPhase.DEVELOPMENT, cancellation)
            time.sleep(0.15)
            cancellation.set()
            with pytest.raises(CandidateExecutionCancelled):
                queued.result(timeout=15)
            assert owner.result(timeout=30)[0].prediction.fields['velocity_x'] == (2.0, 2.0)
        audit = next(a for a in service.audits if a.case_id == 'queued')
        assert audit.pid is None and audit.charged_wall_seconds == 0
        assert service.gpu_device_pool.snapshot()["leased"] == []
    finally:
        service._executor.shutdown(wait=True, cancel_futures=True)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_prompt_and_status_match_enabled_execution():
    from pde_agent_verifier_solver.engine.agent_tool import _status
    root = Path(__file__).resolve().parents[1]
    worker = _module(root / 'scripts/run_canonical_codex_agent.py', '_gpu_prompt_worker')
    request = {
        'agent': {'prompt': 'Improve the solver.'},
        'round': {'round_index': 1, 'hard_wall_seconds': 14400,
                  'development_sampling': {'max_cases': 512, 'allowed_sources': ['base']}},
        'worker_settings': {'candidate_execution': {'device': 'cpu_or_cuda', 'cuda_available': True}},
        'candidate_execution': {'case_charged_wall_seconds': 900},
    }
    prompt = worker._research_prompt(request)
    assert 'CPU and CUDA' in prompt and 'cuda:0' in prompt
    assert 'Use GPU acceleration' in prompt
    assert 'must execute correctly without CUDA' not in prompt
    assert 'CPU-only subprocesses' not in prompt
    assert _status(request, ())['candidate_execution']['cuda_available'] is True
    runner = _module(root / 'scripts/run_canonical_ns_campaign.py', '_gpu_ns_runner')
    assert runner._candidate_device({}) == 'cpu'
    assert runner._candidate_device({'candidate_device': 'cpu_or_cuda'}) == 'cpu_or_cuda'
    with pytest.raises(ValueError, match='candidate_device'):
        runner._candidate_device({'candidate_device': 'auto'})


@pytest.mark.skipif(os.environ.get('PDE_REAL_GPU_TEST') != '1', reason='requires an allocated CUDA GPU')
@pytest.mark.parametrize('phase', list(EvaluationPhase))
def test_real_cuda_solver_uses_worker_gpu_in_every_phase(tmp_path, phase):
    token = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
    service = _service(tmp_path / 'execution', (token,))
    try:
        output = _run(service, _artifact(tmp_path),
                      (_case('cuda', expected_gpu=token, real_cuda=True, value=5),), phase)
        assert output[0].prediction.fields['velocity_x'] == (5.0, 5.0)
        assert service.audits[0].gpu_device == token
        assert service.audits[0].status == 'completed'
    finally:
        service._executor.shutdown(wait=True, cancel_futures=True)
