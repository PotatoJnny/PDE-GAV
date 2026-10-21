"""Run on an allocated GPU: concurrent NS Oracle streams and batched solvers."""
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a scheduled CUDA allocation')


def test_four_ns_oracle_calls_use_distinct_streams_and_preserve_values(monkeypatch):
    from tests.test_ns_oracle_numerical_hot_path import _field, _session, _points, analytic, backend
    from pde_agent_verifier_solver.engine.fair_execution import bind_oracle_policy
    torch.set_num_threads(1)
    answer = backend.StructuredNSMMSAnswer(_field('cuda:0'))
    sessions = [_session(answer, str(i)) for i in range(4)]
    points = _points()[:, (2,0,1)]
    reference = [s.query_forcing(points) for s in sessions]
    original = analytic.analytic_forcing
    barrier, lock = Barrier(4), Lock()
    streams = set()
    def compute(*args, **kwargs):
        with lock: streams.add(torch.cuda.current_stream().cuda_stream)
        barrier.wait(timeout=15)
        return original(*args, **kwargs)
    monkeypatch.setattr(analytic, 'analytic_forcing', compute)
    def query(i):
        with bind_oracle_policy(f'agent-{i%2}', 4):
            return sessions[i].query_forcing(points)
    try:
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(query, range(4)))
        assert len(streams) == 4
        for result, expected in zip(results, reference):
            np.testing.assert_allclose(result, expected, rtol=2e-10, atol=1e-9)
    finally:
        for session in sessions: session.close()


def test_two_batched_solver_workers_share_allocated_gpu_and_compute_on_cuda(tmp_path):
    from tests.test_batch_solver_execution import _batch_artifact, _case, _query, _evaluation_settings
    from pde_agent_verifier_solver.engine import PythonTaskWorkerSpec, SubprocessCandidateExecutionService, PythonCandidateCodec
    from pde_agent_verifier_solver.plugins.ns import NSSolverTask
    from pde_agent_verifier_solver.sdk import TaskKind, write_candidate_bundle_manifest, EvaluationPlan, EvaluationPhase
    # Build a normal frozen bundle; the batch function does a single CUDA tensor operation.
    _batch_artifact(tmp_path)
    source = tmp_path/'source'
    with (source/'submission.py').open('a') as stream:
        stream.write('''
_original_batch = prepare_solutions
def prepare_solutions(cases, solve_budget=None):
    import torch
    assert torch.cuda.is_available()
    values = torch.tensor([c.public_data['value'] for c in cases], device='cuda:0', dtype=torch.float64)
    values = (values * 2 + 1).cpu().tolist()
    import time
    time.sleep(.2)
    return {c.case_id: Handle(v) for c,v in zip(cases,values)}
''')
    write_candidate_bundle_manifest(source, files=('submission.py',), overwrite=True)
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(source,
        storage_root=tmp_path/'gpu_sealed',artifact_id='batch-cuda',name='CUDA',summary='batch',usage='batch')
    token = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
    settings = _evaluation_settings(_query())
    service = SubprocessCandidateExecutionService(tmp_path/'cuda_execution',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),evaluation_settings=settings,
        max_workers=2,solver_batch_size=4,oracle_max_workers=4,case_hard_wall_seconds=30,
        gpu_devices=(token,),gpu_slots_per_device=2,gpu_memory_fraction=.4)
    plan = EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT, query=_query(),settings=settings)
    try:
        cases = tuple(_case(str(i),value=i) for i in range(8))
        outputs = service.execute_cases(artifact,cases,(plan,)*8,NSSolverTask().create_evaluation(settings))
        assert [o.prediction.fields['velocity_x'][0] for o in outputs] == [2*i+1 for i in range(8)]
        assert len({a.pid for a in service.audits}) == 2
        assert all(a.gpu_metadata['readiness_kernel_passed'] and a.gpu_metadata['synchronized'] for a in service.audits)
    finally: service._executor.shutdown(wait=True,cancel_futures=True)


def test_four_cached_ns_oracle_queries_have_separate_graph_buffers(monkeypatch):
    from tests.test_ns_oracle_numerical_hot_path import _field, _session, _points, backend
    from pde_agent_verifier_solver.plugins.ns.analytic_cuda import NSAnalyticCudaCache
    from pde_agent_verifier_solver.engine.fair_execution import bind_oracle_policy
    torch.set_num_threads(1)
    monkeypatch.setattr(NSAnalyticCudaCache, '_capture_after_occurrences', {'forcing':1,'velocity':1})
    cache=NSAnalyticCudaCache('cuda:0')
    answers=[]
    for i in range(4):
        field=_field('cuda:0')
        field.coefficients.psi.mul_(i+1)
        answers.append(backend.StructuredNSMMSAnswer(field))
    sessions=[_session(answer,str(i)) for i,answer in enumerate(answers)]
    points=_points()[:,(2,0,1)]
    expected=[session.query_forcing(points) for session in sessions]
    for answer in answers: answer._analytic_cuda_cache=cache
    def query(i):
        with bind_oracle_policy(f'agent-{i%2}',4):
            return sessions[i].query_forcing(points)
    try:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(query,range(4)))
        assert cache.diagnostics['capture_count'] == 4
        original=NSAnalyticCudaCache._replay
        gate,lock=Barrier(4),Lock()
        buffers=set()
        def overlap(self,entry,field,coordinates):
            with lock:
                buffers.add((entry.coordinates.data_ptr(),entry.output.data_ptr(),self._stream.cuda_stream))
            gate.wait(timeout=15)
            return original(self,entry,field,coordinates)
        monkeypatch.setattr(NSAnalyticCudaCache,'_replay',overlap)
        with ThreadPoolExecutor(4) as pool:
            actual=list(pool.map(query,range(4)))
        assert len(buffers) == 4
        for result,reference in zip(actual,expected):
            np.testing.assert_allclose(result,reference,rtol=2e-10,atol=1e-9)
        retained=[result.copy() for result in actual]
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(query,reversed(range(4))))
        for result,reference in zip(actual,retained):
            np.testing.assert_array_equal(result,reference)
        assert cache.diagnostics['parallel_lanes'] == 4
        assert cache.diagnostics['entries'] <= 4*cache.max_entries
    finally:
        for session in sessions: session.close()
        cache.close()
    assert cache.diagnostics['closed']
    assert cache.diagnostics['entries'] == 0


def test_parallel_cache_numerical_failure_is_sticky_and_cleanup_reports_it(monkeypatch):
    from tests.test_ns_oracle_numerical_hot_path import _field, _session, _points, backend
    from pde_agent_verifier_solver.plugins.ns.analytic_cuda import NSAnalyticCudaCache
    from pde_agent_verifier_solver.engine.fair_execution import bind_oracle_policy
    cache=NSAnalyticCudaCache('cuda:0')
    answer=backend.StructuredNSMMSAnswer(_field('cuda:0'))
    answer._analytic_cuda_cache=cache
    session=_session(answer)
    def fail(*args,**kwargs): raise RuntimeError('injected numerical failure')
    monkeypatch.setattr(NSAnalyticCudaCache,'_compute',fail)
    try:
        with bind_oracle_policy('agent-a',2):
            with pytest.raises(RuntimeError,match='injected numerical failure'):
                session.query_forcing(_points()[:,(2,0,1)])
            assert cache.diagnostics['failed']
            with pytest.raises(RuntimeError,match='unavailable'):
                session.query_forcing(_points()[:,(2,0,1)])
    finally:
        session.close()
        with pytest.raises(RuntimeError,match='cleanup unconfirmed'):
            cache.close()
    assert cache.diagnostics['closed']
    assert cache.diagnostics['entries'] == 0
