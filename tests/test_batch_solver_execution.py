"""Cross-Case solver calls, concurrent Oracle work and shared time attribution."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
import time
import pytest

from pde_agent_verifier_solver.engine import (PythonCandidateCodec, PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService, CandidateExecutionTimeout, CandidateExecutionCancelled)
from pde_agent_verifier_solver.engine.candidate_execution import _CandidateOracleHost
from pde_agent_verifier_solver.engine.fair_execution import FairExecutor, FairAdmission, bind_oracle_policy
from pde_agent_verifier_solver.engine.gpu_execution import GpuDevicePool
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan, TaskKind, write_candidate_bundle_manifest
from pde_agent_verifier_solver.sdk.solve_budget import CandidateSolveBudgetMeter, ORACLE_WAIT_TIMING_SCHEMA
from tests.test_candidate_execution import _case, _query, _evaluation_settings, _candidate_bundle


def _batch_artifact(tmp_path, *, bad_keys=False):
    source = _candidate_bundle(tmp_path / 'source')
    entry = source / 'submission.py'
    with entry.open('a') as stream:
        stream.write('''

def prepare_solutions(cases, solve_budget=None):
    from pde_agent_verifier_solver.sdk.public_sessions import query_public_cases
    assert solve_budget is not None and solve_budget > 0
    if cases[0].public_data.get('sleep_batch'):
        time.sleep(cases[0].public_data['sleep_batch'])
    values = {c.case_id: c.public_data['value'] for c in cases}
    if cases[0].public_data.get('use_oracle'):
        values = query_public_cases(cases, 'marker', values)
    return {key: Handle(value) for key, value in values.items()}
''')
        if bad_keys:
            stream.write('''
_original_batch = prepare_solutions
def prepare_solutions(cases, solve_budget=None):
    result = _original_batch(cases, solve_budget)
    del result[cases[0].case_id]
    return result
''')
    write_candidate_bundle_manifest(source, files=('submission.py',), overwrite=True)
    return PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path/'sealed', artifact_id='batch', name='Batch',
        summary='one vectorized prepare call', usage='batch')


def _service(tmp_path, *, workers=2, batch=4, oracle=4, budget=10):
    settings = _evaluation_settings(_query())
    evaluation = NSSolverTask().create_evaluation(settings)
    service = SubprocessCandidateExecutionService(tmp_path/'execution',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask), evaluation_settings=settings,
        max_workers=workers, solver_batch_size=batch, oracle_max_workers=oracle,
        oracle_concurrency_per_device=oracle, case_hard_wall_seconds=budget)
    plan = EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT, query=_query(), settings=settings)
    return service, evaluation, plan


def test_real_batch_worker_queries_entire_batch_concurrently_and_preserves_cases(tmp_path):
    artifact = _batch_artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path)
    cases = tuple(_case(str(i), value=i, use_oracle=True) for i in range(4))
    gate, lock = Barrier(4), Lock()
    active = peak = 0
    closed = []
    def invoke(case_id, method, args, kwargs):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
        try:
            gate.wait(timeout=5)  # Serial Oracle work cannot pass this test.
            time.sleep(.04)
            return args[0] + 1
        finally:
            with lock: active -= 1
    try:
        outputs = service.execute_cases(artifact, cases, (plan,)*4, evaluation,
            public_session_methods=('marker',), invoke_public_session=invoke,
            close_public_session=closed.append)
        assert [o.prediction.fields['velocity_x'][0] for o in outputs] == [1,2,3,4]
        assert peak == 4 and active == 0
        assert len(service.audits) == 4
        assert len({a.pid for a in service.audits}) == 1
        assert all(a.solver_batch_size == 4 and a.timeout_seconds == 40 for a in service.audits)
        assert all(a.oracle_call_count == 1 for a in service.audits)
        assert sorted(closed) == ['0','1','2','3']
    finally: service._executor.shutdown(wait=True, cancel_futures=True)


def test_batch_and_single_solve_have_equal_predictions_including_remainder(tmp_path):
    artifact = _batch_artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path)
    cases = tuple(_case(str(i), value=i) for i in range(7))
    try:
        batch = service.execute_cases(artifact, cases, (plan,)*7, evaluation)
        service.solver_batch_size = 1
        single = service.execute_cases(artifact, cases, (plan,)*7, evaluation)
        assert [o.prediction for o in batch] == [o.prediction for o in single]
        assert [a.solver_batch_size for a in service.audits[:7]].count(4) == 4
        assert [a.solver_batch_size for a in service.audits[:7]].count(3) == 3
    finally: service._executor.shutdown(wait=True, cancel_futures=True)


def test_wrong_case_keys_fail_all_members_without_fabricating_outputs(tmp_path):
    from pde_agent_verifier_solver.engine import CandidateExecutionError
    artifact = _batch_artifact(tmp_path, bad_keys=True)
    service, evaluation, plan = _service(tmp_path)
    try:
        with pytest.raises(CandidateExecutionError):
            service.execute_cases(artifact, tuple(_case(str(i), value=i) for i in range(4)), (plan,)*4, evaluation)
        assert len(service.audits) == 4
        assert all(a.status == 'failed' and a.process_group_cleaned for a in service.audits)
    finally: service._executor.shutdown(wait=True, cancel_futures=True)


def test_round_robin_prevents_large_request_from_draining_before_new_request():
    executor = FairExecutor(1)
    release, entered = Event(), Event()
    order = []
    def run(label, block=False):
        order.append(label)
        if block:
            entered.set(); assert release.wait(4)
    try:
        first = executor.submit_for('a', run, 'a0', True)
        assert entered.wait(2)
        others = [executor.submit_for('a', run, f'a{i}') for i in range(1,8)]
        b = executor.submit_for('b', run, 'b')
        release.set()
        for f in [first, *others, b]: f.result(timeout=3)
        assert order.index('b') <= 2
    finally: release.set(); executor.shutdown(cancel_futures=True)


def test_shared_gpu_slots_are_bounded_and_quarantine_stops_new_admission():
    pool = GpuDevicePool(('0',), slots_per_device=2)
    assert pool.acquire(timeout=0) == '0'
    assert pool.acquire(timeout=0) == '0'
    assert pool.acquire(timeout=0) is None
    pool.release('0', healthy=False)
    with pytest.raises(RuntimeError): pool.acquire(timeout=0)
    pool.release('0')
    assert pool.snapshot()['leased'] == []


def test_wait_refund_is_intersection_not_sum_or_union(tmp_path):
    host = _CandidateOracleHost(tmp_path/'arrays', case_ids=('a','b'), methods=(), invoke=None, close=None)
    state = host.state
    state.register_wait_group(('a','b'))
    try:
        state._observe_wait('a', True); time.sleep(.02); state._observe_wait('a', False)
        state._observe_wait('b', True); time.sleep(.02)
        assert state.group_wait_seconds(('a','b')) == 0
        state._observe_wait('a', True); time.sleep(.03)
        state._observe_wait('a', False); state._observe_wait('b', False)
        refunded = state.group_wait_seconds(('a','b'))
        assert .025 <= refunded < .06
        assert refunded < state.queue_wait_seconds('a')
        assert refunded < state.queue_wait_seconds('b')
    finally: host.close()


def test_batch_meter_does_not_add_overlapping_or_out_of_order_refunds():
    meter = CandidateSolveBudgetMeter(budget_seconds=30, started_at_monotonic=time.monotonic()-2, batch_mode=True)
    for index, refund in enumerate((.5, .8, .6)):
        meter.record_oracle_response(str(index), {'schema': ORACLE_WAIT_TIMING_SCHEMA,
            'queue_wait_seconds': 1., 'lock_wait_seconds': 0., 'group_wait_seconds': refund}, rpc_wall_seconds=1.1)
    assert meter.snapshot()['candidate_platform_wait_seconds'] == .8


def test_batch_timeout_cleans_all_cases_and_service_can_run_again(tmp_path):
    artifact = _batch_artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path, budget=.5)
    cases = tuple(_case(str(i), value=i, sleep_batch=3) for i in range(4))
    try:
        with pytest.raises(CandidateExecutionTimeout):
            service.execute_cases(artifact, cases, (plan,)*4, evaluation)
        assert len(service.audits) == 4
        assert all(a.process_group_cleaned for a in service.audits)
        outputs = service.execute_cases(artifact, tuple(_case(str(i), value=i) for i in range(4)), (plan,)*4, evaluation)
        assert len(outputs) == 4
    finally: service._executor.shutdown(wait=True, cancel_futures=True)


def test_ns_cuda_admission_allows_parallel_queries_under_configured_policy():
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSCUDAQueryAdmission
    admission = _NSCUDAQueryAdmission()
    rendezvous = Barrier(4)
    def run(i):
        with bind_oracle_policy(str(i % 2), 4), admission.for_device(0):
            rendezvous.wait(timeout=3)
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(run, range(4)))


def test_cancel_one_agents_batch_keeps_other_agents_batch_and_oracle_usable(tmp_path):
    artifact = _batch_artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path, oracle=2)
    cancel = Event(); entered = Event(); lock = Lock()
    active = peak = 0
    def invoke(case_id, method, args, kwargs):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
        try:
            entered.set(); time.sleep(.12)
            return args[0] + 1
        finally:
            with lock: active -= 1
    def run(prefix, cancellation=None):
        cases = tuple(_case(f'{prefix}{i}', value=i, use_oracle=True) for i in range(4))
        return service.execute_cases(artifact, cases, (plan,)*4, evaluation,
            public_session_methods=('marker',), invoke_public_session=invoke,
            close_public_session=lambda key: None, cancellation_event=cancellation)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            a = workers.submit(run, 'a', cancel)
            assert entered.wait(5)
            b = workers.submit(run, 'b')
            cancel.set()
            with pytest.raises(CandidateExecutionCancelled): a.result(timeout=8)
            assert [row.prediction.fields['velocity_x'][0] for row in b.result(timeout=8)] == [1,2,3,4]
        assert peak == 2 and active == 0
        assert len(run('c')) == 4
        assert len(service.audits) == 12
        assert all(a.process_group_cleaned for a in service.audits)
    finally: service._executor.shutdown(wait=True, cancel_futures=True)


def test_batch_session_cannot_open_unassigned_case_and_is_revoked_after_scope():
    from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_batch, open_public_case_session
    closed = []
    class Session:
        def marker(self, value): return value
        def close(self): closed.append(True)
    a,b = _case('a'),_case('b')
    with bind_public_session_batch({'a':lambda _: Session(), 'b':lambda _: Session()}, (a,b), allowed_methods=('marker',)):
        proxy = open_public_case_session(a)
        assert proxy.marker(1) == 1
        with pytest.raises(PermissionError): open_public_case_session(_case('foreign'))
    with pytest.raises(RuntimeError): proxy.marker(2)
    assert closed == [True]


def test_perturbed_ns_cases_share_read_only_parallel_path_and_preserve_values(monkeypatch):
    import numpy as np
    from tests.test_ns_oracle_numerical_hot_path import _field, _session, _points, analytic, backend
    from agent_verifier.neural_generator.structured_ns_robustness import make_perturbation, field_at_amplitude
    parent = _field()
    noise = make_perturbation(parent, family="spectral_complement", seed=102, spectral_max_mode=4)
    child = field_at_amplitude(parent, noise, amplitude=1e-5)
    sessions = [_session(backend.StructuredNSMMSAnswer(field), str(i))
                for i, field in enumerate((parent, child))]
    points = _points()[:, (2, 0, 1)]
    expected = [session.query_forcing(points) for session in sessions]
    assert all(session.public_calls_thread_safe for session in sessions)
    barrier = Barrier(2)
    original = analytic.analytic_forcing
    def concurrent(*args, **kwargs):
        barrier.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(analytic, 'analytic_forcing', concurrent)
    try:
        with ThreadPoolExecutor(2) as executor:
            actual = list(executor.map(lambda session: session.query_forcing(points), sessions))
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result, reference, rtol=0., atol=0.)
    finally:
        for session in sessions: session.close()


def test_cancelled_wait_for_graph_lane_does_not_leak_lane(monkeypatch):
    from pde_agent_verifier_solver.plugins.ns import analytic_cuda
    from pde_agent_verifier_solver.sdk._candidate_timing import bind_queue_wait_observer
    class Cache:
        def __init__(self,*args,**kwargs): self.closed=False
        def close(self): self.closed=True
    monkeypatch.setattr(analytic_cuda,'NSAnalyticCudaCache',Cache)
    lanes=analytic_cuda._ParallelGraphLanes('cuda:0',2)
    waiting,cancel=Event(),Event()
    def check():
        if cancel.is_set(): raise RuntimeError('cancelled graph request')
    def observe(started):
        if started: waiting.set()
    def request():
        with bind_queue_wait_observer(observe,check_cancelled=check):
            with lanes.lease(1): raise AssertionError('cancelled query acquired lane')
    with lanes.lease(1) as cache:
        with ThreadPoolExecutor(1) as executor:
            future=executor.submit(request)
            assert waiting.wait(3)
            cancel.set()
            with pytest.raises(RuntimeError,match='cancelled graph request'):
                future.result(timeout=3)
        assert lanes.busy == {0}
    assert not lanes.busy
    with lanes.lease(1) as reused: assert reused is cache
    lanes.close()
    assert cache.closed
    with pytest.raises(RuntimeError,match='closed'):
        with lanes.lease(1): pass
