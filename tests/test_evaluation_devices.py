"""Evaluation borrows idle Agent GPUs without running CUDA or changing solvers."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Event, RLock, current_thread
from types import SimpleNamespace
import time

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionRequest, AgentGPUConflict, AgentInfrastructureError,
    SubprocessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from pde_agent_verifier_solver.sdk.evaluation_devices import (
    EvaluationDevicePool, bind_evaluation_device_pool, current_evaluation_device_pool,
)
from tests.test_agent_parallel_execution import _agent_context, _command_program


class Resource:
    def __init__(self, close=lambda: None):
        self.close = close


def service_and_programs(tmp_path):
    service = SubprocessAgentExecutionService(
        tmp_path/'turns', hard_wall_seconds=5, controller_gpu_devices=('GPU-control',),
        evaluation_device_map={'GPU-control':'cuda:0','GPU-a':'cuda:1','GPU-b':'cuda:2'},
    )
    programs = tuple(_command_program(tmp_path/'unused.py', gpu_devices=(token,))
                     for token in ('GPU-a','GPU-b'))
    return service, programs


def test_scoped_resources_construct_once_close_in_reverse_order_and_reject_reuse():
    pool = EvaluationDevicePool(('cuda:0','cuda:1','cuda:2'))
    created, closed = [], []

    def factory():
        created.append(1)
        time.sleep(.01)
        return Resource(lambda: closed.append('first'))

    with ThreadPoolExecutor(max_workers=8) as workers:
        resources = list(workers.map(lambda _: pool.get_resource('first', factory), range(16)))
    assert len(created) == 1 and all(r is resources[0] for r in resources)
    pool.get_resource('second', lambda: Resource(lambda: closed.append('second')))
    other = EvaluationDevicePool(pool.devices)
    assert all(pool.device_for(str(i)) == other.device_for(str(i)) for i in range(100))
    assert set(pool.device_for(str(i)) for i in range(100)) == set(pool.devices)
    assert current_evaluation_device_pool() is None
    with bind_evaluation_device_pool(pool):
        assert current_evaluation_device_pool() is pool
        with bind_evaluation_device_pool(None):
            assert current_evaluation_device_pool() is None
        assert current_evaluation_device_pool() is pool
    assert current_evaluation_device_pool() is None
    pool.close()
    pool.close()
    assert closed == ['second','first'] and pool.cleanup_confirmed
    with pytest.raises(RuntimeError, match='closed'):
        pool.get_resource('third', Resource)
    with pytest.raises(RuntimeError, match='closed'):
        pool.device_for('case')
    other.close()


def test_pool_closes_all_resources_and_never_hides_failed_cleanup():
    pool = EvaluationDevicePool(('cuda:0',))
    closed = []

    def fail():
        closed.append('bad')
        raise RuntimeError('synchronization failed')

    pool.get_resource('first', lambda: Resource(lambda: closed.append('good')))
    pool.get_resource('second', lambda: Resource(fail))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='synchronization failed'):
            pool.close()
    assert closed == ['bad','good'] and not pool.cleanup_confirmed


def test_agent_reservation_prevents_borrow_and_evaluation_blocks_next_round(tmp_path):
    service, programs = service_and_programs(tmp_path)
    requests = tuple(AgentExecutionRequest(p, _agent_context(str(i)), 2) for i,p in enumerate(programs))
    reservation = service.reserve_batch(requests)
    try:
        with pytest.raises(AgentGPUConflict, match='reservation is active'):
            with service.reserve_evaluation_devices(programs):
                pass
    finally:
        reservation.close()
    closed = []
    with pytest.raises(ValueError, match='evaluation failed'):
        with service.reserve_evaluation_devices(programs) as pool:
            assert pool.devices == ('cuda:0','cuda:1','cuda:2')
            pool.get_resource('plugin', lambda: Resource(lambda: closed.append('done')))
            with pytest.raises(AgentGPUConflict, match='already reserved'):
                service.reserve_batch(requests)
            raise ValueError('evaluation failed')
    assert closed == ['done'] and pool.cleanup_confirmed
    with service.reserve_evaluation_devices(programs) as next_pool:
        assert next_pool is not pool
    next_round = service.reserve_batch(requests)
    next_round.close()


def test_plugin_cleanup_must_finish_before_agent_can_reserve_again(tmp_path):
    service, programs = service_and_programs(tmp_path)
    entered, release = Event(), Event()
    request = AgentExecutionRequest(programs[0], _agent_context('next'), 2)

    def close():
        entered.set()
        assert release.wait(3)

    def evaluate():
        with service.reserve_evaluation_devices(programs) as pool:
            pool.get_resource('device-work', lambda: Resource(close))

    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(evaluate)
        try:
            assert entered.wait(2)
            with pytest.raises(AgentGPUConflict, match='already reserved'):
                service.reserve_batch((request,))
        finally:
            release.set()
        future.result(timeout=3)
    reservation = service.reserve_batch((request,))
    reservation.close()


def test_evaluation_cleanup_failure_quarantines_all_borrowed_devices(tmp_path):
    service, programs = service_and_programs(tmp_path)

    def fail():
        raise RuntimeError('resource cleanup unconfirmed')

    with pytest.raises(AgentInfrastructureError, match='cleanup could not be confirmed'):
        with service.reserve_evaluation_devices(programs) as pool:
            pool.get_resource('bad', lambda: Resource(fail))
    assert set(service.quarantined_gpu_devices) == {'GPU-control','GPU-a','GPU-b'}
    with pytest.raises(AgentGPUConflict, match='quarantined'):
        with service.reserve_evaluation_devices(programs):
            pass
    request = AgentExecutionRequest(programs[0], _agent_context('next'), 2)
    with pytest.raises(AgentGPUConflict, match='quarantined'):
        service.reserve_batch((request,))


def test_default_service_and_invalid_device_maps(tmp_path):
    for mapping in (None, {}):
        service = SubprocessAgentExecutionService(
            tmp_path/'default', hard_wall_seconds=2,
            controller_gpu_devices=('legacy-controller',), evaluation_device_map=mapping,
        )
        with service.reserve_evaluation_devices((object(),)) as pool:
            assert pool is None  # Disabled adapters need no GPU-specific program.
    for mapping in ({'physical':'0'}, {'a':'cuda:0','b':'cuda:0'}, {' a ':'cuda:0'}):
        with pytest.raises(ValueError):
            SubprocessAgentExecutionService(tmp_path/'invalid', hard_wall_seconds=2,
                                            evaluation_device_map=mapping)
    service, programs = service_and_programs(tmp_path)
    extra = _command_program(tmp_path/'unused.py', gpu_devices=('unallocated',))
    with pytest.raises(ValueError, match='omits'):
        with service.reserve_evaluation_devices((*programs,extra)):
            pass


@pytest.mark.parametrize('phase', [EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL])
@pytest.mark.parametrize('fail', [False, True])
def test_controller_scope_releases_and_reports_after_evaluation_failure(tmp_path, phase, fail):
    service, programs = service_and_programs(tmp_path)
    controller = object.__new__(CampaignController)
    controller.agent_execution = service
    controller.agents = {'a': programs[0], 'b': programs[1], 'unused': object()}
    controller.campaign = SimpleNamespace(rounds=(SimpleNamespace(
        agents=(SimpleNamespace(agent_id='a'), SimpleNamespace(agent_id='b')),
    ),))
    controller._active_round_plan = None  # Bootstrap/Terminal have no active Agent.
    events, closed = [], []
    controller.events = SimpleNamespace(emit=lambda event,*args,**kwargs: events.append((event,kwargs)))

    def evaluate(*_args, **_kwargs):
        pool = current_evaluation_device_pool()
        assert pool is not None
        pool.get_resource('plugin', lambda: Resource(lambda: closed.append(True)))
        if fail:
            raise RuntimeError('evaluation failed')
        return ('reports','evaluations')

    controller._official_reports_in_scope = evaluate
    with pytest.raises(RuntimeError, match='evaluation failed') if fail else nullcontext():
        result = controller._official_reports((), {}, phase=phase)
        assert result == ('reports','evaluations')
    assert closed == [True] and not service._reserved_gpu_devices
    assert current_evaluation_device_pool() is None
    assert [row[0] for row in events] == ['evaluation_devices_borrowed','evaluation_devices_returned']
    assert events[-1][1]['payload']['cleanup_confirmed']


def test_hidden_verifier_carries_pool_to_real_oracle_thread_without_ambient_context(tmp_path):
    from pde_agent_verifier_solver.engine import PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.plugins.ns import NSSolverTask
    from tests.test_candidate_execution import _artifact, _case, _evaluation_settings, _query
    from tests.test_canonical_candidate_boundary import _generator_state

    artifact, query = _artifact(tmp_path), _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    pool = EvaluationDevicePool(('cuda:0','cuda:1'))
    seen, closed = [], []

    class Generator:
        def case_session_methods(self, _access):
            return ('marker',)

        def open_case_session(self, _state, case, _access):
            captured = current_evaluation_device_pool()
            assert captured is pool
            seen.append((case.case_id, current_thread().name, captured.device_for(case.case_id)))
            captured.get_resource('plugin', lambda: Resource(lambda: closed.append(True)))
            return SimpleNamespace(marker=lambda value: value+1, close=lambda: None)

    with bind_evaluation_device_pool(pool):
        verifier = HiddenVerifier(evaluation, query, generator=Generator(), generator_state=_generator_state())
    assert current_evaluation_device_pool() is None
    cases = tuple(_case('case-'+str(i), value=i, use_oracle=True) for i in range(2))
    sessions = {case.case_id: verifier._case_sessions.persistent_public_session(case,case) for case in cases}
    execution = SubprocessCandidateExecutionService(
        tmp_path/'execution', task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=2, oracle_max_workers=2,
        case_hard_wall_seconds=15,
    )
    plan = EvaluationPlan(phase=EvaluationPhase.OFFICIAL, query=query, settings=settings)
    try:
        outputs = execution.execute_cases(
            artifact, cases, (plan,plan), evaluation, public_session_methods=('marker',),
            invoke_public_session=lambda case_id,method,args,kwargs: sessions[case_id].invoke(method,args,kwargs),
            close_public_session=lambda case_id: sessions[case_id].close(),
        )
        assert [row.prediction.fields['velocity_x'][0] for row in outputs] == [1,2]
        assert len(seen) == 2 and all('oracle-science' in name for _,name,_ in seen)
        assert not closed  # Per-Case close leaves the scope's shared plugin resource alive.
        assert all(audit.status == 'completed' for audit in execution.audits)
    finally:
        execution._executor.shutdown(wait=True, cancel_futures=True)
        pool.close()
    assert closed == [True]


def test_hidden_in_process_opener_uses_captured_scope_and_development_does_not():
    from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, EvaluationRequest
    from pde_agent_verifier_solver.sdk import CaseRecord, CaseVisibility, open_public_case_session
    from tests.test_canonical_candidate_boundary import (
        QUERY, _BoundaryEvaluation, _Truth, _artifact, _case, _generator_state, _policy,
    )
    pool = EvaluationDevicePool(('cuda:0',))
    seen = []

    class Generator:
        def case_session_methods(self, _access):
            return ('marker',)

        def open_case_session(self, _state, _case, _access):
            seen.append(current_evaluation_device_pool())
            return SimpleNamespace(marker=lambda: 0.0, close=lambda: None)

    with bind_evaluation_device_pool(pool):
        verifier = HiddenVerifier(_BoundaryEvaluation(), QUERY, generator=Generator(), generator_state=_generator_state())
        raw = _case('development', parent_case_id=None)
        developer = AgentEvaluator((CaseRecord(raw, _Truth()),), _BoundaryEvaluation(), _policy(),
                                   generator=Generator(), generator_state=_generator_state())
    hidden = _case('hidden', visibility=CaseVisibility.HIDDEN, parent_case_id=None)
    artifact = _artifact(lambda case: open_public_case_session(case).marker())
    with ThreadPoolExecutor(max_workers=1) as workers:
        report = workers.submit(verifier.evaluate, artifact, (CaseRecord(hidden,_Truth()),)).result(timeout=3)
    assert report.primary_error == 0.0 and seen == [pool]
    with bind_evaluation_device_pool(pool):
        developer.evaluate(artifact, EvaluationRequest((developer.cases[0].case_id,)))
    assert seen == [pool, None]
    pool.close()


def test_partially_initialized_resource_failure_cannot_return_gpu_to_agents(tmp_path):
    service, programs = service_and_programs(tmp_path)
    allocated, closed = [], []

    def partially_initialized():
        allocated.append(object())
        raise RuntimeError('failed after device allocation')

    with pytest.raises(AgentInfrastructureError, match='cleanup could not be confirmed'):
        with service.reserve_evaluation_devices(programs) as pool:
            pool.get_resource('healthy', lambda: Resource(lambda: closed.append(True)))
            with pytest.raises(RuntimeError, match='after device allocation'):
                pool.get_resource('partial', partially_initialized)
    assert len(allocated) == 1 and closed == [True]
    assert not pool.cleanup_confirmed
    assert set(service.quarantined_gpu_devices) == {'GPU-control','GPU-a','GPU-b'}
    request = AgentExecutionRequest(programs[0], _agent_context('next'), 2)
    with pytest.raises(AgentGPUConflict, match='quarantined'):
        service.reserve_batch((request,))


def fake_cuda_for_workspace_cleanup(calls, *, fail_sync=None, fail_clear=False):
    def synchronize(device):
        calls.append(('sync',device))
        if device == fail_sync:
            raise RuntimeError('CUDA synchronize failed')

    @contextmanager
    def device_context(device):
        calls.append(('device',device))
        yield

    def clear():
        calls.append(('clear_workspaces',))
        if fail_clear:
            raise RuntimeError('CUDA workspace cleanup failed')

    return SimpleNamespace(
        cuda=SimpleNamespace(synchronize=synchronize, device=device_context,
                             empty_cache=lambda: calls.append(('empty_cache',))),
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear),
    )


def test_cuda_workspace_finalizer_runs_once_after_all_plugin_resources():
    from pde_agent_verifier_solver.sdk.evaluation_devices import register_cuda_workspace_cleanup
    pool = EvaluationDevicePool(('cuda:0','cuda:1'))
    calls = []
    torch = fake_cuda_for_workspace_cleanup(calls)
    register_cuda_workspace_cleanup(pool, torch)
    pool.get_resource('ns', lambda: Resource(lambda: calls.append(('ns_close',))))
    register_cuda_workspace_cleanup(pool, torch)  # Another plugin shares the finalizer.
    pool.get_resource('vpfp', lambda: Resource(lambda: calls.append(('vpfp_close',))))
    assert calls == []  # Registration does not initialize CUDA or clear live workspaces.
    pool.close()
    pool.close()
    assert calls == [
        ('vpfp_close',), ('ns_close',), ('sync','cuda:0'), ('sync','cuda:1'),
        ('clear_workspaces',), ('device','cuda:0'), ('empty_cache',),
        ('device','cuda:1'), ('empty_cache',),
    ]
    assert pool.cleanup_confirmed


@pytest.mark.parametrize('failure', ['sync','clear','plugin'])
def test_cuda_workspace_finalizer_failure_quarantines_borrowed_cards(tmp_path, failure):
    from pde_agent_verifier_solver.sdk.evaluation_devices import register_cuda_workspace_cleanup
    service, programs = service_and_programs(tmp_path)
    calls = []
    torch = fake_cuda_for_workspace_cleanup(
        calls, fail_sync='cuda:1' if failure == 'sync' else None,
        fail_clear=failure == 'clear',
    )

    def plugin_close():
        calls.append(('plugin_close',))
        if failure == 'plugin':
            raise RuntimeError('plugin work still active')

    with pytest.raises(AgentInfrastructureError, match='cleanup could not be confirmed'):
        with service.reserve_evaluation_devices(programs) as pool:
            register_cuda_workspace_cleanup(pool, torch)
            pool.get_resource('plugin', lambda: Resource(plugin_close))
    assert not pool.cleanup_confirmed
    assert set(service.quarantined_gpu_devices) == {'GPU-control','GPU-a','GPU-b'}
    assert ('empty_cache',) not in calls
    if failure in {'sync','plugin'}:
        assert ('clear_workspaces',) not in calls
    if failure == 'plugin':
        assert calls == [('plugin_close',)]


def test_cuda_workspace_registration_rejects_unsafe_late_first_registration():
    from pde_agent_verifier_solver.sdk.evaluation_devices import register_cuda_workspace_cleanup
    pool = EvaluationDevicePool(('cuda:0',))
    calls = []
    pool.get_resource('already_allocated', lambda: Resource(lambda: calls.append(('plugin_close',))))
    with pytest.raises(RuntimeError, match='must precede'):
        register_cuda_workspace_cleanup(pool, fake_cuda_for_workspace_cleanup(calls))
    with pytest.raises(RuntimeError, match='must precede'):
        pool.close()
    assert calls == [('plugin_close',)] and not pool.cleanup_confirmed


@pytest.mark.parametrize('phase', [EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL])
@pytest.mark.parametrize('fail', [False, True])
def test_backend_provider_is_exclusive_and_restores_context_after_failure(phase, fail):
    controller = object.__new__(CampaignController)
    calls = []
    cases = (object(), object())
    states = {'base': object(), 'guided': object()}
    artifacts = (object(),)
    hidden = {'base': (SimpleNamespace(case=cases[0]),),
              'guided': (SimpleNamespace(case=cases[1]),)}

    @contextmanager
    def backend_scope(**kwargs):
        assert current_evaluation_device_pool() is None
        calls.append(('enter', kwargs))
        try:
            yield
        finally:
            assert current_evaluation_device_pool() is None
            calls.append(('exit',))

    def reject_local_lease(*_args):
        pytest.fail('backend provider must not enter a local GPU lease')

    def evaluate(passed_artifacts, passed_hidden, **kwargs):
        assert current_evaluation_device_pool() is None
        assert passed_artifacts is artifacts and passed_hidden is hidden
        assert kwargs == {'phase': phase, 'seed_namespace': 'stable-plans',
                          'generator_states_by_distribution': states}
        calls.append(('evaluate',))
        if fail:
            raise LookupError('candidate failed')
        return ('reports', 'evaluations')

    controller.candidate_execution = SimpleNamespace(hidden_evaluation_scope=backend_scope)
    controller.agent_execution = SimpleNamespace(
        evaluation_device_map=None, quarantined_gpu_devices=('quarantined',),
        reserve_evaluation_devices=reject_local_lease,
    )
    controller._official_reports_in_scope = evaluate
    ambient = EvaluationDevicePool(('cuda:0',))
    with bind_evaluation_device_pool(ambient):
        with pytest.raises(LookupError, match='candidate failed') if fail else nullcontext():
            result = controller._official_reports(
                artifacts, hidden, phase=phase, seed_namespace='stable-plans',
                generator_states_by_distribution=states,
            )
            assert result == ('reports', 'evaluations')
        assert current_evaluation_device_pool() is ambient
    assert calls == [
        ('enter', {'cases': cases, 'unavailable_gpu_devices': ('quarantined',)}),
        ('evaluate',), ('exit',),
    ]
    ambient.close()


def test_two_configured_evaluation_gpu_providers_fail_before_either_starts(tmp_path):
    service, programs = service_and_programs(tmp_path)
    controller = object.__new__(CampaignController)
    controller.agent_execution = service

    def forbidden(*_args, **_kwargs):
        pytest.fail('a conflicting GPU provider must not be entered')

    controller.candidate_execution = SimpleNamespace(hidden_evaluation_scope=forbidden)
    controller._official_reports_in_scope = forbidden
    # Even if an Agent already owns a card, a configuration conflict is rejected
    # before any scope can acquire or disturb its existing reservation.
    request = AgentExecutionRequest(programs[0], _agent_context('active'), 2)
    reservation = service.reserve_batch((request,))
    try:
        before = dict(service._reserved_gpu_devices)
        with pytest.raises(ValueError, match='cannot both be configured'):
            controller._official_reports((), {})
        assert service._reserved_gpu_devices == before
    finally:
        reservation.close()


@pytest.mark.parametrize('replicated', [False, True])
def test_per_case_lock_provider_and_opener_receive_captured_pool_in_worker(replicated):
    from tests.test_canonical_candidate_boundary import (
        QUERY, _BoundaryEvaluation, _case, _generator_state,
    )
    pool = EvaluationDevicePool(('cuda:0',))
    lane_lock = RLock()
    seen = []

    class Generator:
        def case_session_methods(self, _access):
            return ('marker',)

        def public_session_operation_lock(self, _state, _case):
            seen.append(('lock', current_evaluation_device_pool(), current_thread().name))
            return lane_lock if replicated else None

        def open_case_session(self, _state, _case, _access):
            seen.append(('open', current_evaluation_device_pool(), current_thread().name))
            return SimpleNamespace(marker=lambda: 7, close=lambda: None)

    with bind_evaluation_device_pool(pool):
        verifier = HiddenVerifier(_BoundaryEvaluation(), QUERY,
                                  generator=Generator(), generator_state=_generator_state())
    case = _case('hidden', parent_case_id=None)
    bindings = verifier._case_sessions
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='captured-scope') as worker:
        session = worker.submit(bindings.persistent_public_session, case, case).result(timeout=3)
        try:
            assert session._operation_lock is (lane_lock if replicated else bindings._operation_lock)
            assert worker.submit(session.invoke, 'marker', (), {}).result(timeout=3) == 7
        finally:
            session.close()
    assert [kind for kind, _, _ in seen] == ['lock', 'open']
    assert all(captured is pool and thread.startswith('captured-scope')
               for _, captured, thread in seen)
    assert current_evaluation_device_pool() is None
    pool.close()
