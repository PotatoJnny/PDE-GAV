from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import numpy as np
import pytest
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionError, CandidatePublicSessionClient, _CandidateOracleHost,
)
from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_batch, query_public_cases
from tests.test_candidate_execution import _case


def _host(tmp_path, invoke, trace=None):
    host = _CandidateOracleHost(tmp_path/'arrays', case_ids=('a','b'),
        methods=('marker',), invoke=invoke, close=None, max_workers=2, trace_call=trace)
    host.start()
    return host, [CandidatePublicSessionClient(host.manifest_for_case(c)) for c in ('a','b')]


def test_combined_transport_overlaps_work_preserves_ragged_arrays_and_reuses_host(tmp_path):
    barrier = Barrier(2)
    events = []
    def invoke(cid, method, args, kwargs):
        barrier.wait(timeout=3)
        return args[0] + (1 if cid == 'a' else 10)
    host, clients = _host(tmp_path, invoke, lambda **row: events.append(row))
    points = {'a': np.arange(10.).reshape(5,2), 'b': np.arange(8.).reshape(4,2)}
    try:
        for _ in range(2):
            out = CandidatePublicSessionClient.invoke_many(clients, 'marker', points)
            np.testing.assert_array_equal(out['a'], points['a']+1)
            np.testing.assert_array_equal(out['b'], points['b']+10)
        assert sum(e.get('action') == 'submit_many' for e in events) == 2
        assert len([e for e in events if e.get('case_id')]) == 4
        assert not host.state._requests
    finally:
        host.close()


def test_bad_batch_capability_does_not_start_any_member(tmp_path):
    seen=[]
    host, clients = _host(tmp_path, lambda cid,*args: seen.append(cid))
    entries = [{'token': c.token,'request_id': c.case_id,'params': {
        'case_id': c.case_id,'method':'marker', 'args': c.codec.encode((1,)),
        'kwargs':c.codec.encode({})}} for c in clients]
    entries[1]['params']['case_id'] = 'a'
    try:
        with pytest.raises(PermissionError): host.state.submit_many(entries)
        assert not seen and not host.state._requests
    finally: host.close()


def test_batch_method_failure_is_consumed_and_following_queries_work(tmp_path):
    def invoke(cid,method,args,kw):
        if args[0] < 0: raise ValueError('numerical failure')
        return args[0] + 1
    host,clients = _host(tmp_path,invoke)
    try:
        with pytest.raises(CandidateExecutionError):
            CandidatePublicSessionClient.invoke_many(clients,'marker',{'a':-1,'b':3})
        assert not host.state._requests
        assert CandidatePublicSessionClient.invoke_many(clients,'marker',{'a':1,'b':3}) == {'a':2,'b':4}
    finally:host.close()


def test_sdk_bulk_callback_keeps_scope_and_method_permissions(tmp_path):
    host, clients=_host(tmp_path,lambda cid,m,args,kw:args[0]+1)
    cases=tuple(_case(c.case_id,value=0) for c in clients)
    calls=[]
    def batch(selected,method,points,limit):
        calls.append(tuple(c.case_id for c in selected))
        return CandidatePublicSessionClient.invoke_many(clients,method,points,max_workers=limit)
    try:
        with bind_public_session_batch({c.case_id:c.session_factory for c in clients},cases,
                allowed_methods=('marker',), query_many=batch):
            assert query_public_cases(cases,'marker',{'a':2,'b':7}) == {'a':3,'b':8}
            with pytest.raises(AttributeError):query_public_cases(cases,'private_truth',{'a':2,'b':7})
            with pytest.raises(PermissionError):query_public_cases((_case('other'),),'marker',{'other':0})
        assert len(calls)==1
        with pytest.raises(RuntimeError):query_public_cases(cases,'marker',{'a':2,'b':7})
    finally:host.close()


def test_older_host_manifest_still_uses_parallel_single_calls(tmp_path):
    gate=Barrier(2)
    def invoke(cid,m,args,kw):gate.wait(timeout=3);return args[0]
    host,clients=_host(tmp_path,invoke)
    for c in clients:c.bulk_queries=False
    try:assert CandidatePublicSessionClient.invoke_many(clients,'marker',{'a':1,'b':2}) == {'a':1,'b':2}
    finally:host.close()

def test_two_agents_can_submit_32_cases_each_with_eight_case_workers(tmp_path):
    from tests.test_batch_solver_execution import _batch_artifact, _service
    from concurrent.futures import ThreadPoolExecutor
    artifact = _batch_artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path, workers=4, batch=8, oracle=4, budget=15)
    def evaluate(owner):
        cases = tuple(_case(f'{owner}-{i}', value=i, use_oracle=True) for i in range(32))
        return service.execute_cases(artifact, cases, (plan,)*len(cases), evaluation,
            public_session_methods=('marker',),
            invoke_public_session=lambda cid,m,args,kw: args[0]+1,
            close_public_session=lambda cid: None)
    try:
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(evaluate, owner) for owner in ('a','b')]
            for future in pending:
                assert [o.prediction.fields['velocity_x'][0] for o in future.result(timeout=45)] == list(range(1,33))
        assert len(service.audits) == 64
        assert len({a.solver_batch_id for a in service.audits}) == 8
        assert all(a.solver_batch_size == 8 and a.oracle_call_count == 1 for a in service.audits)
    finally:
        service.close()

def test_development_oracle_limit_does_not_serialise_hidden_evaluation(tmp_path):
    from dataclasses import replace
    from tests.test_batch_solver_execution import _batch_artifact, _query, _evaluation_settings
    from pde_agent_verifier_solver.engine import PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.plugins.ns import NSSolverTask
    from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
    from threading import Lock
    artifact = _batch_artifact(tmp_path)
    settings = _evaluation_settings(_query())
    service = SubprocessCandidateExecutionService(tmp_path/'execution',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),evaluation_settings=settings,
        max_workers=4,solver_batch_size=8,oracle_max_workers=4,
        development_oracle_max_workers=2,case_hard_wall_seconds=15)
    cases = tuple(_case(str(i), value=i, use_oracle=True) for i in range(8))
    try:
        for phase, expected in ((EvaluationPhase.DEVELOPMENT,2),(EvaluationPhase.OFFICIAL,4)):
            active=peak=0
            barrier, lock = Barrier(expected), Lock()
            def invoke(cid,method,args,kw):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                try:
                    barrier.wait(timeout=3)
                    return args[0]+1
                finally:
                    with lock:active-=1
            plan=EvaluationPlan(phase=phase,query=_query(),settings=settings)
            outputs=service.execute_cases(artifact,cases,(plan,)*8,NSSolverTask().create_evaluation(settings),
                public_session_methods=('marker',),invoke_public_session=invoke,close_public_session=lambda cid:None)
            assert [o.prediction.fields['velocity_x'][0] for o in outputs] == list(range(1,9))
            assert peak==expected and active==0
    finally:service.close()


@pytest.mark.parametrize("trusted", [False, True])
def test_bulk_queries_keep_inline_transport_policy_and_timing_trace(tmp_path, monkeypatch, trusted):
    """The merged bulk path must use the same result policy as single queries."""
    from pde_agent_verifier_solver.engine import agent_wire as wire

    events = []
    points = {cid: np.arange(8192 * 5, dtype=np.float64).reshape(8192, 5)
              for cid in ("a", "b")}
    expected = np.arange(8192, dtype=np.float64)
    expected[0] = -0.0
    host = _CandidateOracleHost(tmp_path / "arrays", case_ids=("a", "b"),
        methods=("source",), invoke=lambda *args: expected, close=None,
        max_workers=2, oracle_inline_array_bytes=8 * 1024 * 1024,
        oracle_trusted_results=trusted, trace_call=lambda **row: events.append(row))
    host.start()
    clients = [CandidatePublicSessionClient(host.manifest_for_case(cid)) for cid in points]
    scans = []
    original = np.isfinite

    def counted(value, *args, **kwargs):
        scans.append(np.shape(value))
        return original(value, *args, **kwargs)

    def no_sidecar(*args, **kwargs):
        raise AssertionError("small bulk queries should remain inline")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(np, "isfinite", counted)
            patch.setattr(wire.AgentWireCodec, "_publish_array", no_sidecar)
            actual = CandidatePublicSessionClient.invoke_many(clients, "source", points)
        assert all(value.tobytes() == expected.tobytes() for value in actual.values())
        assert scans.count((8192, 5)) == 4
        assert scans.count((8192,)) == (0 if trusted else 4)
        assert sum(row.get("action") == "submit_many" for row in events) == 1
        assert {row.get("case_id") for row in events if row.get("case_id")} == {"a", "b"}
        assert not host.state._requests
    finally:
        host.close()
