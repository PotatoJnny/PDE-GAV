"""Trusted requester routing, queue isolation, phase separation and trace completeness."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from dataclasses import replace
import json
import pytest

from pde_agent_verifier_solver.engine import SubprocessCandidateExecutionService, PythonTaskWorkerSpec
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from tests.test_batch_solver_execution import _batch_artifact, _case, _query, _evaluation_settings


def routed_service(tmp_path, **kwargs):
    settings = _evaluation_settings(_query())
    service = SubprocessCandidateExecutionService(tmp_path/'execution',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask), evaluation_settings=settings,
        max_workers=4, case_hard_wall_seconds=20, solver_batch_size=4,
        oracle_max_workers=8, gpu_devices=('3',), gpu_slots_per_device=2,
        development_gpu_devices={'a':('0',), 'b':('1',)}, **kwargs)
    return service, NSSolverTask().create_evaluation(settings), EvaluationPlan(
        phase=EvaluationPhase.DEVELOPMENT, query=_query(), settings=settings)


def test_b_starts_while_a_owns_all_its_slots_even_for_same_artifact(tmp_path, monkeypatch):
    artifact = _batch_artifact(tmp_path)
    service, evaluation, plan = routed_service(tmp_path)
    a_started, a_second_started, release = Event(), Event(), Event()
    routes=[]
    def execute(*args, route_devices, requester_agent_id, submitted_at):
        routes.append((requester_agent_id, route_devices))
        if requester_agent_id == 'a':
            (a_second_started if a_started.is_set() else a_started).set()
            assert release.wait(5)
        members=args[-1]
        return tuple(object() for _ in members)
    monkeypatch.setattr(service, '_execute_one', execute)
    def run(owner, count):
        cases=tuple(_case(f'{owner}-{i}', value=i) for i in range(count))
        return service.for_development(owner).execute_cases(artifact,cases,(plan,)*count,evaluation)
    try:
        with ThreadPoolExecutor(2) as pool:
            a=pool.submit(run,'a',12)
            assert a_started.wait(2) and a_second_started.wait(2)
            b=pool.submit(run,'b',4)
            try:
                assert len(b.result(timeout=2)) == 4
                assert not a.done()
            finally: release.set()
            assert len(a.result(timeout=3)) == 12
        assert all(device == (('0',) if owner=='a' else ('1',)) for owner,device in routes)
        assert service.gpu_devices == ('3',)
        assert service.execution_settings_snapshot()['development_gpu_devices'] == {'a':['0'],'b':['1']}
    finally: release.set(); service.close()


def test_unbound_unknown_and_hidden_agent_routes_are_rejected(tmp_path):
    artifact=_batch_artifact(tmp_path)
    service,evaluation,plan=routed_service(tmp_path)
    case=_case('x',value=1)
    try:
        with pytest.raises(ValueError,match='Controller-bound'):
            service.execute_cases(artifact,(case,),(plan,),evaluation)
        with pytest.raises(ValueError,match='No development GPU'):
            service.for_development('unknown')
        with pytest.raises(ValueError,match='only for its Development'):
            service.for_development('a').execute_cases(artifact,(case,),
                (replace(plan,phase=EvaluationPhase.OFFICIAL),),evaluation)
    finally: service.close()


def test_official_uses_its_dedicated_card_not_the_artifact_author(tmp_path,monkeypatch):
    artifact=_batch_artifact(tmp_path)
    service,evaluation,plan=routed_service(tmp_path)
    seen=[]
    def execute(*args,route_devices,requester_agent_id,submitted_at):
        seen.append((route_devices,requester_agent_id))
        return (object(),)
    monkeypatch.setattr(service,'_execute_one',execute)
    try:
        service.execute_cases(artifact,(_case('official',value=1),),
            (replace(plan,phase=EvaluationPhase.OFFICIAL),),evaluation)
        assert seen == [(('3',),None)]
    finally:service.close()


def test_cpu_trace_records_each_rpc_and_batch_without_private_data(tmp_path):
    artifact=_batch_artifact(tmp_path)
    settings=_evaluation_settings(_query())
    service=SubprocessCandidateExecutionService(tmp_path/'trace',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),evaluation_settings=settings,
        max_workers=2,case_hard_wall_seconds=20,solver_batch_size=4,oracle_max_workers=4,
        trace_calls=True)
    plan=EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT,query=_query(),settings=settings)
    cases=tuple(_case(str(i),value=i,use_oracle=True) for i in range(4))
    try:
        outputs=service.execute_cases(artifact,cases,(plan,)*4,NSSolverTask().create_evaluation(settings),
            public_session_methods=('marker',),invoke_public_session=lambda cid,method,args,kw:args[0]+1,
            close_public_session=lambda cid:None)
        assert [o.prediction.fields['velocity_x'][0] for o in outputs] == [1,2,3,4]
    finally: service.close()
    rows=[json.loads(s) for s in (service.root/'execution_trace.jsonl').read_text().splitlines()]
    assert sum(r['event']=='oracle_call' for r in rows)==4
    assert [r['event'] for r in rows if r['event'].startswith('solver_')] == [
        'solver_submitted','solver_started','solver_finished']
    assert all(r['queue_wait_seconds']>=0 for r in rows if r['event']=='oracle_call')
    assert all(not ({'token','coordinates','value','prediction','truth'} & r.keys()) for r in rows)
    assert all(a.solver_queue_wait_seconds>=0 for a in service.audits)


def test_real_four_gpu_route_cuda_identity_and_oracle_service(tmp_path):
    import os
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.device_count()<4:
        pytest.skip('requires a scheduled four-GPU allocation')
    from pde_agent_verifier_solver.engine import PythonCandidateCodec
    from pde_agent_verifier_solver.sdk import TaskKind, write_candidate_bundle_manifest
    tokens=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    _batch_artifact(tmp_path)
    source=tmp_path/'source'
    with (source/'submission.py').open('a') as stream:
        stream.write("""
_original_batch = prepare_solutions
def prepare_solutions(cases, solve_budget=None):
    import torch
    assert torch.cuda.is_available()
    z=torch.arange(256,device='cuda:0',dtype=torch.float64)
    assert float((z*z).sum())>0
    return _original_batch(cases,solve_budget)
""")
    write_candidate_bundle_manifest(source,files=('submission.py',),overwrite=True)
    artifact=PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(source,
        storage_root=tmp_path/'gpu-sealed',artifact_id='same-author',name='route CUDA',summary='test',usage='test')
    settings=_evaluation_settings(_query())
    service=SubprocessCandidateExecutionService(tmp_path/'four-gpu',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),evaluation_settings=settings,
        max_workers=4,case_hard_wall_seconds=45,solver_batch_size=4,oracle_max_workers=4,
        gpu_devices=(tokens[3],),gpu_slots_per_device=2,
        development_gpu_devices={'a':(tokens[0],),'b':(tokens[1],)},trace_calls=True)
    plan=EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT,query=_query(),settings=settings)
    def invoke(cid,method,args,kwargs):
        # The Controller numerical input work is on a third card.
        with torch.cuda.device(2):
            x=torch.tensor(float(args[0]),device='cuda:2');return float((x+1).cpu())
    def evaluate(owner,phase):
        cases=tuple(_case(f'{owner}-{phase.value}-{i}',value=i,use_oracle=True) for i in range(4))
        bound=service.for_development(owner) if phase==EvaluationPhase.DEVELOPMENT else service
        return bound.execute_cases(artifact,cases,(replace(plan,phase=phase),)*4,NSSolverTask().create_evaluation(settings),
            public_session_methods=('marker',),invoke_public_session=invoke,close_public_session=lambda cid:None)
    try:
        with ThreadPoolExecutor(2) as pool:
            a=pool.submit(evaluate,'a',EvaluationPhase.DEVELOPMENT)
            b=pool.submit(evaluate,'b',EvaluationPhase.DEVELOPMENT)
            assert len(a.result(timeout=60))==len(b.result(timeout=60))==4
        assert len(evaluate(None,EvaluationPhase.OFFICIAL))==4
        for audit in service.audits:
            index={'a':0,'b':1,None:3}[audit.requester_agent_id]
            assert audit.gpu_metadata['uuid']==str(torch.cuda.get_device_properties(index).uuid)
            assert audit.gpu_device==tokens[index]
    finally:service.close()


def test_queued_oracle_cancellation_is_traced_without_invoking_science(tmp_path):
    from pde_agent_verifier_solver.engine.candidate_execution import _CandidateOracleHost
    entered, release = Event(), Event()
    events = []
    def invoke(case_id, method, args, kwargs):
        assert case_id == "running"
        entered.set()
        assert release.wait(3)
        return 1
    host = _CandidateOracleHost(tmp_path/"cancel-arrays", case_ids=("running","queued"),
        methods=("marker",), invoke=invoke, close=lambda cid:None, max_workers=1,
        trace_call=lambda **row: events.append(row))
    host.start()
    try:
        for cid in ("running","queued"):
            manifest=host.manifest_for_case(cid)
            host.state.submit(manifest["token"],cid,{"case_id":cid,"method":"marker",
                "args":host.codec.encode(()),"kwargs":host.codec.encode({})})
            if cid == "running": assert entered.wait(2)
        host.state.revoke_case("queued")
        release.set()
    finally:
        release.set();host.close()
    by_case={row["case_id"]:row for row in events}
    assert len(events)==2
    assert by_case["running"]["status"]=="completed"
    assert by_case["queued"]["status"]=="cancelled_before_service"
    assert by_case["queued"]["service_wall_seconds"]==0


def test_development_route_must_belong_to_allocation(tmp_path,monkeypatch):
    monkeypatch.setenv("SLURM_JOB_GPUS","3")
    with pytest.raises(ValueError,match="Slurm-allocated"):
        routed_service(tmp_path)
