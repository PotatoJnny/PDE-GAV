#!/usr/bin/env python3
"""Four-card routing and throughput qualification, independent of paper accuracy runs.

Fixed NS workload through actual subprocesses/RPC; no artificial compute sleeps.
One delayed second client probes cross-Agent scheduling. GPU3 is checked separately
with Official evaluation; GPU0/1 host Development, GPU2 hosts continuous Oracles.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from threading import Event, Thread
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'legacy/ns/src'),str(ROOT/'scripts')]
from benchmark_parallel_evaluation import SOLVER_SOURCE, _fields


class GpuSampler:
    def __init__(self,path):
        self.path=Path(path); self.stop=Event(); self.thread=Thread(target=self._run,daemon=True)
    def __enter__(self): self.thread.start(); return self
    def _run(self):
        commands={
          'gpus':['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],
          'processes':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']}
        with self.path.open('w',buffering=65536) as stream:
            while not self.stop.is_set():
                row={'time_unix':time.time()}
                for key,cmd in commands.items():
                    try:
                        result=subprocess.run(cmd,capture_output=True,text=True,timeout=4)
                        row[key]=result.stdout.strip().splitlines(); row[key+'_exit']=result.returncode
                    except Exception as error: row[key+'_error']=str(error)
                stream.write(json.dumps(row)+'\n');stream.flush()
                self.stop.wait(1.)
    def __exit__(self,*args): self.stop.set();self.thread.join(timeout=10)


def run(output, *, tokens, grid=64, steps=64, seconds=600, oracle_cache=False, trials=None, oracle_workers=8, time_batch=1, combine_transport=True, case_counts=(16,4), include_official=True, oracle_only=False):
    import torch
    from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.plugins.ns import NSSolverTask
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSPublicSession
    from pde_agent_verifier_solver.plugins.ns.analytic_cuda import NSAnalyticCudaCache
    from pde_agent_verifier_solver.sdk import AnswerQuery,Case,CaseSource,CaseVisibility,EvaluationPlan,EvaluationPhase,TaskKind,write_candidate_bundle_manifest
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if oracle_only:
        if len(tokens)!=1 or tokens[0] not in visible: raise ValueError('Oracle-only probe needs one allocated GPU')
        tokens=tokens*4
    elif len(set(tokens))!=4 or not set(tokens)<=set(visible): raise ValueError('need four distinct allocated GPUs')
    device='cuda:'+str(visible.index(tokens[2]))
    roles={'agent_a':tokens[0],'agent_b':tokens[1],'oracle':tokens[2],'official':tokens[3]}
    if oracle_only: roles.update(agent_a='cpu',agent_b='cpu',official='cpu')
    identities={token:str(torch.cuda.get_device_properties(visible.index(token)).uuid) for token in tokens}
    source=output/'source';source.mkdir();(source/'submission.py').write_text(SOLVER_SOURCE)
    write_candidate_bundle_manifest(source,files=('submission.py',),overwrite=True)
    artifact=PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(source,
        storage_root=output/'sealed',artifact_id='fixed-performance-probe',name='NS throughput probe',
        summary='Fixed coarse numerical solver for resource qualification',usage='Not a paper accuracy baseline')
    fields=_fields(device,sum(case_counts),.02)
    cases=tuple(Case(f'probe-{i}',CaseSource.GENERATED,CaseVisibility.DEVELOPMENT,
        {'grid':grid,'steps':steps,'end':.02,'viscosity':.01,'cuda':not oracle_only,'query_time_batch':time_batch,'combine_transport':combine_transport}) for i in range(sum(case_counts)))
    points=tuple({'t':.02,'x':(x+.5)/grid,'y':(y+.5)/grid} for x in range(2,grid-1,max(1,grid//4)) for y in range(2,grid-1,max(1,grid//4)))
    query=AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=points)
    settings={'default_query':query,'primary_metric':'velocity_relative_l2','include_pressure':False}
    plan=EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT,query=query,settings=settings)
    task=NSSolverTask();evaluation=task.create_evaluation(settings)
    rows=[]; reference=None;deadline=time.monotonic()+seconds
    variants=[(4,2,2),(1,2,2),(8,2,2),(16,2,2),(4,1,2),(4,2,1),(4,2,4),(4,2,8),(4,4,2)]
    random.Random(20261003).shuffle(variants)
    # Same workload on the previous shared placement is the routing control.
    pending=[]
    for repeat in range(2):
        entries=[(repeat,b,s,o,"agent_gpu") for b,s,o in variants[repeat:]+variants[:repeat]]
        entries.insert(repeat*len(entries),(repeat,4,2,2,"shared"))
        pending.extend(entries)
    if trials is not None:
        pending=[tuple(row) for row in trials]
        if not pending or any(len(row)!=5 or row[4] not in ('agent_gpu','shared')
                or any(type(n) is not int for n in row[:4]) or row[0]<0
                or min(row[1:4])<1 for row in pending):
            raise ValueError('trials must contain repeat, batch, slots, Oracle concurrency, placement')
    def measure(repeat,batch,slots,oracle,phase='development',placement='agent_gpu'):
        nonlocal reference
        label=f'{phase}-{placement}-b{batch}-s{slots}-o{oracle}-r{repeat}'
        cache=NSAnalyticCudaCache(device) if oracle_cache else None
        for f in fields:f._analytic_cuda_cache=cache
        sessions={c.case_id:_NSPublicSession(c,f.evaluate_forcing,f.evaluate_velocity,0.,f.domain_bounds,f.time_interval,grid*grid*max(time_batch,8)) for c,f in zip(cases,fields)}
        service=SubprocessCandidateExecutionService(output/label,
            task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),evaluation_settings=settings,
            max_workers=(4 if oracle_only else max(8,2*slots)),case_hard_wall_seconds=120,solver_batch_size=batch,
            oracle_max_workers=oracle_workers,oracle_concurrency_per_device=oracle,
            gpu_devices=(() if oracle_only else (tokens[3],)),gpu_slots_per_device=slots,
            development_gpu_devices=({'a':(tokens[0],),'b':(tokens[1],)} if placement=='agent_gpu' and not oracle_only else {}),trace_calls=True)
        client=[];started=time.time(); started_mono=time.monotonic()
        def evaluate(owner,selected,delay=0):
            if delay: time.sleep(delay) # intentional stagger only; not inside a measured solver
            begin=time.time(); tick=time.monotonic()
            bound=service.for_development(owner) if phase=='development' else service
            chosen_plan=plan if phase=='development' else replace(plan,phase=EvaluationPhase.OFFICIAL)
            result=bound.execute_cases(artifact,selected,(chosen_plan,)*len(selected),evaluation,
                public_session_methods=('query_forcing','query_initial_velocity'),
                invoke_public_session=lambda cid,method,args,kw:getattr(sessions[cid],method)(*args,**kw),
                close_public_session=lambda cid:sessions[cid].close())
            client.append({'agent':owner,'submitted_at_unix':begin,'wall_seconds':time.monotonic()-tick,'cases':len(selected),'case_ids':[c.case_id for c in selected]})
            return result
        try:
            if phase=='development':
                with ThreadPoolExecutor(2) as pool:
                    a=pool.submit(evaluate,'a',cases[:case_counts[0]]);b=pool.submit(evaluate,'b',cases[case_counts[0]:],.2)
                    outputs=(*a.result(),*b.result())
            else: outputs=evaluate(None,cases[:4])
            elapsed=time.monotonic()-started_mono
            predictions=[np.array((o.prediction.fields['velocity_x'],o.prediction.fields['velocity_y'])) for o in outputs]
            if reference is None:
                reference=predictions
                np.save(output/"predictions.npy",np.stack(predictions),allow_pickle=False)
            for got,want in zip(predictions,reference):np.testing.assert_allclose(got,want,rtol=2e-9,atol=1e-11)
            expected={owner:identities[token] for owner,token in [('a',tokens[0]),('b',tokens[1]),(None,tokens[3])]}
            if not oracle_only:
                assert all(a.gpu_metadata['uuid']==expected[a.requester_agent_id] for a in service.audits)
            else:
                assert all(a.gpu_device is None for a in service.audits)
            service.close()
            events=[json.loads(line) for line in (service.root/'execution_trace.jsonl').read_text().splitlines()]
            calls=[r for r in events if r['event']=='oracle_call']
            transports=[r for r in events if r['event']=='oracle_transport']
            starts=[r for r in events if r['event']=='solver_started']
            for c in client:
                mine=[r for r in starts if set(r['case_ids']) & set(c['case_ids'])]
                c['first_worker_start_delay_seconds']=min(r['time_unix'] for r in mine)-c['submitted_at_unix']
            audits=[];seen=set()
            for a in service.audits:
                key=a.solver_batch_id or (a.execution_id,a.case_id)
                if key in seen:continue
                seen.add(key);audits.append(a)
            row={'label':label,'phase':phase,'placement':placement,'repeat':repeat,'batch':batch,'slots_per_agent_gpu':slots,
                'oracle_concurrency':oracle,'oracle_global_workers':oracle_workers,
                'time_batch':time_batch,'combined_transport':combine_transport,'started_at_unix':started,'finished_at_unix':time.time(),
                'wall_seconds':elapsed,'cases_per_second':len(outputs)/elapsed,'clients':sorted(client,key=lambda r:r['agent'] or ''),
                'oracle_calls':len(calls),'http_exchanges':len(transports),
                'http_handler_sum_seconds':sum(r['handler_seconds'] for r in transports),
                'http_json_encode_sum_seconds':sum(r['json_encode_seconds'] for r in transports),
                'http_send_sum_seconds':sum(r['send_seconds'] for r in transports),
                'numerical_profile_sum':{key:sum((r.get('numerical_profile') or {}).get(key,0.) for r in calls)
                    for key in ('math_enqueue_seconds','math_device_seconds','download_and_sync_seconds')},'oracle_wait_sum_seconds':sum(r['queue_wait_seconds'] for r in calls),
                'oracle_call_wait_p50':float(np.median([r['queue_wait_seconds'] for r in calls])),
                'oracle_call_wait_p95':float(np.quantile([r['queue_wait_seconds'] for r in calls],.95)),
                'oracle_call_wall_p95':float(np.quantile([r['total_wall_seconds'] for r in calls],.95)),
                'solver_queue_wait_p95':float(np.quantile([a.solver_queue_wait_seconds for a in audits],.95)),
                'peak_torch_memory_bytes_per_worker':max(a.gpu_metadata.get('peak_memory_allocated_bytes',0) for a in audits),
                'predictions_match':True,'max_prediction_difference':max(float(np.max(np.abs(x-y))) for x,y in zip(predictions,reference)),
                'cache':cache.diagnostics if cache else None,'device_uuid_verified':not oracle_only,'oracle_only':oracle_only}
            rows.append(row)
            (output/'progress.json').write_text(json.dumps({'rows':rows,'roles':roles,'identities':identities},indent=2))
            print(json.dumps({k:row[k] for k in ['label','wall_seconds','oracle_call_wait_p95','solver_queue_wait_p95','clients']}),flush=True)
        finally:
            service.close()
            for s in sessions.values():s.close()
            for f in fields:f._analytic_cuda_cache=None
            if cache is not None:cache.close()
    for repeat,batch,slots,oracle,placement in pending:
        if time.monotonic()>deadline-35:break
        measure(repeat,batch,slots,oracle,placement=placement)
    grouped={}
    for row in rows:grouped.setdefault((row['placement'],row['batch'],row['slots_per_agent_gpu'],row['oracle_concurrency']),[]).append(row)
    summary=[{'placement':p,'batch':b,'slots':s,'oracle':o,'repeats':len(v),'median_seconds':statistics.median(r['wall_seconds'] for r in v),
              'b_start_delay_median':statistics.median(next(c['first_worker_start_delay_seconds'] for c in r['clients'] if c['agent']=='b') for r in v)} for (p,b,s,o),v in grouped.items()]
    best=min((r for r in summary if r['placement']=='agent_gpu'),key=lambda r:r['median_seconds'])
    if include_official and time.monotonic()<deadline-30:
        measure(0,best['batch'],best['slots'],best['oracle'],phase='official')
    result={'status':('completed' if sum(r['phase']=='development' for r in rows)==len(pending) else 'partial'),
        'scope':('single-GPU Oracle with CPU solver producers' if oracle_only else 'four-card solver and Oracle throughput'),
        'grid':grid,'steps':steps,'case_counts':case_counts,'oracle_path':'cached' if oracle_cache else 'eager_development','roles':roles,'identities':identities,'rows':rows,
        'summary':sorted(summary,key=lambda r:r['median_seconds']),'best_measured':best,
        'planned_development_variants':len(pending),'completed_development_variants':sum(r['phase']=='development' for r in rows)}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--tokens',default=None);parser.add_argument('--grid',type=int,default=64)
    parser.add_argument('--steps',type=int,default=64);parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--oracle-cache',action='store_true')
    parser.add_argument('--oracle-workers',type=int,default=8)
    parser.add_argument('--time-batch',type=int,default=1)
    parser.add_argument('--separate-requests',action='store_true')
    args=parser.parse_args();root=Path(args.output);root.parent.mkdir(parents=True,exist_ok=True)
    tokens=tuple((args.tokens or os.environ['CUDA_VISIBLE_DEVICES']).split(','))
    with GpuSampler(root.parent/'gpu_utilization.jsonl'):
        run(root,tokens=tokens,grid=args.grid,steps=args.steps,seconds=args.seconds,oracle_cache=args.oracle_cache,
            oracle_workers=args.oracle_workers,time_batch=args.time_batch,combine_transport=not args.separate_requests)
