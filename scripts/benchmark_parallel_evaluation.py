#!/usr/bin/env python3
"""Measured NS workload through the real isolated solver and public Oracle service.

This is a performance fixture, not a paper accuracy baseline. The same coarse
Fourier-projection integrator, fields, mesh and steps are used in every mode.
There are no artificial sleeps in the timed workload.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import sys
from threading import Lock
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'legacy/ns/src')]

SOLVER_SOURCE = '''
import numpy as np
import torch
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, query_public_cases

torch.set_num_threads(1)

class Handle:
    def __init__(self, history, dt):
        self.history, self.dt = history, dt
    def evaluate(self, query):
        n = self.history.shape[-1]
        rows = []
        for point in query.points:
            ft = np.clip(float(point['t'])/self.dt, 0., len(self.history)-1)
            lo = min(int(ft), len(self.history)-2)
            a = ft-lo
            ix = (float(point['x'])*n-.5) % n
            iy = (float(point['y'])*n-.5) % n
            x, y = int(ix), int(iy)
            bx, by = ix-x, iy-y
            field = (1-a)*self.history[lo] + a*self.history[lo+1]
            rows.append((1-bx)*(1-by)*field[:,x,y]
                        + bx*(1-by)*field[:,(x+1)%n,y]
                        + (1-bx)*by*field[:,x,(y+1)%n]
                        + bx*by*field[:,(x+1)%n,(y+1)%n])
        values = np.asarray(rows)
        return AnswerView(kind=QueryKind.POINTS,
            fields={'velocity_x':tuple(values[:,0]), 'velocity_y':tuple(values[:,1])},
            points=query.points, shape=(len(rows),))

def _prepare(cases):
    data = cases[0].public_data
    n, steps, end = data['grid'], data['steps'], data['end']
    device = torch.device('cuda:0' if data['cuda'] else 'cpu')
    axis = (np.arange(n)+.5)/n
    x, y = np.meshgrid(axis, axis, indexing='ij')
    xy = np.column_stack((x.ravel(),y.ravel()))
    combine = data.get('combine_transport', True)
    values = query_public_cases(cases, 'query_initial_velocity', {c.case_id:xy for c in cases}, combine_transport=combine)
    def tensor(arrays):
        return torch.as_tensor(np.stack(arrays).reshape(len(cases),n,n,2),
            dtype=torch.float64, device=device).permute(0,3,1,2).contiguous()
    u = tensor([values[c.case_id] for c in cases])
    kx = (2*np.pi*torch.fft.fftfreq(n,d=1/n,device=device,dtype=torch.float64))[:,None]
    ky = (2*np.pi*torch.fft.rfftfreq(n,d=1/n,device=device,dtype=torch.float64))[None,:]
    k2 = kx*kx+ky*ky
    safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
    dt, nu = end/steps, data['viscosity']
    history = [u.detach().cpu().numpy().copy()]
    time_batch = data.get('query_time_batch', 1)
    force_block = None
    for step in range(steps):
        if step % time_batch == 0:
            times = np.arange(step, min(steps, step + time_batch)) * dt
            coords = np.column_stack((np.repeat(times,len(xy)),np.tile(xy,(len(times),1))))
            force = query_public_cases(cases,'query_forcing',{c.case_id:coords for c in cases},
                                       combine_transport=combine)
            force_block = [tensor([force[c.case_id][i*len(xy):(i+1)*len(xy)] for c in cases])
                           for i in range(len(times))]
        f = force_block[step % time_batch]
        hat = torch.fft.rfft2(u)
        dx = torch.fft.irfft2(hat*(1j*kx),s=(n,n))
        dy = torch.fft.irfft2(hat*(1j*ky),s=(n,n))
        lap = torch.fft.irfft2(-hat*k2,s=(n,n))
        rhs = -u[:,0:1]*dx-u[:,1:2]*dy+nu*lap+f
        hat = torch.fft.rfft2(u+dt*rhs)
        pressure = (kx*hat[:,0]+ky*hat[:,1])/safe
        hat[:,0] -= kx*pressure
        hat[:,1] -= ky*pressure
        u = torch.fft.irfft2(hat,s=(n,n))
        history.append(u.detach().cpu().numpy().copy())
    values = np.stack(history)
    return {case.case_id:Handle(values[:,i],dt) for i,case in enumerate(cases)}

def prepare_solution(case, solve_budget=None):
    return _prepare((case,))[case.case_id]

def prepare_solutions(cases, solve_budget=None):
    return _prepare(cases)
'''


def _fields(device, count, end):
    import torch
    from agent_verifier.neural_generator import structured_ns_answer_first as mms
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
    config=mms.StructuredNSProjectionConfig(spatial_mode_x=3,spatial_mode_y=3,time_order=3)
    geometry=ContinuousGeometrySpec(domain_bounds=(0.,1.,0.,1.),
        obstacle_polygons=(((.3,.3),(.6,.3),(.45,.6)),))
    random=torch.Generator().manual_seed(20261003)
    result=[]
    for _ in range(count):
        coeff=mms.StructuredNSCaseCoefficients(
            psi=torch.randn((4,16),generator=random,dtype=torch.float64)*1e-5,
            pressure=torch.randn((4,16),generator=random,dtype=torch.float64)*1e-5)
        field=mms.ContinuousStructuredNSMMS(coeff,config=config,geometry_spec=geometry,
            domain_bounds=geometry.domain_bounds,viscosity=.01,t0=0.,t1=end,device=device)
        result.append(StructuredNSMMSAnswer(field))
    return result


class OracleActivity:
    def __init__(self):
        self.lock=Lock()
        self.active=self.peak=0
        self.calls=0
        self.service_seconds=0.
    @contextmanager
    def observe(self):
        started=time.perf_counter()
        with self.lock:
            self.active+=1
            self.peak=max(self.peak,self.active)
            self.calls+=1
        try: yield
        finally:
            with self.lock:
                self.active-=1
                self.service_seconds+=time.perf_counter()-started


def run(device, work_root, *, repeats=3, grid=32, steps=12, cases_per_agent=4, output=None, oracle_cache=False, mode_names=None):
    import torch
    from agent_verifier.neural_generator import structured_ns_analytic as analytic
    from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.plugins.ns import NSSolverTask
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSPublicSession
    from pde_agent_verifier_solver.sdk import (AnswerQuery, Case, CaseSource, CaseVisibility,
        EvaluationPlan, EvaluationPhase, TaskKind, write_candidate_bundle_manifest)
    torch.set_num_threads(1)
    work_root=Path(work_root)
    work_root.mkdir(parents=True,exist_ok=True)
    source=work_root/'source'
    source.mkdir(exist_ok=True)
    (source/'submission.py').write_text(SOLVER_SOURCE)
    write_candidate_bundle_manifest(source,files=('submission.py',),overwrite=True)
    artifact=PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(source,
        storage_root=work_root/'sealed',artifact_id='batch-throughput',name='Throughput probe',
        summary='same coarse NS spectral integrator in all modes',usage='performance fixture only')
    count=2*cases_per_agent
    fields=_fields(device,count,.02)
    cases=tuple(Case(f'agent-{i//cases_per_agent}-case-{i}',CaseSource.GENERATED,CaseVisibility.DEVELOPMENT,
        {'grid':grid,'steps':steps,'end':.02,'viscosity':.01,'cuda':device.startswith('cuda')}) for i in range(count))
    points=tuple({'t':.02,'x':(x+.5)/grid,'y':(y+.5)/grid}
                 for x in range(2,grid-1,max(1,grid//4)) for y in range(2,grid-1,max(1,grid//4)))
    query=AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=points)
    settings={'default_query':query,'primary_metric':'velocity_relative_l2','include_pressure':False}
    plan=EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT,query=query,settings=settings)
    task=NSSolverTask()
    truth=[f.evaluate_velocity(np.array([[p['t'],p['x'],p['y']] for p in points])) for f in fields]
    token=None
    if device.startswith('cuda'):
        token=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[int(device.split(':')[1])]
    # Same total CPU allocation and Cases. Only allowed concurrency/batch changes.
    modes=[('serial',1,1,1),('parallel_cases',1,2,4),
           ('batch_serial_oracle',4,2,1),('batch_oracle_2',4,2,2),('batch_parallel_oracle',4,2,4)]
    if mode_names:
        chosen = set(mode_names.split(','))
        if not chosen <= {row[0] for row in modes}:
            raise ValueError('unknown benchmark mode')
        modes = [row for row in modes if row[0] in chosen]
    rows=[]
    baseline=None
    original_forcing=analytic.analytic_forcing
    original_velocity=analytic.analytic_velocity
    for repeat in range(repeats):
        # Rotate order to reduce cache / warm-start bias; preserve every sample.
        order=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
        for label,batch,workers,oracle in order:
            activity=OracleActivity()
            def tracked(function):
                def call(*args,**kwargs):
                    with activity.observe(): return function(*args,**kwargs)
                return call
            analytic.analytic_forcing=tracked(original_forcing)
            analytic.analytic_velocity=tracked(original_velocity)
            cache = None
            if oracle_cache:
                from pde_agent_verifier_solver.plugins.ns.analytic_cuda import NSAnalyticCudaCache
                cache = NSAnalyticCudaCache(device)
            for field in fields:
                field._analytic_cuda_cache = cache
            sessions={c.case_id:_NSPublicSession(c,f.evaluate_forcing,f.evaluate_velocity,0.,f.domain_bounds,f.time_interval,grid*grid)
                      for c,f in zip(cases,fields)}
            service=SubprocessCandidateExecutionService(work_root/f'{label}-{repeat}',
                task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),evaluation_settings=settings,
                max_workers=workers,solver_batch_size=batch,oracle_max_workers=min(oracle,workers*batch),
                oracle_concurrency_per_device=oracle,case_hard_wall_seconds=120,
                gpu_devices=(token,) if token else (),gpu_slots_per_device=workers,
                gpu_memory_fraction=.3 if token else None)
            spans=[]
            def evaluate(owner):
                start=time.perf_counter()
                selected=cases[owner*cases_per_agent:(owner+1)*cases_per_agent]
                result=service.execute_cases(artifact,selected,(plan,)*len(selected),task.create_evaluation(settings),
                    public_session_methods=('query_forcing','query_initial_velocity'),
                    invoke_public_session=lambda cid,method,args,kw:getattr(sessions[cid],method)(*args,**kw),
                    close_public_session=lambda cid:sessions[cid].close())
                spans.append({'agent':owner,'seconds':time.perf_counter()-start})
                return result
            started=time.perf_counter()
            try:
                with ThreadPoolExecutor(2) as executor:
                    futures=[executor.submit(evaluate,i) for i in (0,1)]
                    outputs=[value for future in futures for value in future.result()]
                elapsed=time.perf_counter()-started
                predictions=[np.column_stack((o.prediction.fields['velocity_x'],o.prediction.fields['velocity_y'])) for o in outputs]
                if baseline is None: baseline=predictions
                for actual,expected in zip(predictions,baseline):
                    np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=1e-11)
                errors=[float(np.mean(np.sqrt(np.sum((p-t)**2,axis=0)/np.maximum(np.sum(t*t,axis=0),1e-30))))
                        for p,t in zip(predictions,truth)]
                audits=service.audits
                row={'mode':label,'repeat':repeat,'seconds':elapsed,'agent_seconds':sorted(spans,key=lambda row:row['agent']),
                     'oracle_numerical_peak':activity.peak,'oracle_calls':activity.calls,
                     'oracle_rpc_calls':sum(a.oracle_call_count for a in audits),
                     'cache':cache.diagnostics if cache is not None else None,
                     'oracle_numerical_service_seconds_sum':activity.service_seconds,
                     'workers':len({a.pid for a in audits}),'batch_size':batch,
                     'execution_settings':service.execution_settings_snapshot(),
                     'errors':errors,'mean_error':float(np.mean(errors)),
                     'maximum_prediction_difference':float(max(np.max(np.abs(a-b)) for a,b in zip(predictions,baseline))),
                     'queue_wait_seconds_sum':float(sum(a.oracle_case_queue_wait_seconds if a.oracle_case_queue_wait_seconds is not None
                                                      else a.oracle_queue_wait_seconds for a in audits))}
                rows.append(row)
                print(json.dumps(row),flush=True)
            finally:
                service._executor.shutdown(wait=True,cancel_futures=True)
                for session in sessions.values(): session.close()
                if cache is not None: cache.close()
                analytic.analytic_forcing=original_forcing
                analytic.analytic_velocity=original_velocity
            if output: Path(output).write_text(json.dumps({'status':'running','device':device,'rows':rows},indent=2)+'\n')
    summary=[]
    baseline_label=modes[0][0]
    serial=statistics.median(r['seconds'] for r in rows if r['mode']==baseline_label)
    for label,*_ in modes:
        selected=[r for r in rows if r['mode']==label]
        median=statistics.median(r['seconds'] for r in selected)
        summary.append({'mode':label,'median_seconds':median,'speedup_over_serial':serial/median if baseline_label=='serial' else None,
            'baseline_mode':baseline_label,'speedup_over_baseline':serial/median,
            'min_seconds':min(r['seconds'] for r in selected),'max_seconds':max(r['seconds'] for r in selected),
            'oracle_numerical_peak':max(r['oracle_numerical_peak'] for r in selected)})
    result={'status':'completed','kind':'NS_real_Oracle_spectral_solver_performance_fixture',
        'not_a_paper_accuracy_baseline':True,'device':device,'cases':count,'agents':2,'grid':grid,'steps':steps,
        'repeats':repeats,'oracle_cache':oracle_cache,'same_solver_all_modes':True,'predictions_match':True,'summary':summary,'rows':rows,
        'includes':['worker_startup','public_RPC','Oracle_numerics','solver_numerics','hidden_point_projection','cleanup'],
        'excludes':['Agent_LLM','Generator_training','Generator_sampling','Controller_truth_materialization','Promotion']}
    if output: Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes',default=None)
    parser.add_argument('--oracle-cache',action='store_true')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--grid',type=int,default=32)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--cases-per-agent',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(run(args.device,args.work_root,repeats=args.repeats,grid=args.grid,steps=args.steps,
        cases_per_agent=args.cases_per_agent,output=args.output,oracle_cache=args.oracle_cache,mode_names=args.modes),indent=2))
