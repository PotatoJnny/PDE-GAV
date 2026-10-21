"""Compare real NS Oracle warm cache and eager lanes without solver timing."""
import argparse
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import time
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'legacy/ns/src'),str(ROOT/'scripts')]
from benchmark_parallel_evaluation import _fields
from pde_agent_verifier_solver.plugins.ns.analytic_cuda import NSAnalyticCudaCache
from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSPublicSession
from pde_agent_verifier_solver.engine.fair_execution import bind_oracle_policy
from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
import torch

def run(output):
 torch.set_num_threads(1)
 rows=[]
 for attach_cache in (False,True):
  for concurrency in (1,2,4):
   fields=_fields('cuda:0',8,.02)
   cache=NSAnalyticCudaCache('cuda:0') if attach_cache else None
   sessions=[]
   for i,f in enumerate(fields):
    f._analytic_cuda_cache=cache
    case=Case(str(i),CaseSource.GENERATED,CaseVisibility.DEVELOPMENT,{})
    sessions.append(_NSPublicSession(case,f.evaluate_forcing,f.evaluate_velocity,0.,f.domain_bounds,f.time_interval,2048))
   xy=np.random.default_rng(101).uniform(size=(2048,2))
   coords=np.column_stack((np.full(2048,.01),xy))
   def query(i):
    with bind_oracle_policy(f'agent-{i%2}',concurrency):
     return sessions[i].query_forcing(coords)
   for field in fields: field._analytic_cuda_cache=None
   reference=[field.evaluate_forcing(coords) for field in fields]
   for field in fields: field._analytic_cuda_cache=cache
   durations=[]
   try:
    with ThreadPoolExecutor(concurrency) as pool:
     start=time.perf_counter()
     for _ in range(8):
      values=list(pool.map(query,range(8)))
      for actual,expected in zip(values,reference):np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=1e-9)
     warmup=time.perf_counter()-start
     for _ in range(5):
      start=time.perf_counter()
      values=list(pool.map(query,range(8)))
      durations.append(time.perf_counter()-start)
      for actual,expected in zip(values,reference):np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=1e-9)
    row={'cache_attached':attach_cache,'concurrency':concurrency,'seconds':durations,
         'median_seconds':statistics.median(durations),'warmup_seconds':warmup,'uncached_reference_parity':True,
         'cache':cache.diagnostics if cache else None}
    rows.append(row); print(json.dumps(row),flush=True)
    Path(output).write_text(json.dumps({'status':'running','rows':rows},indent=2)+'\n')
   finally:
    for s in sessions:s.close()
    if cache:cache.close()
 result={'status':'completed','kind':'NS_forcing_8_cases_2048_points_uniform_time','rows':rows}
 Path(output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
