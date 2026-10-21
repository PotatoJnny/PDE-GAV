#!/usr/bin/env python3
"""Small synthetic NS throughput check; no Generator training or Agent calls."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'legacy/ns/src')]


def run(device, cases=8, points=2048, repeats=3):
    import torch
    from agent_verifier.neural_generator import structured_ns_answer_first as mms
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer, _NSPublicSession
    from pde_agent_verifier_solver.engine.fair_execution import bind_oracle_policy
    from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
    torch.set_num_threads(1)
    config = mms.StructuredNSProjectionConfig(spatial_mode_x=3, spatial_mode_y=3, time_order=3)
    geometry = ContinuousGeometrySpec(domain_bounds=(0.,1.,0.,1.),
        obstacle_polygons=(((.3,.3),(.6,.3),(.45,.6)),))
    random = torch.Generator().manual_seed(111)
    sessions = []
    for i in range(cases):
        coefficients = mms.StructuredNSCaseCoefficients(
            psi=torch.randn((4,16),generator=random,dtype=torch.float64)*1e-4,
            pressure=torch.randn((4,16),generator=random,dtype=torch.float64)*1e-4)
        field = mms.ContinuousStructuredNSMMS(coefficients,config=config,geometry_spec=geometry,
            domain_bounds=(0.,1.,0.,1.),viscosity=.01,t0=0.,t1=.12,device=device)
        answer = StructuredNSMMSAnswer(field)
        case = Case(str(i),CaseSource.GENERATED,CaseVisibility.DEVELOPMENT,{})
        sessions.append(_NSPublicSession(case,answer.evaluate_forcing,answer.evaluate_velocity,
            0.,answer.domain_bounds,answer.time_interval,points))
    coordinates = np.random.default_rng(101).random((points,3))
    coordinates[:,0] *= .12
    def query(i, concurrency):
        with bind_oracle_policy(f'agent-{i%2}', concurrency):
            return sessions[i].query_forcing(coordinates)
    reference = [query(i,1) for i in range(cases)]
    results = []
    for concurrency in (1,2,4):
        durations = []
        with ThreadPoolExecutor(concurrency) as pool:
            for _ in range(repeats):
                started = time.perf_counter()
                values = list(pool.map(lambda i: query(i,concurrency),range(cases)))
                durations.append(time.perf_counter()-started)
                for actual, expected in zip(values,reference):
                    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=1e-9)
        results.append({'concurrency':concurrency,'seconds':durations,'median_seconds':float(np.median(durations))})
    for session in sessions: session.close()
    return {'kind':'synthetic_NS_public_forcing_only','device':device,'cases':cases,
        'points_per_case':points,'repeats':repeats,'numerical_parity':True,'measurements':results,
        'includes_solver_or_generation':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run(args.device)
    text=json.dumps(result,indent=2)
    print(text)
    if args.output: args.output.write_text(text+'\n')
