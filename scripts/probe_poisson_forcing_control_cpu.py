#!/usr/bin/env python3
"""Hold Gaussian forcing change fixed across spatial scales; author diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from probe_poisson_perturbation_cpu import (
    ROOT, CENTERS, ProbeField, FineControl, frozen_program, replay, rel, check_mms, scale,
)


def equivalent_solution_rms(length, forcing_rms):
    # Deterministic ensemble normalization. Retains a Gaussian field, without
    # adapting amplitude to a realization or to a solver's error.
    x = np.linspace(0.,1.,max(8193,int(np.ceil(16/length))+1))
    d = x[:,None]-CENTERS
    k = np.exp(-.5*(d/length)**2)/np.sqrt(len(CENTERS))
    kp = -d/length**2*k
    kpp = (d*d/length**4-1/length**2)*k
    curvature = -2*k+2*(1-2*x[:,None])*kp+(x*(1-x))[:,None]*kpp
    variance = np.trapezoid(np.sum(curvature*curvature,axis=1),x)
    current_forcing_rms = scale(length,.1)*np.sqrt(2*variance)/np.pi**2
    return .1*forcing_rms/current_forcing_rms


def run(run_dir, destination, forcing_rms):
    started=time.perf_counter()
    if destination.exists(): raise FileExistsError(destination)
    bank=json.loads((run_dir/'data/terminal.json').read_text())
    children=[r for r in bank['cases'] if r['gaussian_coefficients']]
    methods={name:frozen_program(run_dir,name)[0] for name in ('common','without_p')}
    methods['author_fine_32769']=FineControl()
    dense=np.arange(1,4096)/4096
    rows=[];summaries=[]
    for length in (.2,.05,.02,.01,.005,.002):
        strength=equivalent_solution_rms(length,forcing_rms)
        for item in children:
            field=ProbeField(item['amplitude'],tuple(item['gaussian_coefficients']),length,strength)
            x=np.asarray(item['test_x'])
            base=item['amplitude']*np.sin(np.pi*x)
            row={'length_scale':length,'case_id':item['case']['case_id'],
                 'expected_solution_rms':strength,'expected_forcing_rms':forcing_rms,
                 'actual_solution_change':rel(field.solution(x),base),
                 'actual_forcing_change':rel(field.forcing(x),np.pi**2*base),
                 'mms_residual':check_mms(field),
                 'methods':{name:replay(method,field,x,dense) for name,method in methods.items()}}
            rows.append(row)
        group=rows[-len(children):]
        summary={'length_scale':length,'expected_solution_rms':strength,
                 'actual_solution_change_mean':float(np.mean([r['actual_solution_change'] for r in group])),
                 'actual_forcing_change_mean':float(np.mean([r['actual_forcing_change'] for r in group])),
                 'actual_forcing_change_max':max(r['actual_forcing_change'] for r in group),
                 'methods':{name:{metric:float(np.mean([r['methods'][name][metric] for r in group]))
                            for metric in ('original_512_error','dense_4095_error')} for name in methods}}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    result={'cpu_only':True,'agent_calls':0,'author_diagnostic':True,
            'normalization':'Expected L2 forcing change, fixed for the ensemble at each scale; not per-draw normalization.',
            'expected_forcing_rms':forcing_rms,'all_eight_archived_directions_retained':True,
            'scope':'Post-hoc diagnosis, not evidence of a new Agent learning or adaptation run.',
            'summaries':summaries,'cases':rows,'wall_seconds':time.perf_counter()-started}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'completed','wall_seconds':result['wall_seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'.paper_runs/poisson_perturbed_gaussian_cpu_20260923')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--forcing-rms',type=float,default=.3)
    args=parser.parse_args()
    if not np.isfinite(args.forcing_rms) or args.forcing_rms<=0:parser.error('forcing RMS must be finite and positive')
    run(args.run.resolve(),args.output.resolve(),args.forcing_rms)
