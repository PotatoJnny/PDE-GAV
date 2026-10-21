#!/usr/bin/env python3
"""Author resolution-only controls on the archived no-P numerical method."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from types import ModuleType
import numpy as np
from probe_poisson_perturbation_cpu import ROOT, ProbeField, frozen_program, replay


def run(run_dir, destination, forcing_rms=None):
    started = time.perf_counter()
    if destination.exists():
        raise FileExistsError(destination)
    _, locator = frozen_program(run_dir, 'without_p')
    source = Path(locator).read_text()
    if source.count('n = 256') != 1:
        raise ValueError('Expected exactly one frozen Chebyshev resolution setting')
    bank = json.loads((run_dir/'data/terminal.json').read_text())
    children = [r for r in bank['cases'] if r['gaussian_coefficients']]
    dense = np.arange(1,4096)/4096
    rows=[]
    for degree in (256,512,1024):
        module = ModuleType('author_resolution_'+str(degree))
        exec(compile(source.replace('n = 256', f'n = {degree}'), '<author-resolution-control>', 'exec'), module.__dict__)
        for length in (.005,.002):
            for item in children:
                strength = .1
                if forcing_rms is not None:
                    from probe_poisson_forcing_control_cpu import equivalent_solution_rms
                    strength = equivalent_solution_rms(length, forcing_rms)
                field = ProbeField(item['amplitude'], tuple(item['gaussian_coefficients']), length, strength)
                metrics = replay(module, field, np.asarray(item['test_x']), dense)
                rows.append({'degree':degree,'length_scale':length,'case_id':item['case']['case_id'],**metrics})
            group=rows[-len(children):]
            print(json.dumps({'degree':degree,'length':length,
                              'mean_error':float(np.mean([r['original_512_error'] for r in group]))}),flush=True)
    summary=[{'degree':degree,'length_scale':length,**{
        metric:float(np.mean([r[metric] for r in rows if r['degree']==degree and r['length_scale']==length]))
        for metric in ('original_512_error','dense_4095_error')}}
        for degree in (256,512,1024) for length in (.005,.002)]
    result={'cpu_only':True,'agent_calls':0,'author_controls':True,
            'change':'Only n=256 changed to 512 or 1024 in an in-memory copy of the archived solver.',
            'source':locator,'expected_forcing_rms':forcing_rms,'scope':'Functional numerical diagnosis; not Agent adaptation evidence.',
            'wall_seconds':time.perf_counter()-started,'summaries':summary,'cases':rows}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'completed','wall_seconds':result['wall_seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'.paper_runs/poisson_perturbed_gaussian_cpu_20260923')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--forcing-rms',type=float)
    args=parser.parse_args()
    if args.forcing_rms is not None and (not np.isfinite(args.forcing_rms) or args.forcing_rms<=0):
        parser.error('forcing RMS must be finite and positive')
    run(args.run.resolve(),args.output.resolve(),args.forcing_rms)
