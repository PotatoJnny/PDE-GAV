#!/usr/bin/env python
"""Offline spatial-locality and time-compression diagnostics; no NS answers exported."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'legacy/ns/src')]
import numpy as np
import torch
from study_ns_measured_representation import read_windows,error_report
from prepare_ns_reconstructed_generator import write_json,file_sha256
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid,numerical_symbols


def main(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(args.study_plan).read_text())
    plan['selections']={k:v for k,v in plan['selections'].items() if k=='generator_validation'}
    plan['diagnostic']={'method':'four cardinal nearest measured neighbors, equal weights; all four are retained by fixed holdout mask',
        'temporal':'original Chebyshev order12 / ridge1e-6 projector, applied independently at each spatial location',
        'interpretation':'local interpolation is unconstrained, does not produce an incompressible NS answer; temporal-only residual is a representation diagnostic, not a noise estimate',
        'source_plan_sha256':file_sha256(args.study_plan)}
    write_json(out/'plan.json',plan)
    data,(x,y,times)=read_windows(plan);grid,(ix,iy)=completed_grid(x,y,plan['geometry'])
    assert grid.fluid_mask.all()
    values=data['generator_validation'];target=np.empty((*values.shape[:3],len(grid.x),len(grid.y)))
    target[...,ix,iy]=values
    xx,yy=np.indices(grid.fluid_mask.shape)
    hold=((xx+2*yy)%5==0)&(xx>0)&(xx<len(grid.x)-1)&(yy>0)&(yy<len(grid.y)-1)
    hx,hy=np.where(hold)
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        assert not hold[hx+dx,hy+dy].any()
    interpolated=sum(target[...,hx+dx,hy+dy] for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])/4
    module=numerical_symbols();cfg=module.StructuredNSProjectionConfig(spatial_mode_x=18,spatial_mode_y=18,
        spatial_extension=.075,time_order=12,ridge_relative=1e-6)
    projector=module.StructuredNSCoefficientProjector(geometry=grid,t=times,config=cfg)
    transform=(projector.temporal_design@projector.temporal_projection).numpy()
    temporal=np.einsum('ts,bscxy->btcxy',transform,target)
    time_interpolated=np.einsum('ts,bscp->btcp',transform,interpolated)
    result={'schema':'ns_measured_locality_diagnosis_v1','plan_sha256':file_sha256(out/'plan.json'),
        'heldout_points':len(hx),'local_four_neighbor_spatial_holdout':error_report(interpolated,target[...,hold]),
        'local_four_neighbor_plus_original_temporal_projection_holdout':error_report(time_interpolated,target[...,hold]),
        'original_temporal_projection_only_all_points':error_report(temporal,target),
        'caution':plan['diagnostic']['interpretation']}
    write_json(out/'results.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study-plan',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
