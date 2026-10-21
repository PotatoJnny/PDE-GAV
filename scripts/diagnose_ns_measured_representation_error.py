#!/usr/bin/env python
"""Separate finite-basis capacity from the shared-streamfunction constraint.

Independent component fits are diagnostic only: they are generally divergent
and must never be exported as incompressible NS answers or Generator corpora.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'legacy/ns/src')]
import numpy as np
import torch
from study_ns_measured_representation import read_windows, error_report
from prepare_ns_reconstructed_generator import write_json, file_sha256
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid, numerical_symbols


def fixed_penalty_operator(design, penalty):
    """Same physical coefficient L2 penalty, with safe column equilibration."""
    scale=design.square().mean(0).sqrt()
    active=scale > scale.max().clamp_min(1e-15)*1e-10
    a=design[:,active]; s=scale[active]
    normalized=a/s
    solution=torch.linalg.solve(normalized.T@normalized+torch.diag(penalty/s.square()), normalized.T)/s[:,None]
    output=torch.zeros((design.shape[1],design.shape[0]),dtype=design.dtype)
    output[active]=solution
    return output


def main(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path(args.study_plan).read_text())
    source['selections']={k:v for k,v in source['selections'].items() if k=='generator_validation'}
    cfgs=[]
    for k,e in [(16,.25),(18,.075),(24,.075),(32,.075)]:
        cfgs.append(numerical_symbols().StructuredNSProjectionConfig(spatial_mode_x=k,spatial_mode_y=k,
            time_order=12,spatial_extension=e,ridge_relative=1e-6).to_dict())
    source['diagnostic_configs']=cfgs
    source['diagnostic_policy']={'capacity':'increase K with same basis family, extension=.075, same time order and ridge',
        'constraint':'separate psi_u/psi_v in EXACT original component design spaces, same physical ridge penalty',
        'only_diagnostic':'independent component fits generally violate incompressibility; high-K is not a P configuration',
        'scoring':'observations used for per-window fitting, plus fixed interior spatial holdout; no final/verifier data',
        'source_plan_sha256':file_sha256(args.study_plan)}
    write_json(out/'plan.json',source)
    data,(x,y,times)=read_windows(source)
    grid,(ix,iy)=completed_grid(x,y,source['geometry'])
    raw=data['generator_validation']; target=np.zeros((*raw.shape[:3],len(grid.x),len(grid.y)))
    target[...,ix,iy]=raw
    from dataclasses import replace
    xx,yy=np.indices(grid.fluid_mask.shape)
    hold=((xx+2*yy)%5==0)&(xx>0)&(xx<len(grid.x)-1)&(yy>0)&(yy<len(grid.y)-1)
    results=[]
    for config in cfgs:
        started=time.time();row={'projection_config':config}
        module=numerical_symbols();cfg=module.StructuredNSProjectionConfig.from_dict(config)
        full=module.StructuredNSCoefficientProjector(geometry=grid,t=times,config=cfg)
        design_full=full.spatial_operators['velocity_design']
        for label,fit_grid in [('all_observations',grid),('spatial_holdout',replace(grid,fluid_mask=grid.fluid_mask&~hold))]:
            projector=full if label=='all_observations' else module.StructuredNSCoefficientProjector(geometry=fit_grid,t=times,config=cfg)
            design=projector.spatial_operators['velocity_design'];npoints=int(fit_grid.fluid_mask.sum())
            values=torch.tensor(target[...,fit_grid.fluid_mask],dtype=torch.float64).reshape(len(target),len(times),-1)
            scale=design.square().mean(0).sqrt();active=scale>scale.max().clamp_min(1e-15)*1e-10
            penalty=cfg.ridge_relative*design.shape[0]*scale[active].median().square()
            shared=(values@projector.velocity_projection.T)
            explicit_shared=values@fixed_penalty_operator(design,penalty).T
            torch.testing.assert_close(shared,explicit_shared,rtol=2e-8,atol=1e-10)
            # Equal physical regularization isolates the relaxation of psi sharing.
            separated=[]
            for j in range(2):
                op=fixed_penalty_operator(design[j*npoints:(j+1)*npoints],penalty)
                separated.append(values[...,j*npoints:(j+1)*npoints]@op.T)
            rows={}
            for method,per_time in [('shared_streamfunction',None),('independent_components_diagnostic',None)]:
                spatial=[];spacetime=[]
                for j in range(2):
                    coefficients=shared if method=='shared_streamfunction' else separated[j]
                    eval_design=design_full[j*grid.fluid_mask.size:(j+1)*grid.fluid_mask.size]
                    spatial.append(coefficients@eval_design.T)
                    continuous=torch.einsum('tr,rs,bsf->btf',projector.temporal_design,projector.temporal_projection,coefficients)
                    spacetime.append(continuous@eval_design.T)
                pred=torch.stack(spatial,2).numpy();continuous=torch.stack(spacetime,2).numpy()
                truth=target.reshape(*target.shape[:3],-1)
                if label=='spatial_holdout':
                    pred=pred[...,hold.ravel()];continuous=continuous[...,hold.ravel()];truth=truth[...,hold.ravel()]
                rows[method]={'spatial_only':error_report(pred,truth),'space_time':error_report(continuous,truth)}
            rows['physical_ridge_penalty']=float(penalty)
            row[label]=rows
        row['elapsed_seconds']=time.time()-started
        results.append(row);write_json(out/'results.partial.json',results)
        print(json.dumps({'K':cfg.spatial_mode_x,'extension':cfg.spatial_extension,'errors':{label:{method:row[label][method]['space_time']['velocity']['mean'] for method in ['shared_streamfunction','independent_components_diagnostic']} for label in ['all_observations','spatial_holdout']},'seconds':row['elapsed_seconds']}),flush=True)
    write_json(out/'results.json',{'schema':'ns_measured_constraint_capacity_diagnosis_v1','plan_sha256':file_sha256(out/'plan.json'),'results':results,'caution':'A finite-basis constraint-relaxation comparison; not a mathematical approximation lower bound or proof of sensor noise/three-dimensional physics.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study-plan',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
