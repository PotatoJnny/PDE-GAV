#!/usr/bin/env python
"""Training-only blocked holdout selection for explicit NS continuous bases."""
from __future__ import annotations
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'legacy/ns/src'),str(ROOT/'scripts')]
import numpy as np
import torch
from study_ns_measured_representation import read_windows,error_report,derivatives,evaluate
from prepare_ns_reconstructed_generator import write_json,file_sha256
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid
from agent_verifier.neural_generator.structured_ns_answer_first import StructuredNSProjectionConfig,StructuredNSCoefficientProjector
from agent_verifier.neural_generator.structured_ns_spline_projection import JointSmoothNSCoefficientProjector


def plan(args):
    manifest=Path(args.manifest).resolve(); payload=json.loads(manifest.read_text())
    source=payload['splits']['generator_training']
    chosen=[source[i] for i in np.linspace(0,len(source)-1,args.windows,dtype=int)]
    for a,b in itertools.combinations(chosen,2):
        if a['source_recording']==b['source_recording'] and a['source_frames'][0]<b['source_frames'][1] and b['source_frames'][0]<a['source_frames'][1]:
            raise ValueError('selected training windows overlap raw frames')
    families=[]
    for k,nt in ((18,13),(26,21)):
        for space,temporal in itertools.product(('cosine','bspline'),('chebyshev','bspline')):
            cfg=StructuredNSProjectionConfig(spatial_mode_x=k,spatial_mode_y=k,time_order=nt-1,
                spatial_extension=.075 if space=='cosine' else 0.,spatial_basis=space,temporal_basis=temporal,
                fitting_method='joint_smoothing',ridge_relative=1e-6)
            families.append({'id':f'{space}_{temporal}_s{k+1}_t{nt}','config':cfg.to_dict()})
    return {'schema':'ns_spline_training_cv_v2','manifest':str(manifest),'manifest_sha256':file_sha256(manifest),
        'geometry':json.loads(Path(args.geometry_json).read_text()),'selections':{'generator_training':chosen},
        'families':families,'lambda_grid':[0.,1e-8,1e-6], 'viscosity':1.0006053268765131e-6,
        'time_holdout_frames':[8,9,10,11], 'space_holdout':'(ix+2*iy)%5==0 on interior; edges retained',
        'fit_mask':'retained times x retained sites; same raw frames never appear in another selected window',
        'score':'mean of spatial holdout at retained times and temporal holdout at retained sites, mean per-case relative velocity L2',
        'tie':'lowest score then fewer coefficients; deterministic original grid order for exact ties',
        'eligibility_search':'within each family, inspect CV-ordered full fits until first eligible; failed best lambda does not discard other eligible lambdas',
        'confirmation':'generator_validation only after this training selection; already used in earlier studies, not untouched test',
        'gate':'require >=0.003 absolute CV improvement over v2 same-mask control; derivative RMS/forcing <=2x v2 and maxima <=3x v2 engineering limits, not physical truth',
        'excluded_roles':['generator_validation','development','verifier','final_test','reconstruction_validation']}


def derivative_gate(row,baseline):
    return (row['score']<=baseline['score']-.003 and all(
        row['derivatives'][q]['rms']<=2*baseline['derivatives'][q]['rms'] and
        row['derivatives'][q]['maximum_absolute']<=3*baseline['derivatives'][q]['maximum_absolute']
        for q in ('gradient','time_derivative','laplacian','forcing')))


def run(args):
    torch.set_num_threads(args.threads)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    study=plan(args);write_json(out/'plan.json',study)
    data,(x,y,t)=read_windows(study); native=data['generator_training']
    grid,(ix,iy)=completed_grid(x,y,study['geometry'])
    if not grid.fluid_mask.all():
        raise ValueError('this diagnostic requires a complete measured rectangle')
    target=np.zeros((*native.shape[:3],len(grid.x),len(grid.y)));target[...,ix,iy]=native
    flat=target.reshape(*target.shape[:3],-1)
    xx,yy=np.meshgrid(np.arange(len(grid.x)),np.arange(len(grid.y)),indexing='ij')
    sh=((xx+2*yy)%5==0)&(xx>0)&(xx<len(grid.x)-1)&(yy>0)&(yy<len(grid.y)-1);sh=sh.ravel()
    th=np.zeros(len(t),bool);th[study['time_holdout_frames']]=True
    def metrics(pred):
        space=error_report(pred[:,~th,:][:,:,:,sh],flat[:,~th,:][:,:,:,sh])
        temporal=error_report(pred[:,th,:][:,:,:,~sh],flat[:,th,:][:,:,:,~sh])
        return {'spatial_holdout':space,'time_holdout':temporal,
                'score':.5*(space['velocity']['mean']+temporal['velocity']['mean'])}
    rows=[];full_rows=[]
    for name,k,e in [('historical_v1',16,.25),('current_v2',18,.075)]:
        started=time.time();cfg=StructuredNSProjectionConfig(spatial_mode_x=k,spatial_mode_y=k,time_order=12,spatial_extension=e)
        cg=replace(grid,fluid_mask=grid.fluid_mask&(~sh.reshape(grid.fluid_mask.shape)))
        cv=StructuredNSCoefficientProjector(geometry=cg,t=t[~th],config=cfg)
        cc,_=cv.project(velocity=target[:,~th],pressure=np.zeros_like(target[:,~th,0]))
        full=StructuredNSCoefficientProjector(geometry=grid,t=t,config=cfg)
        row={'id':name,'config':cfg.to_dict(),**metrics(evaluate(full,cc))}
        fc,_=full.project(velocity=target,pressure=np.zeros_like(target[:,:,0]))
        row['full_fit']=error_report(evaluate(full,fc),flat)
        row['derivatives']=derivatives(fc.detach().numpy(),cfg.to_dict(),study,t,grid)
        row['seconds']=time.time()-started;rows.append(row);full_rows.append(row)
        np.save(out/f'{name}_coefficients.npy',fc.detach().numpy())
        print(json.dumps({'event':'control',**row}),flush=True)
    for family in study['families']:
        start=time.time();cfg=StructuredNSProjectionConfig.from_dict(family['config'])
        projector=JointSmoothNSCoefficientProjector(geometry=grid,t=t,config=cfg)
        system=projector.prepare_system(time_mask=~th,spatial_mask=~sh)
        family_rows=[]
        for ls,lt in itertools.product(study['lambda_grid'],repeat=2):
            tick=time.time();chosen=replace(cfg,smoothing_spatial=ls,smoothing_temporal=lt)
            row={'id':family['id']+f'_ls{ls:g}_lt{lt:g}','family':family['id'],'config':chosen.to_dict()}
            try:
                c,diag=projector.solve(flat,system=system,smoothing_spatial=ls,smoothing_temporal=lt)
                row.update(metrics(projector.predict(c).detach().numpy()));row['convergence']=diag
            except RuntimeError as exc:
                row['failure']=str(exc)
            row['seconds']=time.time()-tick;family_rows.append(row);rows.append(row)
            print(json.dumps({'event':'candidate',**row}),flush=True)
            write_json(out/'progress.json',{'rows':rows,'full_fits':full_rows})
        valid=[r for r in family_rows if 'score' in r]
        if not valid: continue
        baseline=next(r for r in full_rows if r['id']=='current_v2')
        ordered=sorted(valid,key=lambda r:r['score'])
        for index,best in enumerate(ordered):
            if index and best['score']>baseline['score']-.003: break
            bcfg=StructuredNSProjectionConfig.from_dict(best['config'])
            c,diag=projector.solve(flat,smoothing_spatial=bcfg.smoothing_spatial,smoothing_temporal=bcfg.smoothing_temporal)
            cc=torch.stack((c,torch.zeros_like(c)),dim=1).reshape(len(c),2,bcfg.time_coefficient_count,bcfg.spatial_mode_x+1,bcfg.spatial_mode_y+1)
            fitted={**best,'full_fit':error_report(projector.predict(c).detach().numpy(),flat),
                'derivatives':derivatives(cc.detach().numpy(),bcfg.to_dict(),study,t,grid),'full_convergence':diag}
            full_rows.append(fitted);np.save(out/f"{best['id']}_coefficients.npy",cc.detach().numpy())
            print(json.dumps({'event':'family_full_fit','family':family['id'],'seconds':time.time()-start,'candidate':fitted}),flush=True)
            write_json(out/'progress.json',{'rows':rows,'full_fits':full_rows})
            if derivative_gate(fitted,baseline): break
    baseline=next(r for r in full_rows if r['id']=='current_v2')
    candidates=[]
    for row in full_rows:
        if 'family' not in row:continue
        gate=derivative_gate(row,baseline)
        row['eligible_for_generator']=gate
        if gate:candidates.append(row)
    best=min(candidates,key=lambda r:(r['score'],(r['config']['spatial_mode_x']+1)**2*(r['config']['time_order']+1))) if candidates else None
    write_json(out/'results.json',{'plan':study,'rows':rows,'full_fits':full_rows,'selected':best})
    if best:write_json(out/'selected_projection_config.json',best['config'])
    print(json.dumps({'event':'complete','selected':None if best is None else best['id']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--geometry-json',required=True)
    p.add_argument('--out',required=True);p.add_argument('--windows',type=int,default=12);p.add_argument('--threads',type=int,default=8)
    run(p.parse_args())
