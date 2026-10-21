"""Budgeted exact-source grid selection and atomic common-time refinement."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

_spec=importlib.util.spec_from_file_location('_source_geometry',Path(__file__).with_name('source_geometry.py'))
_geometry=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_geometry)

PILOT_SECONDS=15.
SOURCE_ORDER=12
MAX_SOURCE_NODES=480
TOLERANCE=1.e-5


def design(case,velocity,quadrature,query,began,allowance,reserve_exception):
    started=candidate_solve_clock()
    limit=min(PILOT_SECONDS,max(0.,allowance-(started-began)-120.))
    domain=case.public_data['equation']['domain']
    tmin,tmax=map(float,domain['time_interval']);horizon=tmax-tmin
    xmin,xmax=map(float,domain['x_interval']);length=xmax-xmin
    lower,upper=map(float,domain['velocity_query_window'])
    info=dict(accepted=False,reason='not_started',pilot_limit=limit,
              tolerance=TOLERANCE,source_order=SOURCE_ORDER)
    reserve=allowance-(started-began)-limit
    def sample(v,times,x):
        if candidate_solve_clock()-started>=limit:
            raise reserve_exception('Source compression pilot budget')
        tt=np.repeat(times,len(x));xx=np.tile(x,len(times));nv=len(v)
        points=np.column_stack((np.repeat(tt,nv),np.repeat(xx,nv),np.tile(v,len(tt))))
        values=horizon*query(points,reserve=reserve).reshape(len(tt),nv)
        if not np.isfinite(values).all():
            raise reserve_exception('Nonfinite source compression pilot')
        return values
    golden=.6180339887498948482
    times=tmin+horizon*np.unique(np.r_[.5*(1-np.cos(np.pi*np.arange(17)/16)),
                                     (np.arange(8)*golden+.3141592653589793)%1])
    x=xmin+length*((np.arange(6)*golden+.1732050807568877)%1)
    try:
        edges,adapt_info=_geometry.adapt_elements(
            lambda v:sample(v,times,x),lower,upper,SOURCE_ORDER,MAX_SOURCE_NODES,
            TOLERANCE,target_v=velocity,target_weights=quadrature)
        info.update(adapt_info)
        if not adapt_info['passed']:
            info['reason']='adaptation_defect'
            return None,info
        source_v,_=_geometry.gauss_nodes(edges,SOURCE_ORDER)
        target_time_nodes=257 if len(source_v)<=360 else 193
        stencil=_geometry.element_interpolation(edges,SOURCE_ORDER,velocity)
        # A broad independent spatial panel catches components localized away
        # from the small fitting x set. Its first slice is the initial source;
        # the other times are irrational interior fractions of this prefix.
        spatial_times=tmin+horizon*np.array([0.,.3819660112501052,.7548776662466927])
        spatial_x=xmin+length*(np.arange(64)+.3717320508075689)/64
        spatial_ref=sample(velocity,spatial_times,spatial_x)
        spatial_values=sample(source_v,spatial_times,spatial_x)
        spatial_residual=_geometry.interpolate(spatial_values,stencil)-spatial_ref
        spatial_defect=float(np.max(np.sqrt(np.sum(
            spatial_residual**2*(quadrature*(1+velocity**6)),axis=-1))))
        info['spatial_validation_defect']=spatial_defect
        if not np.isfinite(spatial_defect) or spatial_defect>TOLERANCE:
            info['reason']='spatial_defect'
            return None,info
        # All proposed source times and an additional irregular time holdout,
        # at three independent irrational x locations, compared with every
        # original DG velocity node. Independent order24 v probes already
        # participate in the adaptive fit at all150 pilot (t,x) rows.
        held_times=tmin+horizon*np.unique(np.r_[.5*(1-np.cos(np.pi*np.arange(target_time_nodes)/(target_time_nodes-1))),
                                  (np.arange(19)*.4142135623730951+.2718281828459045)%1])
        held_x=xmin+length*((np.arange(3)*.7548776662466927+.4142135623730951)%1)
        ref=sample(velocity,held_times,held_x)
        vals=sample(source_v,held_times,held_x)
        residual=_geometry.interpolate(vals,stencil)-ref
        defect=float(np.max(np.sqrt(np.sum(residual**2*(quadrature*(1+velocity**6)),axis=-1))))
        info['validation_defect']=defect
        if not np.isfinite(defect) or defect>TOLERANCE:
            info['reason']='independent_defect'
            return None,info
        info.update(accepted=True,reason='validated',source_velocity_nodes=len(source_v),target_time_nodes=target_time_nodes)
        return dict(velocity=source_v,indices=stencil[0],weights=stencil[1],edges=edges,time_nodes=target_time_nodes),info
    except reserve_exception:
        info['reason']='pilot_budget_or_nonfinite'
        return None,info
    finally:
        info['pilot_seconds']=candidate_solve_clock()-started


def refine(source_samples,times,x,source_velocity,project,query,began,allowance,
           slice_costs,reserve_exception,refinement_reserve,new_nt=None):
    old_nt,nx,nv=source_samples.shape
    new_nt=2*old_nt-1 if new_nt is None else int(new_nt)
    ratio=(new_nt-1)//(old_nt-1)
    if ratio<2 or ratio*(old_nt-1)!=new_nt-1:
        raise ValueError('Source refinement must be an integer nested Chebyshev level')
    new_count=new_nt-old_nt
    info=dict(attempted=False,completed=False,parent_nodes=old_nt,source_nodes=new_nt,
              sampled_slices=0,method='compressed_common_panel')
    estimate=max(float(np.mean(slice_costs)),float(np.mean(slice_costs[-4:])))
    remaining=allowance-(candidate_solve_clock()-began)
    if estimate*new_count>=remaining-refinement_reserve:
        info['reason']='forecast_budget';return source_samples,times,info
    info['attempted']=True
    refined_times=times[0]+(times[-1]-times[0])*(1-np.cos(np.pi*np.arange(new_nt)/(new_nt-1)))/2
    refined_times[::ratio]=times
    refined=torch.empty((new_nt,nx,nv),dtype=source_samples.dtype,device=source_samples.device)
    refined[::ratio]=source_samples
    ns=len(source_velocity)
    points=np.column_stack((np.zeros(nx*ns),np.repeat(x,ns),np.tile(source_velocity,nx)))
    measured=[]
    try:
        for ordinal,index in enumerate(i for i in range(new_nt) if i%ratio):
            start=candidate_solve_clock()
            predicted=max(estimate,float(np.mean(measured[-4:])) if measured else 0.)
            if predicted*(new_count-ordinal)>=allowance-(start-began)-refinement_reserve:
                info['reason']='measured_budget';return source_samples,times,info
            points[:,0]=refined_times[index]
            values=query(points,reserve=refinement_reserve).reshape(nx,ns)
            if not np.isfinite(values).all():
                info['reason']='nonfinite_source';return source_samples,times,info
            refined[index].copy_(project(values))
            torch.cuda.synchronize(source_samples.device)
            measured.append(max(1e-6,candidate_solve_clock()-start))
            info['sampled_slices']=ordinal+1
    except reserve_exception:
        info['reason']='query_budget';return source_samples,times,info
    info.update(completed=True,reason='complete',measured_seconds=float(sum(measured)))
    slice_costs.extend(measured)
    return refined,refined_times,info
