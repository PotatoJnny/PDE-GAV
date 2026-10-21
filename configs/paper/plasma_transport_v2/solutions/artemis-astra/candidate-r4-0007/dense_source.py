"""Budget-selected complete Chebyshev panel sharing a solved coarse panel."""
from __future__ import annotations
import numpy as np
import torch
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

MAX_SOURCE_NODES = 257


def refine_source(source_samples, times, x, velocity, quadrature, query,
                  began, allowance, slice_costs, reserve_exception,
                  refinement_reserve=40., forecast_factor=1.):
    nt,nx,nv=source_samples.shape
    info=dict(attempted=False,completed=False,source_nodes=nt,
              refinement_reserve=float(refinement_reserve),
              forecast_factor=float(forecast_factor),method='complete_budget_chebyshev',
              sampled_slices=0,baseline_source_nodes=nt)
    if nt not in (17,33):
        info['reason']='unsupported_checkpoint'
        return source_samples,times,info
    if not slice_costs or not np.isfinite(slice_costs).all():
        info['reason']='invalid_slice_costs'
        return source_samples,times,info
    estimate=max(float(np.mean(slice_costs)),float(np.mean(slice_costs[-4:])))
    remaining=allowance-(candidate_solve_clock()-began)
    options=[1+(nt-1)*m for m in range(2,(MAX_SOURCE_NODES-1)//(nt-1)+1)]
    affordable=[n for n in options if forecast_factor*estimate*(n-nt)<remaining-refinement_reserve]
    info.update(measured_slice_cost=estimate,remaining_before_sampling=remaining,
                candidate_source_nodes=options)
    if not affordable:
        info['reason']='forecast_budget'
        return source_samples,times,info
    target=max(affordable)
    info.update(attempted=True,target_source_nodes=target)
    tmin,tmax=float(times[0]),float(times[-1])
    refined_times=tmin+(tmax-tmin)*(1-np.cos(np.pi*np.arange(target)/(target-1)))/2
    stride=(target-1)//(nt-1)
    refined_times[::stride]=times
    refined=torch.empty((target,nx,nv),dtype=source_samples.dtype,device=source_samples.device)
    refined[::stride]=source_samples
    points=np.column_stack((np.zeros(nx*nv),np.repeat(x,nv),np.tile(velocity,nx)))
    missing=[j for j in range(target) if j%stride]
    measured=[]
    try:
        for ordinal,index in enumerate(missing):
            started=candidate_solve_clock()
            remaining=allowance-(started-began)
            predicted=max(estimate,float(np.mean(measured[-4:])) if measured else 0.)
            if forecast_factor*predicted*(len(missing)-ordinal)>=remaining-refinement_reserve:
                info['reason']='measured_budget'
                return source_samples,times,info
            points[:,0]=refined_times[index]
            values=query(points,reserve=refinement_reserve).reshape(nx,nv)
            if not np.isfinite(values).all():
                info['reason']='nonfinite_source'
                return source_samples,times,info
            refined[index].copy_(torch.as_tensor(values,device=source_samples.device))
            if source_samples.is_cuda:torch.cuda.synchronize(source_samples.device)
            measured.append(max(1.e-6,candidate_solve_clock()-started))
            info['sampled_slices']=ordinal+1
    except reserve_exception:
        info['reason']='query_budget'
        return source_samples,times,info
    info.update(completed=True,source_nodes=target,reason='complete',
                measured_seconds=float(sum(measured)))
    return refined,refined_times,info
