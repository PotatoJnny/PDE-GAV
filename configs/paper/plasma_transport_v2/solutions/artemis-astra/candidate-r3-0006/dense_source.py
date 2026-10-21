"""Exact, atomic 65-to-129 source refinement after a completed baseline solve."""
from __future__ import annotations

import numpy as np
import torch
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


def refine_source(source_samples, times, x, velocity, quadrature, query,
                  began, allowance, slice_costs, reserve_exception,
                  refinement_reserve=30., forecast_factor=1.):
    """Reuse the 65 exact even slices, querying every x/v at all odd times.

    Only a complete finite level is returned. The caller already owns a
    completed baseline trajectory and uses it if sampling or evolution stops.
    """
    nt,nx,nv=source_samples.shape
    info=dict(attempted=False,completed=False,source_nodes=nt,
              refinement_reserve=float(refinement_reserve),
              forecast_factor=float(forecast_factor),method='exact_dense_129',
              sampled_slices=0)
    if nt!=65:
        info['reason']='parent_incomplete'
        return source_samples,times,info
    if not slice_costs or not np.isfinite(slice_costs).all():
        info['reason']='invalid_slice_costs'
        return source_samples,times,info
    estimate=max(float(np.mean(slice_costs)),float(np.mean(slice_costs[-4:])))
    remaining=allowance-(candidate_solve_clock()-began)
    if forecast_factor*estimate*64>=remaining-refinement_reserve:
        info['reason']='forecast_budget'
        return source_samples,times,info
    info['attempted']=True
    tmin,tmax=float(times[0]),float(times[-1])
    refined_times=tmin+(tmax-tmin)*(1-np.cos(np.pi*np.arange(129)/128))/2
    refined_times[::2]=times
    refined=torch.empty((129,nx,nv),dtype=source_samples.dtype,device=source_samples.device)
    refined[::2]=source_samples
    points=np.column_stack((np.zeros(nx*nv),np.repeat(x,nv),np.tile(velocity,nx)))
    measured=[]
    try:
        for ordinal,index in enumerate(range(1,129,2)):
            started=candidate_solve_clock()
            remaining=allowance-(started-began)
            predicted=max(estimate,float(np.mean(measured[-4:])) if measured else 0.)
            if forecast_factor*predicted*(64-ordinal)>=remaining-refinement_reserve:
                info['reason']='measured_budget'
                return source_samples,times,info
            points[:,0]=refined_times[index]
            values=query(points,reserve=refinement_reserve).reshape(nx,nv)
            if not np.isfinite(values).all():
                info['reason']='nonfinite_source'
                return source_samples,times,info
            refined[index].copy_(torch.as_tensor(values,device=source_samples.device))
            if source_samples.is_cuda:
                torch.cuda.synchronize(source_samples.device)
            measured.append(max(1.e-6,candidate_solve_clock()-started))
            info['sampled_slices']=ordinal+1
    except reserve_exception:
        info['reason']='query_budget'
        return source_samples,times,info
    info.update(completed=True,source_nodes=129,reason='complete',
                measured_seconds=float(sum(measured)))
    return refined,refined_times,info
