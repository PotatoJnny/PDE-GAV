"""Budget-selected complete Chebyshev panels, with atomic intermediate levels."""
from __future__ import annotations

import numpy as np
import torch
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


def refine_source(source_samples, times, x, velocity, quadrature, query,
                  began, allowance, slice_costs, reserve_exception,
                  refinement_reserve=30., forecast_factor=1.):
    """Refine a solved33 panel; return only a complete regular source panel.

    Choose the largest affordable target from193/129/97/65 for39G12, or
    129/97/65 otherwise. A129 attempt completes65 first; a193 attempt
    completes97 first. Budget or data failure retains the largest fully
    sampled panel, which the caller can evolve from its unchanged f0.
    """
    nt,nx,nv=source_samples.shape
    info=dict(attempted=False,completed=False,source_nodes=nt,
              returned_source_nodes=nt,largest_complete_nodes=nt,
              target_nodes=nt,planned_target_nodes=nt,
              refinement_reserve=float(refinement_reserve),
              forecast_factor=float(forecast_factor),
              method='budget_regular_chebyshev',sampled_slices=0,
              attempted_slices=0,completed_levels=[int(nt)])
    best_panel,best_times=source_samples,times
    measured=[]

    def finish(reason):
        count=len(best_times)
        info.update(reason=reason,completed=count>nt,source_nodes=count,
                    returned_source_nodes=count,largest_complete_nodes=count,
                    target_completed=count==info['planned_target_nodes'],
                    measured_seconds=float(sum(measured)))
        return best_panel,best_times,info

    if nt!=33:
        return finish('parent_incomplete')
    costs=np.asarray(slice_costs,dtype=float)
    if not costs.size or not np.isfinite(costs).all() or np.any(costs<=0):
        return finish('invalid_slice_costs')
    estimate=max(float(np.mean(costs)),float(np.mean(costs[-4:])))
    remaining=float(allowance)-(candidate_solve_clock()-float(began))
    # Actual source-query cost includes coarse sampling, independent probes
    # and exact odd-node fallback when required. Use that measured cost.
    targets=(385,257,193,129,97,65)
    info.update(target_choices=list(targets),initial_slice_estimate=estimate,
                initial_remaining_seconds=remaining)
    affordable=[target for target in targets
                if forecast_factor*estimate*(target-nt)<remaining-refinement_reserve]
    if not affordable:
        return finish('forecast_budget')
    target=affordable[0]
    stages={65:(65,),97:(97,),129:(65,129),193:(97,193),
            257:(65,129,257),385:(97,193,385)}[target]
    info.update(attempted=True,target_nodes=target,planned_target_nodes=target,
                required_slices=target-nt,planned_levels=list(stages))
    tmin,tmax=float(times[0]),float(times[-1])
    refined_times=tmin+(tmax-tmin)*(1-np.cos(np.pi*np.arange(target)/(target-1)))/2
    parent_stride=(target-1)//(nt-1)
    refined_times[::parent_stride]=times
    refined=torch.empty((target,nx,nv),dtype=source_samples.dtype,
                        device=source_samples.device)
    refined[::parent_stride]=source_samples
    sampled=np.zeros(target,dtype=bool)
    sampled[::parent_stride]=True
    points=np.column_stack((np.zeros(nx*nv),np.repeat(x,nv),np.tile(velocity,nx)))
    for level in stages:
        stride=(target-1)//(level-1)
        indices=np.arange(0,target,stride)
        missing=[int(index) for index in indices if not sampled[index]]
        for ordinal,index in enumerate(missing):
            started=candidate_solve_clock()
            remaining=allowance-(started-began)
            recent=float(np.mean(measured[-4:])) if measured else 0.
            predicted=max(estimate,recent)
            # Forecast only to the next complete useful level. This lets a
            # slower-than-expected attempt finish65/97 before giving up.
            if forecast_factor*predicted*(len(missing)-ordinal)>=remaining-refinement_reserve:
                return finish('measured_budget')
            points[:,0]=refined_times[index]
            info['attempted_slices']+=1
            try:
                values=np.asarray(query(points,reserve=refinement_reserve),dtype=float).reshape(nx,nv)
            except reserve_exception:
                return finish('query_budget')
            if not np.isfinite(values).all():
                return finish('nonfinite_source')
            refined[index].copy_(torch.as_tensor(values,device=source_samples.device))
            if source_samples.is_cuda:
                torch.cuda.synchronize(source_samples.device)
            measured.append(max(1.e-6,candidate_solve_clock()-started))
            sampled[index]=True
            info['sampled_slices']+=1
        # Every value in this view is now finite and complete. Later stages
        # touch only its complement, so this checkpoint remains unchanged.
        best_panel=refined[::stride]
        best_times=refined_times[::stride]
        info['completed_levels'].append(int(level))
        info['largest_complete_nodes']=int(level)
        info['measured_seconds']=float(sum(measured))
    return finish('complete')


def enrich_source(source_samples, times, x, velocity, quadrature, query,
                  began, allowance, slice_costs, reserve_exception,
                  refinement_reserve=45., forecast_factor=1.15,
                  maximum_nodes=385, batch_size=16):
    """Spend spare time bisecting widest gaps in atomic, immutable batches.

    This is called only after the regular panel has a completed trajectory.
    Each committed panel contains every old observation exactly. Incomplete
    batches are discarded, so no partially initialized tensor is returned.
    """
    nt, nx, nv = source_samples.shape
    best_panel, best_times = source_samples, times
    measured = []
    info = dict(method='widest_gap_midpoints', attempted=False, completed=False,
                initial_source_nodes=nt, source_nodes=nt, returned_source_nodes=nt,
                sampled_slices=0, attempted_slices=0, completed_batches=0,
                refinement_reserve=float(refinement_reserve),
                forecast_factor=float(forecast_factor), maximum_nodes=int(maximum_nodes),
                batch_size=int(batch_size), batch_source_nodes=[])

    def finish(reason):
        count = len(best_times)
        info.update(reason=reason, completed=count > nt, source_nodes=count,
                    returned_source_nodes=count, measured_seconds=float(sum(measured)),
                    discarded_slices=info['sampled_slices']-(count-nt))
        return best_panel, best_times, info

    costs = np.asarray(slice_costs, dtype=float)
    if nt < 65 or nt >= maximum_nodes:
        return finish('node_limit')
    if not costs.size or not np.isfinite(costs).all() or np.any(costs <= 0):
        return finish('invalid_slice_costs')
    estimate = max(float(np.mean(costs)), float(np.mean(costs[-4:])))
    info['initial_slice_estimate'] = estimate
    points = np.column_stack((np.zeros(nx*nv), np.repeat(x, nv), np.tile(velocity, nx)))
    while len(best_times)+batch_size <= maximum_nodes:
        started = candidate_solve_clock()
        remaining = float(allowance)-(started-float(began))
        recent = float(np.mean(measured[-4:])) if measured else 0.
        predicted = max(estimate, recent)
        if forecast_factor*predicted*batch_size >= remaining-refinement_reserve:
            return finish('forecast_budget')
        # A stable ordering resolves floating-point ties deterministically.
        gaps = np.diff(best_times)
        chosen = np.argsort(-gaps, kind='stable')[:batch_size]
        new_times = np.sort((best_times[chosen]+best_times[chosen+1])/2.)
        if np.any(new_times <= best_times[0]) or np.any(new_times >= best_times[-1]):
            return finish('unrepresentable_midpoint')
        combined_times = np.concatenate((best_times, new_times))
        order = np.argsort(combined_times, kind='stable')
        if np.any(np.diff(combined_times[order]) <= 0):
            return finish('duplicate_midpoint')
        fresh = torch.empty((batch_size, nx, nv), dtype=source_samples.dtype,
                            device=source_samples.device)
        info['attempted'] = True
        for ordinal, t in enumerate(new_times):
            started = candidate_solve_clock()
            remaining = float(allowance)-(started-float(began))
            recent = float(np.mean(measured[-4:])) if measured else 0.
            predicted = max(estimate, recent)
            if forecast_factor*predicted*(batch_size-ordinal) >= remaining-refinement_reserve:
                return finish('measured_budget')
            points[:, 0] = t
            info['attempted_slices'] += 1
            try:
                values = np.asarray(query(points, reserve=refinement_reserve), dtype=float).reshape(nx, nv)
            except reserve_exception:
                return finish('query_budget')
            if not np.isfinite(values).all():
                return finish('nonfinite_source')
            fresh[ordinal].copy_(torch.as_tensor(values, device=source_samples.device))
            if source_samples.is_cuda:
                torch.cuda.synchronize(source_samples.device)
            measured.append(max(1.e-6, candidate_solve_clock()-started))
            info['sampled_slices'] += 1
        combined = torch.cat((best_panel, fresh), dim=0)
        best_panel = combined[torch.as_tensor(order, device=source_samples.device)]
        best_times = combined_times[order]
        if source_samples.is_cuda:
            torch.cuda.synchronize(source_samples.device)
        info['completed_batches'] += 1
        info['batch_source_nodes'].append(len(best_times))
    return finish('node_limit')
