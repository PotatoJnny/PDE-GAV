"""Conservative optional source targets and bounded complete-panel rescue."""
from __future__ import annotations
import numpy as np
import torch
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

MAX_SOURCE_NODES=321
LEGACY_TARGETS=(65,97,129,193)
NEW_TARGET_FACTOR=1.10
MAX_ATTEMPTS=9


def refinement_levels(target,parent=33):
    multiple,remainder=divmod(int(target)-1,int(parent)-1)
    if remainder or multiple<2:
        raise ValueError('Target must refine the complete parent Chebyshev panel')
    multiples=[multiple]
    while multiple>1:
        factor=next(d for d in range(2,multiple+1) if multiple%d==0)
        multiple//=factor
        if multiple>1:multiples.append(multiple)
    return tuple(1+(parent-1)*d for d in reversed(multiples))


def refine_source(source_samples,times,x,velocity,quadrature,query,
                  began,allowance,slice_costs,reserve_exception,
                  refinement_reserve=30.,forecast_factor=1.):
    """Return only a complete regular panel, preserving the best checkpoint.

    Optional new sizes use a conservative estimate. A forecast abort may try a
    smaller affordable regular grid from the same original33 source samples.
    Exact-coordinate cached values are reused; no partial/irregular panel is
    ever returned. Oracle reserve exceptions or invalid data stop immediately.
    """
    nt,nx,nv=source_samples.shape
    info=dict(attempted=False,completed=False,source_nodes=nt,
              returned_source_nodes=nt,largest_complete_nodes=nt,
              target_nodes=nt,planned_target_nodes=nt,initial_target_nodes=nt,
              refinement_reserve=float(refinement_reserve),forecast_factor=float(forecast_factor),
              method='guarded_regular_chebyshev_with_rescue',sampled_slices=0,
              attempted_slices=0,completed_levels=[int(nt)],attempts=[],
              cached_slices_reused=0,retarget_count=0,new_target_factor=NEW_TARGET_FACTOR)
    best_panel,best_times=source_samples,times
    measured=[]

    def finish(reason):
        count=len(best_times)
        info.update(reason=reason,completed=count>nt,source_nodes=count,
                    returned_source_nodes=count,largest_complete_nodes=count,
                    target_completed=count==info['planned_target_nodes'],
                    measured_seconds=float(sum(measured)))
        return best_panel,best_times,info

    if nt!=33:return finish('parent_incomplete')
    costs=np.asarray(slice_costs,dtype=float)
    if not costs.size or not np.isfinite(costs).all() or np.any(costs<=0):
        return finish('invalid_slice_costs')
    baseline_estimate=max(float(np.mean(costs)),float(np.mean(costs[-4:])))
    guarded_estimate=max(NEW_TARGET_FACTOR*baseline_estimate,float(np.max(costs)))
    current_estimate=baseline_estimate
    targets=tuple(range(MAX_SOURCE_NODES,nt,-(nt-1)))
    info.update(target_choices=list(targets),initial_slice_estimate=baseline_estimate,
                guarded_slice_estimate=guarded_estimate,
                initial_remaining_seconds=float(allowance)-(candidate_solve_clock()-float(began)))
    tmin,tmax=float(times[0]),float(times[-1])
    cache={float(t).hex():source_samples[i] for i,t in enumerate(times)}
    plans={}
    points=np.column_stack((np.zeros(nx*nv),np.repeat(x,nv),np.tile(velocity,nx)))

    def plan(target):
        if target not in plans:
            tt=tmin+(tmax-tmin)*(1-np.cos(np.pi*np.arange(target)/(target-1)))/2
            stride=(target-1)//(nt-1)
            tt[::stride]=times
            plans[target]=(tt,[float(t).hex() for t in tt],stride)
        return plans[target]

    maximum=MAX_SOURCE_NODES
    for attempt_number in range(MAX_ATTEMPTS):
        remaining=float(allowance)-(candidate_solve_clock()-float(began))
        affordable=[]
        for target in targets:
            if target>maximum or target<=len(best_times):continue
            tt,keys,parent_stride=plan(target)
            needed=sum(key not in cache for key in keys)
            estimate=max(current_estimate,guarded_estimate) if target not in LEGACY_TARGETS else current_estimate
            if forecast_factor*estimate*needed<remaining-refinement_reserve:
                affordable.append((target,estimate,needed))
        if not affordable:
            return finish('forecast_budget' if not info['attempted'] else 'rescue_budget')
        target,estimate,needed=affordable[0]
        stages=refinement_levels(target,nt)
        refined_times,keys,parent_stride=plan(target)
        trace=dict(target=target,planned_levels=list(stages),forecast_estimate=float(estimate),
                   required_uncached_slices=needed,completed_levels=[int(nt)],
                   sampled_slices=0,cached_slices_reused=0,reason='running')
        info['attempts'].append(trace)
        info.update(attempted=True,target_nodes=target,planned_target_nodes=target,
                    required_slices=needed,planned_levels=list(stages),retarget_count=attempt_number)
        if attempt_number==0:info['initial_target_nodes']=target
        refined=torch.empty((target,nx,nv),dtype=source_samples.dtype,device=source_samples.device)
        refined[::parent_stride]=source_samples
        sampled=np.zeros(target,dtype=bool);sampled[::parent_stride]=True
        aborted=False
        for level in stages:
            stride=(target-1)//(level-1)
            indices=np.arange(0,target,stride)
            missing=[int(i) for i in indices if not sampled[i]]
            uncached_left=sum(keys[i] not in cache for i in missing)
            for index in missing:
                key=keys[index]
                if key in cache:
                    refined[index].copy_(cache[key]);sampled[index]=True
                    info['cached_slices_reused']+=1;trace['cached_slices_reused']+=1
                    continue
                started=candidate_solve_clock()
                remaining=allowance-(started-began)
                recent=float(np.mean(measured[-4:])) if measured else 0.
                predicted=max(estimate,recent)
                if forecast_factor*predicted*uncached_left>=remaining-refinement_reserve:
                    current_estimate=max(current_estimate,predicted)
                    trace['reason']='measured_budget';aborted=True
                    break
                points[:,0]=refined_times[index]
                info['attempted_slices']+=1
                try:
                    values=np.asarray(query(points,reserve=refinement_reserve),dtype=float).reshape(nx,nv)
                except reserve_exception:
                    trace['reason']='query_budget'
                    return finish('query_budget')
                if not np.isfinite(values).all():
                    trace['reason']='nonfinite_source'
                    return finish('nonfinite_source')
                refined[index].copy_(torch.as_tensor(values,device=source_samples.device))
                if source_samples.is_cuda:torch.cuda.synchronize(source_samples.device)
                measured.append(max(1.e-6,candidate_solve_clock()-started))
                cache[key]=refined[index]
                sampled[index]=True;uncached_left-=1
                info['sampled_slices']+=1;trace['sampled_slices']+=1
            if aborted:break
            trace['completed_levels'].append(int(level))
            if level>len(best_times):
                best_panel=refined[::stride];best_times=refined_times[::stride]
                info['completed_levels'].append(int(level))
                info['largest_complete_nodes']=int(level)
        if not aborted:
            trace['reason']='complete'
            return finish('complete')
        maximum=target-(nt-1)
        if maximum<=len(best_times):return finish('measured_budget')
    return finish('attempt_limit')
