"""Streamed dyadic acquisition with conservative partial salvage.

Each complete observed frame is usable. A failed refinement patches complete
new odd frames into the previous uniform history's natural-cubic refinement;
all unobserved target knots retain their previous estimates. The first complete
level is still all-or-nothing. No clocks or Oracle capabilities are owned here.
"""
from __future__ import annotations
import numpy as np
from green_ops import refine_source_history


def acquire_dyadic_source_history(target_times,xy,relative_velocity,sample,*,
                                  check=None,oom_errors=(),first_count=5):
    """Acquire nested source frames and preserve completed partial refinements.

    A sampler may expose ``stream_rows(sites, velocity, method, callback)``.
    It must deliver contiguous complete finite rows as ``callback(offset,values)``
    with all full velocity columns (including expansion of a retained-node mask).
    A later failure leaves preceding callbacks usable. Ordinary samplers retain
    the framewise compatibility path. No interpolation runs on full success.
    """
    check=check or (lambda:None)
    times=np.asarray(target_times,float);sites_xy=np.asarray(xy,float)
    velocity=np.asarray(relative_velocity,float).reshape(-1,2)
    if times.ndim!=1 or len(times)<3 or not np.isfinite(times).all():
        raise ValueError('finite one-dimensional target times required')
    nt=len(times);intervals=nt-1
    if intervals&(intervals-1):raise ValueError('target time count must equal2**k+1')
    if isinstance(first_count,bool) or int(first_count)!=first_count or first_count<3 or first_count>nt:
        raise ValueError('valid first-level time count required')
    first_count=int(first_count);first_intervals=first_count-1
    if first_intervals&(first_intervals-1) or intervals%first_intervals:
        raise ValueError('first-level time count must be nested2**k+1')
    differences=np.diff(times)
    if differences[0]<=0 or not np.allclose(differences,differences[0],rtol=1e-10,atol=1e-13*abs(differences[0])):
        raise ValueError('uniform increasing target times required')
    if sites_xy.ndim!=2 or sites_xy.shape[1]!=2 or not np.isfinite(sites_xy).all():
        raise ValueError('finite (n**2,2) spatial sites required')
    n=int(round(np.sqrt(len(sites_xy))))
    if n<1 or n*n!=len(sites_xy) or len(velocity)<1 or not np.isfinite(velocity).all():
        raise ValueError('square spatial mesh and finite nonempty velocity nodes required')
    nv=len(velocity);spatial_points=n*n;frame_points=spatial_points*nv
    recoverable=(TimeoutError,ValueError,FloatingPointError,MemoryError)+tuple(oom_errors)
    stream=getattr(sample,'stream_rows',None)
    if stream is not None and not callable(stream):raise ValueError('callable stream_rows required')
    diagnostics={'method':'streamed dyadic acquisition with patch-only partial-level salvage' if stream is not None else 'framewise dyadic acquisition with patch-only partial-level salvage',
        'target_time_count':nt,'first_time_count':first_count,'spatial_n':n,'velocity_nodes':nv,
        'completed_time_count':0,'completed_levels':[],'represented_time_count':0,
        'observed_time_count':0,'interpolated_time_count':0,'acquired_complete_frame_count':0,
        'requested_source_points':0,'completed_source_points':0,'levels':[],
        'point_count_scope':'full Cartesian tensor entries before optional velocity pruning; completed counts include every fully acquired frame; authoritative actual Oracle points are tracked by the sampler'}
    committed=None;committed_times=None;count=first_count
    while count<=nt:
        stride=intervals//(count-1);indices=np.arange(0,nt,stride,dtype=int)
        target_new=indices if committed is None else indices[1::2]
        local_new=np.arange(count) if committed is None else np.arange(1,count,2)
        level={'time_count':count,'new_time_count':len(target_new),
            'target_frame_indices':target_new.tolist(),'status':'planned',
            'requested_new_frames':0,'completed_new_frames':0}
        diagnostics['levels'].append(level)
        completed_new=0
        try:
            check()
            candidate=np.empty((count,n,n,nv),dtype=float)
            if committed is not None:candidate[::2]=committed
            candidate_times=times[indices].copy()
            if stream is not None:
                query_sites=np.column_stack((np.repeat(times[target_new],spatial_points),
                    np.tile(sites_xy,(len(target_new),1))))
                accepted_rows=0
                def receive(offset,values):
                    nonlocal accepted_rows,completed_new
                    answer=np.asarray(values,float)
                    if (isinstance(offset,bool) or not isinstance(offset,(int,np.integer))
                            or offset!=accepted_rows or answer.ndim!=2 or answer.shape[1]!=nv
                            or not 0<len(answer)<=len(query_sites)-accepted_rows
                            or not np.isfinite(answer).all()):
                        raise FloatingPointError('invalid contiguous complete source row block')
                    consumed=0
                    while consumed<len(answer):
                        row=accepted_rows+consumed
                        frame,within=divmod(row,spatial_points)
                        length=min(spatial_points-within,len(answer)-consumed)
                        candidate[local_new[frame]].reshape(spatial_points,nv)[within:within+length]=answer[consumed:consumed+length]
                        consumed+=length
                    accepted_rows+=len(answer)
                    complete=accepted_rows//spatial_points
                    newly_complete=complete-completed_new
                    completed_new=complete;level['completed_new_frames']=complete
                    level['completed_new_rows']=accepted_rows
                    diagnostics['acquired_complete_frame_count']+=newly_complete
                    diagnostics['completed_source_points']+=newly_complete*frame_points
                level['status']='acquiring';level['requested_new_frames']=len(target_new)
                diagnostics['requested_source_points']+=len(target_new)*frame_points
                stream(query_sites,velocity,'source',receive)
                if accepted_rows!=len(query_sites):
                    raise FloatingPointError('stream ended before every source row was acquired')
                del query_sites
            else:
                for target_index,local_index in zip(target_new,local_new):
                    check()
                    query_sites=np.column_stack((np.full(spatial_points,times[target_index]),sites_xy))
                    level['status']='acquiring';level['requested_new_frames']+=1
                    diagnostics['requested_source_points']+=frame_points
                    answer=np.asarray(sample(query_sites,velocity,'source'),float)
                    if answer.shape!=(spatial_points,nv) or not np.isfinite(answer).all():
                        raise FloatingPointError('invalid complete source frame response')
                    candidate[local_index]=answer.reshape(n,n,nv)
                    completed_new+=1;level['completed_new_frames']=completed_new
                    diagnostics['acquired_complete_frame_count']+=1
                    diagnostics['completed_source_points']+=frame_points
                    del answer,query_sites
            committed,committed_times=candidate,candidate_times
            level['status']='committed'
            diagnostics['completed_time_count']=count;diagnostics['completed_levels'].append(count)
            diagnostics.update(represented_time_count=count,observed_time_count=count,interpolated_time_count=0)
            del candidate,candidate_times
        except recoverable as error:
            level.update(status='failed',reason=type(error).__name__+': '+str(error))
            if committed is None:raise
            diagnostics.update(status='coarse_complete',stopped_before_time_count=count,
                reason=type(error).__name__+': '+str(error))
            if completed_new:
                try:
                    check()
                    enriched,_=refine_source_history(committed,committed_times,factor=2)
                    check()
                    complete_positions=local_new[:completed_new]
                    enriched[complete_positions]=candidate[complete_positions]
                    check()
                    # Commit the enriched representation only after every
                    # allocation, patch, and cooperative deadline check passes.
                    observed_count=len(committed)+completed_new
                    diagnostics.update(status='partial_level_salvaged',
                        represented_time_count=count,observed_time_count=observed_count,
                        interpolated_time_count=count-observed_count,
                        salvage='complete odd frames overwrite exact coarse natural-cubic refinement; missing knots unchanged')
                    level['status']='partial_salvaged'
                    return enriched,candidate_times,diagnostics
                except recoverable as salvage_error:
                    diagnostics['salvage_failure']=type(salvage_error).__name__+': '+str(salvage_error)
            return committed,committed_times,diagnostics
        count=2*count-1
    diagnostics['status']='target_complete'
    return committed,committed_times,diagnostics
