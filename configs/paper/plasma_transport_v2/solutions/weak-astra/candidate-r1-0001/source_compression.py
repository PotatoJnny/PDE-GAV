"""Validated CUR approximation of a prescribed public kinetic source.

All learning data are solver-selected evaluations of the public source.  The
small velocity basis is learned independently of later macro query points.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import qr, svd
from scipy.optimize import nnls
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

INITIAL_ROWS = 128
VALIDATION_ROWS = 64
MAXIMUM_ROWS = 384
MAXIMUM_RANK = 256
SOURCE_POINT_TARGET = 12_000_000
RELATIVE_TOLERANCE = 0.003
MAXIMUM_TIME_NODES = 129
MAXIMUM_SPATIAL_NODES = 32
TRAINING_TIME_COUNT = 17  # Set to zero for independent Sobol training times.
VALIDATION_TIME_COUNT = 16  # Set to zero for independent held-out times.


def velocity_weights(velocity):
    r2=np.sum(velocity**2,axis=1)
    return 1+(r2/25.)**1.5


def source_moments(velocity):
    vx,vy=velocity.T
    r2=vx*vx+vy*vy
    return np.stack((np.ones(len(vx)),vx,vy,vx*vx,vx*vy,vy*vy,.5*r2*vx,.5*r2*vy),axis=1)


def select_columns(basis,count):
    """QR pivots followed by greedy determinant-increasing oversampling."""
    rank,nvel=basis.shape
    count=min(nvel,max(rank,int(count)))
    _,_,piv=qr(basis,mode="economic",pivoting=True,check_finite=False)
    selected=list(map(int,piv[:rank]))
    block=basis[:,selected]
    gram_inv=np.linalg.inv(block@block.T)
    transformed=gram_inv@basis
    score=np.sum(basis*transformed,axis=0)
    available=np.ones(nvel,dtype=bool)
    available[selected]=False
    for _ in range(count-rank):
        if not np.isfinite(score).all() or np.min(score)<-1e-8:
            block=basis[:,selected]
            transformed=np.linalg.solve(block@block.T,basis)
            score=np.sum(basis*transformed,axis=0)
        column=int(np.argmax(np.where(available,score,-np.inf)))
        direction=transformed[:,column].copy()
        cross=direction@basis
        denominator=1+score[column]
        transformed-=np.outer(direction,cross)/denominator
        score-=cross*cross/denominator
        score=np.maximum(score,0.)
        selected.append(column)
        available[column]=False
    return np.asarray(selected,dtype=int)


def fit_basis(training,weights,maximum_rank=MAXIMUM_RANK):
    weighted=np.asarray(training,dtype=np.float64)*weights[None,:]
    _,singular,vh=svd(weighted,full_matrices=False,check_finite=False,lapack_driver="gesdd")
    if singular[0] < 1e-30:
        return np.ones((1,training.shape[1]))/np.sqrt(training.shape[1]),singular
    tail=np.sqrt(np.maximum(0.,np.sum(singular**2)-np.cumsum(singular**2)))/np.linalg.norm(singular)
    passing=np.flatnonzero(tail <= 2e-4)
    rank=int(passing[0]+1) if len(passing) else len(singular)
    rank=min(rank,int(maximum_rank),len(singular))
    return vh[:rank],singular


def reconstruction_operator(basis,selected):
    restriction=basis[:,selected].T
    return np.linalg.pinv(restriction,rcond=1e-11).T


def correct_source_moments(training,basis,selected,operator,weights,moments,cell_area):
    """Fit nonnegative empirical quadrature and enforce its eight raw moments.

    Quadrature weights are fitted anew from the current Case's public training
    rows; no held-out values enter this fit.  The coefficient change is the
    minimum weighted L2 change within the existing orthonormal source basis.
    """
    started=candidate_solve_clock()
    stats={"method":"positive_shared_quadrature","ridge_strength":0.,"applied":False}
    try:
        with threadpool_limits(limits=1):
            sample=training[:,selected]
            truth=training@moments*cell_area
            scale=np.maximum(np.sqrt(np.mean(truth**2,axis=0)),[.1,.3,.3,1.,.5,1.,1.,1.])
            moment_map=(basis/weights)@moments*cell_area
            original=weights[selected,None]*operator
            original_error=np.sqrt(np.mean((sample@original@moment_map-truth)**2,axis=0))
            stats["original_training_moment_rms"]=original_error.tolist()
            targets=np.array([.001,.003,.003,.01,.005,.01,.01,.01])
            if np.max(original_error/targets)<.01:
                stats["status"]="training_already_accurate"
                return operator,stats
            polynomial=moments[selected]
            design=(sample[:,None,:]*polynomial.T[None,:,:]/scale[None,:,None]).reshape(-1,len(selected))
            target=(truth/scale).ravel()
            quadrature,_=nnls(design,target,maxiter=20*len(selected))
            target_map=quadrature[:,None]*polynomial
            normalized=moment_map/scale
            condition=float(np.linalg.cond(normalized))
            stats["moment_map_condition"]=condition
            if not np.isfinite(condition) or condition>1e8:
                stats["status"]="ill_conditioned_moment_map"
                return operator,stats
            inverse=np.linalg.pinv(normalized,rcond=1e-12)
            corrected=original+((target_map-original@moment_map)/scale)@inverse
            growth=float(np.linalg.norm(corrected,2)/max(np.linalg.norm(original,2),1e-30))
            stats.update(operator_norm_growth=growth,weight_l1=float(np.sum(quadrature)),
                         weight_l2=float(np.linalg.norm(quadrature)),weight_max=float(np.max(quadrature)),
                         positive_weights=int(np.sum(quadrature>1e-12)),
                         fitted_training_moment_rms=np.sqrt(np.mean((sample@target_map-truth)**2,axis=0)).tolist())
            if not np.isfinite(corrected).all() or not np.isfinite(growth) or growth>3.:
                stats["status"]="operator_growth_guard"
                return operator,stats
            enforced=corrected@moment_map
            mismatch=float(np.linalg.norm((enforced-target_map)/scale)/max(np.linalg.norm(target_map/scale),1e-30))
            stats["moment_enforcement_relative_error"]=mismatch
            if not np.isfinite(mismatch) or mismatch>1e-8:
                stats["status"]="moment_enforcement_guard"
                return operator,stats
            stats.update(status="applied",applied=True)
            return corrected/weights[selected,None],stats
    except Exception as error:
        stats.update(status="fit_error",error=type(error).__name__+": "+str(error))
        return operator,stats
    finally:
        stats["seconds"]=candidate_solve_clock()-started


def validate(basis,selected,operator,values,weights,moments,cell_area):
    coeff=(values[:,selected]*weights[selected])@operator
    predicted=(coeff@basis)/weights[None,:]
    difference=predicted-values
    weighted=values*weights
    error=difference*weights
    weighted_rms=float(np.linalg.norm(error)/max(np.linalg.norm(weighted),1e-30))
    row_den=np.maximum(np.linalg.norm(weighted,axis=1),.05*np.linalg.norm(weighted)/np.sqrt(len(values)))
    weighted_max=float(np.max(np.linalg.norm(error,axis=1)/np.maximum(row_den,1e-30)))
    true_moments=values@moments*cell_area
    error_moments=difference@moments*cell_area
    floors=np.array([.1,.3,.3,1.,.5,1.,1.,1.])
    scales=np.maximum(np.sqrt(np.mean(true_moments**2,axis=0)),floors)
    normalized=np.sqrt(np.mean(error_moments**2,axis=0))/scales
    absolute=np.sqrt(np.mean(error_moments**2,axis=0))
    absolute_targets=np.array([.001,.003,.003,.01,.005,.01,.01,.01])
    return {"weighted_relative_rms":weighted_rms,
            "weighted_relative_max":weighted_max,
            "moment_max_normalized_rms":float(np.max(normalized)),
            "moment_normalized_rms":normalized.tolist(),
            "moment_absolute_rms":absolute.tolist(),
            "moment_absolute_target_ratio":float(np.max(absolute/absolute_targets))}


def validation_passes(metrics):
    return bool(metrics["weighted_relative_rms"]<RELATIVE_TOLERANCE
                and metrics["moment_max_normalized_rms"]<RELATIVE_TOLERANCE
                and metrics["moment_absolute_target_ratio"]<1.
                and metrics["weighted_relative_max"]<5*RELATIVE_TOLERANCE)


def coefficient_metrics(predicted,reference,residual_squared,reference_norm_squared,
                        reference_moments,moment_map):
    """Exact weighted norm from orthonormal coefficients and projection residual."""
    squared=np.sum((predicted-reference)**2,axis=1)+residual_squared
    norm2=np.asarray(reference_norm_squared)
    weighted_rms=float(np.sqrt(np.sum(squared)/max(np.sum(norm2),1e-30)))
    denominator=np.maximum(np.sqrt(norm2),.05*np.sqrt(np.mean(norm2)))
    weighted_max=float(np.max(np.sqrt(np.maximum(squared,0.))/np.maximum(denominator,1e-30)))
    moment_error=predicted@moment_map-reference_moments
    absolute=np.sqrt(np.mean(moment_error**2,axis=0))
    scales=np.maximum(np.sqrt(np.mean(reference_moments**2,axis=0)),[.1,.3,.3,1.,.5,1.,1.,1.])
    normalized=absolute/scales
    return {"rows":len(predicted),"weighted_relative_rms":weighted_rms,
            "weighted_relative_max":weighted_max,"moment_max_normalized_rms":float(np.max(normalized)),
            "moment_normalized_rms":normalized.tolist(),"moment_absolute_rms":absolute.tolist(),
            "moment_absolute_target_ratio":float(np.max(absolute/[.001,.003,.003,.01,.005,.01,.01,.01]))}


def chebyshev_times(interval,count):
    if count==1:
        return np.array([interval[0]])
    return interval[0]+.5*(interval[1]-interval[0])*(1-np.cos(np.arange(count)*np.pi/(count-1)))


def spatial_grid(periods,count):
    return np.stack(np.meshgrid(*(np.arange(count)*L/count for L in periods),
                                indexing="ij"),axis=-1).reshape(-1,2)


def coincident_grid_indices(old_count,new_count):
    """Exact integer correspondence of periodic Cartesian grid sites."""
    common=int(np.gcd(old_count,new_count))
    old_axis=np.arange(common)*(old_count//common)
    new_axis=np.arange(common)*(new_count//common)
    old=(old_axis[:,None]*old_count+old_axis[None,:]).reshape(-1)
    new=(new_axis[:,None]*new_count+new_axis[None,:]).reshape(-1)
    return old,new


def interpolate_coefficients(grid,times,rows,periods):
    """Fourier spatial projection followed by Chebyshev barycentric projection."""
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    n=grid.shape[1]
    data=torch.as_tensor(grid,device=device,dtype=torch.float64)
    modes=torch.fft.fft2(data,dim=(1,2))/(n*n)
    wave=np.fft.fftfreq(n)*n
    ex=np.exp(2j*np.pi*np.outer(rows[:,1]/periods[0],wave))
    ey=np.exp(2j*np.pi*np.outer(rows[:,2]/periods[1],wave))
    # Match separable Fourier resampling's split of each even-grid Nyquist mode.
    if n%2==0:
        ex[:,n//2]=np.cos(2*np.pi*rows[:,1]/periods[0]*wave[n//2])
        ey[:,n//2]=np.cos(2*np.pi*rows[:,2]/periods[1]*wave[n//2])
    spatial=torch.einsum("qi,tijr,qj->tqr",torch.as_tensor(ex,device=device),modes,
                         torch.as_tensor(ey,device=device))
    if len(times)==1:
        return spatial[0].real.cpu().numpy()
    bary=(-1.)**np.arange(len(times));bary[[0,-1]]*=.5
    tw=np.empty((len(rows),len(times)))
    for i,t in enumerate(rows[:,0]):
        delta=t-times;hit=np.flatnonzero(abs(delta)<1e-13)
        if len(hit):
            tw[i]=0;tw[i,hit[0]]=1
        else:
            tw[i]=bary/delta;tw[i]/=tw[i].sum()
    return torch.einsum("qt,tqr->qr",torch.as_tensor(tw,device=device,dtype=torch.complex128),spatial).real.cpu().numpy()


def compress_source(read,method,xy,velocity,interval,remaining,source_nt=17,reserve=70.):
    nxy=len(xy)
    spatial_n=int(round(np.sqrt(nxy)))
    initial_spatial_n=spatial_n
    nv=int(round(np.sqrt(len(velocity))))
    if spatial_n**2!=nxy or nv**2!=len(velocity):
        raise ValueError("tensor source mesh required")
    # Spatial grid excludes its periodic endpoint; recover the full periods.
    periods=np.max(xy,axis=0)*spatial_n/(spatial_n-1)
    pool=qmc.Sobol(3,scramble=True,seed=1729).random_base2(10)
    pool[:,0]=interval[0]+(interval[1]-interval[0])*pool[:,0]
    pool[:,1:]*=periods
    continuous_times=pool[:,0].copy()
    if TRAINING_TIME_COUNT>0:
        training_times=chebyshev_times(interval,TRAINING_TIME_COUNT)
        pool[:,0]=training_times[np.arange(len(pool))%len(training_times)]
    else:
        training_times=None
    validation_times=None
    if VALIDATION_TIME_COUNT>1:
        independent=qmc.Sobol(1,scramble=True,seed=983).random_base2(6)[:VALIDATION_TIME_COUNT,0]
        validation_times=interval[0]+(interval[1]-interval[0])*independent
        validation_times[:2]=interval
    # Explicit endpoint representatives guard source changes near t0 and t1.
    pool[:8,0]=interval[0]
    pool[8:16,0]=interval[1]
    weights=velocity_weights(velocity)
    moments=source_moments(velocity)
    cell_area=float((velocity[nv,0]-velocity[0,0])*(velocity[1,1]-velocity[0,1]))
    completed_points=0
    source_seconds=0.
    def sample_rows(rows,columns=None):
        nonlocal completed_points,source_seconds
        vv=velocity if columns is None else velocity[columns]
        ordering=np.argsort(rows[:,0],kind="stable")
        ordered_rows=rows[ordering]
        values=np.empty((len(rows),len(vv)),dtype=np.float64)
        # Bound physical-point temporaries even at129 time knots.  Full
        # velocity rows remain intact; the caller enforces its RPC limit.
        rows_per_batch=max(1,131072//len(vv))
        for first in range(0,len(rows),rows_per_batch):
            batch=ordered_rows[first:first+rows_per_batch]
            points=np.concatenate((np.repeat(batch,len(vv),axis=0),np.tile(vv,(len(batch),1))),axis=1)
            started=candidate_solve_clock()
            ordered_values=read(method,points).reshape(len(batch),len(vv))
            source_seconds+=candidate_solve_clock()-started
            completed_points+=len(points)
            values[ordering[first:first+len(batch)]]=ordered_values
        return values
    training=sample_rows(pool[:INITIAL_ROWS])
    cursor=INITIAL_ROWS
    def heldout_coordinates(first):
        rows=pool[first:first+VALIDATION_ROWS].copy()
        rows[:,0]=continuous_times[first:first+VALIDATION_ROWS]
        half=len(rows)//2
        if VALIDATION_TIME_COUNT>1:
            rows[:half,0]=validation_times[np.arange(half)%len(validation_times)]
            rows[half:,0]=validation_times[np.arange(len(rows)-half)%len(validation_times)]
        indices=np.rint(rows[:half,1:]/periods*spatial_n).astype(int)%spatial_n
        rows[:half,1:]=xy[indices[:,0]*spatial_n+indices[:,1]]
        endpoint=min(4,max(1,half//4))
        rows[:endpoint,0]=interval[0];rows[endpoint:2*endpoint,0]=interval[1]
        rows[half:half+endpoint,0]=interval[0];rows[half+endpoint:half+2*endpoint,0]=interval[1]
        return rows
    validation_rows=heldout_coordinates(cursor)
    validation=sample_rows(validation_rows)
    cursor+=VALIDATION_ROWS
    rounds=[]
    while True:
        basis,singular=fit_basis(training,weights)
        rank=len(basis)
        selected=select_columns(basis,min(len(velocity),int(np.ceil(1.5*rank))))
        base_operator=reconstruction_operator(basis,selected)
        operator=base_operator
        if remaining()>reserve+20.:
            operator,correction_stats=correct_source_moments(training,basis,selected,operator,
                                                            weights,moments,cell_area)
        else:
            correction_stats={"method":"positive_shared_quadrature","applied":False,"status":"budget"}
        metrics=validate(basis,selected,operator,validation,weights,moments,cell_area)
        metrics.update({"training_rows":len(training),"rank":rank,"selected_velocities":len(selected),
                        "moment_correction":correction_stats})
        rounds.append(metrics)
        accurate=validation_passes(metrics)
        point_cost=max(source_seconds/completed_points,1e-8)
        # Keep at least nine temporal frames affordable before expanding.
        expansion_points=(2*VALIDATION_ROWS)*len(velocity)
        future_column_points=7*nxy*min(int(np.ceil(1.5*(rank+2*VALIDATION_ROWS))),int(1.5*MAXIMUM_RANK))
        affordable=(expansion_points+future_column_points)*point_cost*1.35 < remaining()-reserve
        if accurate or len(training)>=MAXIMUM_ROWS or not affordable:
            break
        # Promote independently held-out rows, then add independent new rows.
        fresh=sample_rows(pool[cursor:cursor+VALIDATION_ROWS])
        cursor+=VALIDATION_ROWS
        training=np.concatenate((training,validation,fresh),axis=0)[:MAXIMUM_ROWS]
        validation_rows=heldout_coordinates(cursor)
        validation=sample_rows(validation_rows)
        cursor+=VALIDATION_ROWS

    velocity_stop_reason="passed" if accurate else ("row_cap" if len(training)>=MAXIMUM_ROWS else "budget")
    point_cost=max(source_seconds/completed_points,1e-8)
    affordable_points=max(0,int((remaining()-reserve)/(1.35*point_cost)))
    desired_points=max(0,SOURCE_POINT_TARGET-completed_points)
    column_budget=min(affordable_points,desired_points)
    selected_count=len(selected)
    frame_points=nxy*selected_count
    allowed=min(int(source_nt),max(1,column_budget//max(frame_points,1)))
    nt=max(n for n in (1,3,5,9,17,33,65,129) if n<=allowed)
    times=chebyshev_times(interval,nt)
    rows=np.column_stack((np.repeat(times,nxy),np.tile(xy,(nt,1))))
    sampled=sample_rows(rows,selected).reshape(nt,nxy,len(selected))
    validation_base_xy=pool[cursor-VALIDATION_ROWS:cursor,1:3].copy()
    def validation_information():
        weighted=validation*weights
        projected=weighted@basis.T
        selected_coefficients=(validation[:,selected]*weights[selected])@operator
        norm_squared=np.sum(weighted**2,axis=1)
        residual_squared=np.maximum(0.,norm_squared-np.sum(projected**2,axis=1))
        return projected,selected_coefficients,norm_squared,residual_squared,validation@moments*cell_area
    projected,selected_coefficients,true_norm_squared,residual_squared,true_moments=validation_information()
    moment_map=(basis/weights[None,:])@moments*cell_area
    half=len(validation)//2
    time_mask=np.arange(len(validation))<half
    off_grid=~time_mask
    endpoint_mask=off_grid&((abs(validation_rows[:,0]-interval[0])<1e-13)
                            |(abs(validation_rows[:,0]-interval[1])<1e-13))
    def interpolation_metrics(interpolated):
        result={}
        # At one time knot, t1 is not available; calling its error spatial-only
        # would mislabel a temporal approximation error.
        spatial_only_mask=endpoint_mask&np.any(
            np.abs(validation_rows[:,0,None]-times[None,:])<1e-13,axis=1)
        for name,mask in (("temporal",time_mask),("spatial_endpoints",spatial_only_mask),
                          ("off_grid_interpolation",off_grid)):
            reference=selected_coefficients[mask]
            result[name]=coefficient_metrics(interpolated[mask],reference,np.zeros(np.sum(mask)),
                np.sum(reference**2,axis=1),reference@moment_map,moment_map)
            result[name]["passed"]=validation_passes(result[name])
        for name,mask in (("combined",np.ones(len(validation),dtype=bool)),("off_grid_combined",off_grid)):
            result[name]=coefficient_metrics(interpolated[mask],projected[mask],residual_squared[mask],
                true_norm_squared[mask],true_moments[mask],moment_map)
            result[name]["passed"]=validation_passes(result[name])
        return result
    interpolation_rounds=[]
    spatial_refinements=[]
    spatial_knot_validation_rounds=[]
    temporal_stop_reason="passed"
    spatial_stop_reason="passed"
    while True:
        check_started=candidate_solve_clock()
        coefficients=((sampled.reshape(-1,len(selected))*weights[selected])@operator).astype(np.float32)
        coefficient_grid=coefficients.reshape(nt,spatial_n,spatial_n,len(basis))
        interpolated=interpolate_coefficients(coefficient_grid,times,validation_rows,periods)
        checks=interpolation_metrics(interpolated)
        checks["time_nodes"]=nt
        checks["spatial_nodes"]=spatial_n
        interpolation_rounds.append(checks)
        validation_seconds=candidate_solve_clock()-check_started
        point_cost=max(source_seconds/completed_points,1e-8)
        if checks["temporal"]["passed"]:
            temporal_stop_reason="passed"
        else:
            if nt>=MAXIMUM_TIME_NODES:
                temporal_stop_reason="node_cap"
            else:
                next_nt=3 if nt==1 else 2*nt-1
                additional=(next_nt-nt)*frame_points
                overhead=2*validation_seconds+5.
                if additional*point_cost*1.35+overhead>=remaining()-reserve:
                    temporal_stop_reason="budget"
                else:
                    next_times=chebyshev_times(interval,next_nt)
                    new_indices=np.array([1,2]) if nt==1 else np.arange(1,next_nt,2)
                    new_rows=np.column_stack((np.repeat(next_times[new_indices],nxy),np.tile(xy,(len(new_indices),1))))
                    new_sampled=sample_rows(new_rows,selected).reshape(len(new_indices),nxy,len(selected))
                    refined=np.empty((next_nt,nxy,len(selected)))
                    if nt==1:
                        refined[0]=sampled[0]
                    else:
                        refined[::2]=sampled
                    refined[new_indices]=new_sampled
                    sampled=refined;times=next_times;nt=next_nt
                    continue

        spatial_failure=not checks["spatial_endpoints"]["passed"]
        spatial_evidence="endpoint" if spatial_failure else None
        if not spatial_failure and not checks["off_grid_interpolation"]["passed"]:
            # The off-grid interpolant mixes space and time.  Recheck the same
            # positions at existing time knots before spending on spatial nodes.
            probe_rows=validation_rows[off_grid].copy()
            nearest=np.argmin(np.abs(probe_rows[:,0,None]-times[None,:]),axis=1)
            probe_rows[:,0]=times[nearest]
            probe_points=len(probe_rows)*selected_count
            probe_overhead=max(5.,2*validation_seconds)
            if probe_points*point_cost*1.35+probe_overhead>=remaining()-reserve:
                spatial_stop_reason="probe_budget"
                break
            probe_values=sample_rows(probe_rows,selected)
            probe_coefficients=(probe_values*weights[selected])@operator
            probe_interpolated=interpolate_coefficients(coefficient_grid,times,probe_rows,periods)
            probe_metrics=coefficient_metrics(probe_interpolated,probe_coefficients,
                np.zeros(len(probe_rows)),np.sum(probe_coefficients**2,axis=1),
                probe_coefficients@moment_map,moment_map)
            probe_metrics.update({"passed":validation_passes(probe_metrics),
                                  "time_nodes":nt,"spatial_nodes":spatial_n,
                                  "source_points":probe_points})
            spatial_knot_validation_rounds.append(probe_metrics)
            spatial_failure=not probe_metrics["passed"]
            spatial_evidence="independent_time_knots" if spatial_failure else None
        if not spatial_failure:
            spatial_stop_reason="passed"
            break
        if spatial_n>=MAXIMUM_SPATIAL_NODES:
            spatial_stop_reason="node_cap"
            break

        new_n=MAXIMUM_SPATIAL_NODES
        new_xy=spatial_grid(periods,new_n)
        old_common,new_common=coincident_grid_indices(spatial_n,new_n)
        fresh_mask=np.ones(new_n*new_n,dtype=bool)
        fresh_mask[new_common]=False
        fresh_indices=np.flatnonzero(fresh_mask)
        new_validation_rows=validation_rows.copy()
        indices=np.rint(validation_base_xy[:half]/periods*new_n).astype(int)%new_n
        new_validation_rows[:half,1:]=new_xy[indices[:,0]*new_n+indices[:,1]]
        moved=np.flatnonzero(np.any(new_validation_rows[:half]!=validation_rows[:half],axis=1))
        grid_points=nt*len(fresh_indices)*selected_count
        validation_points=len(moved)*len(velocity)
        additional=grid_points+validation_points
        point_cost=max(source_seconds/completed_points,1e-8)
        # The state grid and its integration cost stay fixed.  Preserve the
        # caller's integration reserve and allow extra host/GPU algebra for the
        # larger source grid, independently of the Oracle-point forecast.
        algebra_margin=max(15.,2*validation_seconds*(new_n/spatial_n)**2)
        forecast={"from_spatial_nodes":spatial_n,"to_spatial_nodes":new_n,
                  "time_nodes":nt,"selected_velocities":selected_count,
                  "reused_spatial_sites_per_time":len(old_common),
                  "new_spatial_sites_per_time":len(fresh_indices),
                  "new_grid_source_points":grid_points,
                  "new_validation_source_points":validation_points,
                  "additional_source_points":additional,"point_seconds":point_cost,
                  "forecast_oracle_seconds":1.5*additional*point_cost,
                  "algebra_margin_seconds":algebra_margin,"integration_reserve_seconds":reserve,
                  "remaining_seconds_before":float(remaining()),"evidence":spatial_evidence}
        if forecast["forecast_oracle_seconds"]+algebra_margin>=remaining()-reserve:
            forecast["status"]="budget"
            spatial_refinements.append(forecast)
            spatial_stop_reason="budget"
            break
        completed_before=completed_points
        new_rows=np.column_stack((np.repeat(times,len(fresh_indices)),
                                  np.tile(new_xy[fresh_indices],(nt,1))))
        new_values=sample_rows(new_rows,selected).reshape(nt,len(fresh_indices),selected_count)
        refined=np.empty((nt,new_n*new_n,selected_count))
        refined[:,new_common]=sampled[:,old_common]
        refined[:,fresh_indices]=new_values
        if len(moved):
            validation[moved]=sample_rows(new_validation_rows[moved])
        validation_rows=new_validation_rows
        metrics=validate(basis,selected,operator,validation,weights,moments,cell_area)
        metrics.update({"training_rows":len(training),"rank":rank,
                        "selected_velocities":len(selected),"after_spatial_refinement":True,
                        "moment_correction":correction_stats})
        rounds.append(metrics)
        accurate=validation_passes(metrics)
        velocity_stop_reason="passed" if accurate else "post_spatial_validation"
        projected,selected_coefficients,true_norm_squared,residual_squared,true_moments=validation_information()
        sampled=refined;xy=new_xy;nxy=len(new_xy);spatial_n=new_n
        del new_values,new_rows,refined
        frame_points=nxy*selected_count
        forecast["actual_additional_source_points"]=completed_points-completed_before
        forecast["status"]="refined"
        spatial_refinements.append(forecast)
        spatial_stop_reason="refined"
    rows=np.column_stack((np.repeat(times,nxy),np.tile(xy,(nt,1))))
    # Double-precision GPU products preserve cancellation in high raw moments;
    # chunking keeps temporary storage bounded for a large spacetime mesh.
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    basis_gpu=torch.as_tensor(basis/weights[None,:],device=device,dtype=torch.float64)
    reconstruction=np.empty((len(rows),len(velocity)),dtype=np.float32)
    with torch.no_grad():
        for first in range(0,len(rows),2048):
            batch=torch.as_tensor(coefficients[first:first+2048],device=device,dtype=torch.float64)
            reconstructed=batch@basis_gpu
            reconstruction[first:first+2048]=reconstructed.float().cpu().numpy()
    values=reconstruction.reshape(nt,spatial_n,spatial_n,nv,nv)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite compressed public source")
    # Re-extract coefficients from the actual float32 source grid, so final
    # checks include its storage conversion as well as interpolation.
    # The empirical-moment estimator is not an inverse of the basis restriction.
    # Use the original basis inverse here to avoid applying the correction twice.
    stored_coefficients=(reconstruction[:,selected].astype(np.float64)*weights[selected])@base_operator
    stored_grid=stored_coefficients.reshape(nt,spatial_n,spatial_n,len(basis))
    final_checks=interpolation_metrics(interpolate_coefficients(stored_grid,times,validation_rows,periods))
    unresolved=[]
    if not accurate:
        unresolved.append("velocity")
    for name in ("temporal","spatial_endpoints","combined","off_grid_combined","off_grid_interpolation"):
        if not final_checks[name]["passed"]:
            unresolved.append(name)
    if temporal_stop_reason!="passed" and "temporal" not in unresolved:
        unresolved.append("temporal_refinement_stopped")
    if spatial_stop_reason!="passed" and not any(name in unresolved for name in
            ("spatial_endpoints","off_grid_combined","off_grid_interpolation")):
        unresolved.append("spatial_refinement_stopped")
    stats={"training_rows":len(training),"rank":len(basis),"selected_velocities":len(selected),
           "time_nodes":nt,"source_points":completed_points,"source_seconds":source_seconds,
           "validation_rounds":rounds,"interpolation_condition":float(np.linalg.cond(basis[:,selected])),
           "moment_correction":correction_stats,
           "velocity_validation_passed":bool(accurate),
           "velocity_stop_reason":velocity_stop_reason,
           "interpolation_validation_rounds":interpolation_rounds,
           "temporal_stop_reason":temporal_stop_reason,"final_interpolation_validation":final_checks,
           "initial_source_spatial_nodes":initial_spatial_n,"source_spatial_nodes":spatial_n,
           "spatial_stop_reason":spatial_stop_reason,"spatial_refinements":spatial_refinements,
           "spatial_knot_validation_rounds":spatial_knot_validation_rounds,
           "validation_passed":not unresolved,"unresolved_validation":unresolved,
           "validation_rows_on_spatial_grid":int(np.sum(time_mask)),
           "validation_rows_off_spatial_grid":int(np.sum(off_grid)),
           "source_grid_storage_checked":True,
           "source_grid_storage_check":"selected-node coefficient round-trip after float32 source storage",
           "configured_training_time_count":TRAINING_TIME_COUNT,
           "heldout_unique_time_count":len(np.unique(validation_rows[:,0])),
           "time_grouping_performance_hypothesis_unverified":True}
    return times,values,stats
