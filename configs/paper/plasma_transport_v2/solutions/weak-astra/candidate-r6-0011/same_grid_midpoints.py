"""Research-only temporal residuals measured on one fixed coarse grid.

Both coarse histories are raw public S, sampled on identical spatial and
velocity coordinates. Therefore arbitrary stationary spatial content and its
aliases cancel without reconstructing the constant initial operator.
"""
from __future__ import annotations
import numpy as np
from green_ops import refine_source_history


def compact_low_mask(coarse_nx,passband,stopband):
    if not 0<=passband<stopband<coarse_nx/2:
        raise ValueError('require 0 <= passband < stopband < coarse Nyquist')
    frequency=np.rint(np.fft.fftfreq(coarse_nx)*coarse_nx).astype(int)
    modes=np.stack(np.meshgrid(frequency,frequency,indexing='ij'),-1).reshape(-1,2)
    radius=np.max(abs(modes),axis=1)
    phase=np.clip((radius-passband)/(stopband-passband),0,1)
    window=.5*(1+np.cos(np.pi*phase));window[radius>=stopband]=0.
    return modes,window


def retain_coarse_original_samples(original_source_values,coarse_nx):
    """Copy an exact divisor subgrid BEFORE adding any constant Lf0 defect.

    Input axes are(time,x,y,velocity). The copy prevents later in-place
    addition of Lf0 to the full source array from changing these raw samples.
    """
    values=np.asarray(original_source_values)
    if (values.ndim!=4 or values.shape[1]!=values.shape[2] or coarse_nx<1
            or values.shape[1]%coarse_nx):
        raise ValueError('coarse count must divide the original source mesh exactly')
    stride=values.shape[1]//coarse_nx
    return values[:,::stride,::stride].copy()


def refine_same_grid_midpoints(source,source_times,coarse_original_source,coarse_midpoint_source,
                              *,passband=3.,stopband=4.5,blend=1.,check=None):
    """Refine fine defect modes using same-coarse-grid raw-source residuals.

    source: (nt,fine_nx**2,nv), possibly containing constant Lf0.
    coarse_original_source: (nt,nc,nc,nv), raw S at original times.
    coarse_midpoint_source: (nt-1,nc,nc,nv), raw S at consecutive midpoints.
    Both coarse arrays must use EXACTLY the same spatial/velocity coordinates.
    Only new midpoint Fourier values change; original source modes are exact.
    """
    check=check or (lambda:None);check()
    source=np.asarray(source);times=np.asarray(source_times,float)
    original=np.asarray(coarse_original_source);observed=np.asarray(coarse_midpoint_source)
    if source.ndim!=3:raise ValueError('source requires time,mode,velocity axes')
    nt,nm,nv=source.shape;nx=int(round(np.sqrt(nm)))
    if (nx*nx!=nm or len(times)!=nt or original.ndim!=4 or original.shape[0]!=nt
            or original.shape[1]!=original.shape[2] or original.shape[-1]!=nv
            or observed.shape!=(nt-1,)+original.shape[1:] or not 0<=blend<=1):
        raise ValueError('compatible original and midpoint source arrays required')
    nc=original.shape[1]
    if nc>nx:raise ValueError('coarse mesh exceeds the fine Fourier representation')
    modes,window=compact_low_mask(nc,passband,stopband);selected=np.flatnonzero(window)
    ids=(modes[selected,0]%nx)*nx+modes[selected,1]%nx
    coarse_refined,_=refine_source_history(original.reshape(nt,nc*nc,nv),times,factor=2)
    refined,new_times=refine_source_history(source,times,factor=2)
    energy=0.;mean=np.zeros((len(selected),nv),complex)
    for frame in range(nt-1):
        check()
        residual=observed[frame]-coarse_refined[2*frame+1].reshape(nc,nc,nv)
        modes_residual=np.fft.fft2(residual,axes=(0,1)).reshape(nc*nc,nv)/(nc*nc)
        correction=blend*window[selected,None]*modes_residual[selected]
        refined[2*frame+1,ids]+=correction
        energy+=float(np.sum(abs(correction)**2));mean+=correction
    refined[::2]=source
    diagnostics={'method':'same-grid temporal residual; raw S at original and midpoint times',
        'coarse_nx':nc,'passband':float(passband),'stopband':float(stopband),'blend':float(blend),
        'corrected_spatial_modes':len(selected),'inserted_unweighted_velocity_norm':float(np.sqrt(energy)),
        'inserted_temporal_mean_energy_fraction':float(np.sum(abs(mean)**2)/max((nt-1)*energy,1e-300)),
        'limitation':'fast high-spatial temporal residuals can still alias into corrected low modes'}
    return refined,new_times,diagnostics


def optional_same_grid_refinement(source,times,coarse_original,*,source_nx,periods,
                                 relative_velocity,sample,source_rate,solve_clock,
                                 deadline,evolution_reserve,oom_errors=(),check=None):
    """Try extra same-grid observations while retaining the completed source.

    source and times are never modified. Both sampling and helper failures
    return their ordinary natural refinement. sample is the preparation's
    existing sampler, whose per-batch worst-rate/evolution-reserve checks
    remain in force. The supplied clock must be candidate_solve_clock.
    """
    check=check or (lambda:None)
    diagnostics={'status':'skipped','source_nx':int(source_nx),
        'upfront_cost_margin':1.15,'helper_reserve_seconds':10.,
        'evolution_reserve_seconds':float(evolution_reserve),
        'passband':3.,'stopband':5.}
    refined=None;new_times=None
    if source_nx<28 or source_nx%2:
        diagnostics['reason']='source mesh lacks an eligible divisor subgrid'
    elif coarse_original is None:
        diagnostics['reason']='raw original coarse source samples unavailable'
    else:
        nc=source_nx//2;nv=source.shape[-1]
        points=(len(times)-1)*nc*nc*nv
        estimated=1.15*float(source_rate())*points
        remaining=float(deadline-solve_clock())
        diagnostics.update(coarse_nx=int(nc),extra_source_points=int(points),
            estimated_acquisition_seconds=float(estimated),
            remaining_seconds_at_decision=remaining)
        if remaining<=estimated+evolution_reserve+10.:
            diagnostics['reason']='remaining shared budget cannot fund optional source and core evolution'
        else:
            diagnostics['status']='attempted';started=solve_clock()
            observed=None;completed=None;completed_times=None
            def optional_check():
                check()
                if solve_clock()+evolution_reserve+5.>=deadline:
                    raise TimeoutError('optional midpoint helper would consume core evolution reserve')
            try:
                optional_check()
                midpoint=(np.asarray(times[:-1])+np.asarray(times[1:]))*.5
                xy=np.stack(np.meshgrid(*(np.arange(nc)*p/nc for p in periods),indexing='ij'),-1).reshape(-1,2)
                sites=np.column_stack((np.repeat(midpoint,nc*nc),np.tile(xy,(len(midpoint),1))))
                observed=sample(sites,relative_velocity,'source').reshape(len(midpoint),nc,nc,nv)
                optional_check()
                completed,completed_times,detail=refine_same_grid_midpoints(source,times,coarse_original,
                    observed,passband=3.,stopband=5.,check=optional_check)
                optional_check()
                refined,new_times=completed,completed_times
                diagnostics.update(status='applied',correction=detail)
            except (TimeoutError,ValueError,FloatingPointError,MemoryError)+tuple(oom_errors) as error:
                diagnostics.update(status='fallback',reason=type(error).__name__+': '+str(error))
                # Release private optional allocations before allocating the
                # ordinary fallback, especially after a late helper failure.
                observed=None;completed=None;completed_times=None
            diagnostics['optional_elapsed_seconds']=float(solve_clock()-started)
    if refined is None:
        refined,new_times=refine_source_history(source,times,factor=2)
    return refined,new_times,diagnostics
