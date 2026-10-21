"""Midpoint residuals on a fixed checkerboard source sublattice.

An n-by-n periodic source grid contains even/even and odd/odd n/2 grids.
Their phase-aligned average cancels axial half-grid aliases. The same sampled
sublattice is used at old and new times, preserving stationary cancellation.
"""
from __future__ import annotations
import numpy as np
from green_ops import refine_source_history
from same_grid_midpoints import refine_same_grid_midpoints


def retain_checkerboard_original_samples(original_source_values):
    values=np.asarray(original_source_values)
    if values.ndim!=4 or values.shape[1]!=values.shape[2] or values.shape[1]%2:
        raise ValueError('square even source grid required')
    return np.stack((values[:,::2,::2],values[:,1::2,1::2]),axis=1)


def checkerboard_sites(midpoints,source_nx,periods):
    n=int(source_nx);nc=n//2
    if n%2 or n<4:raise ValueError('even fine grid required')
    xy=np.stack([np.stack(np.meshgrid(*((2*np.arange(nc)+parity)*p/n for p in periods),indexing='ij'),-1)
                 for parity in (0,1)])
    xy=xy.reshape(-1,2)
    return np.column_stack((np.repeat(midpoints,len(xy)),np.tile(xy,(len(midpoints),1))))


def checkerboard_low_modes(values,modes):
    """Values (...,2,nc,nc,nv), output (...,nm,nv)."""
    values=np.asarray(values);modes=np.asarray(modes,int);nc=values.shape[-2]
    if values.shape[-4]!=2 or values.shape[-3]!=nc or modes.ndim!=2 or modes.shape[1]!=2:
        raise ValueError('matching checkerboard and Fourier modes required')
    transforms=np.fft.fft2(values,axes=(-3,-2))/(nc*nc)
    first=transforms[...,0,modes[:,0]%nc,modes[:,1]%nc,:]
    second=transforms[...,1,modes[:,0]%nc,modes[:,1]%nc,:]
    phase=np.exp(-2j*np.pi*modes.sum(axis=1)/(2*nc))
    return .5*(first+second*phase[:,None])


def refine_checkerboard_midpoints(source,source_times,original_samples,midpoint_samples,
                                  *,passband=3.,stopband=5.,blend=1.,check=None):
    return _checkerboard_midpoint_update(source,source_times,original_samples,midpoint_samples,
                                        passband=passband,stopband=stopband,blend=blend,check=check)


def _checkerboard_midpoint_update(source,source_times,original_samples,midpoint_samples,
                                 *,passband=3.,stopband=5.,blend=1.,check=None,prepared_refinement=None):
    check=check or (lambda:None);check()
    source=np.asarray(source);times=np.asarray(source_times,float)
    original=np.asarray(original_samples);observed=np.asarray(midpoint_samples)
    nt,nm,nv=source.shape;nx=int(round(np.sqrt(nm)))
    if (nx*nx!=nm or original.ndim!=5 or original.shape[0]!=nt or original.shape[1]!=2
            or original.shape[2]!=original.shape[3] or original.shape[-1]!=nv
            or observed.shape!=(nt-1,)+original.shape[1:] or not 0<=blend<=1):
        raise ValueError('compatible original and midpoint source arrays required')
    nc=original.shape[2]
    if not 0<=passband<stopband<=nc/2:raise ValueError('compact low modes at or below coarse Nyquist required')
    frequencies=np.rint(np.fft.fftfreq(nc)*nc).astype(int)
    all_modes=np.stack(np.meshgrid(frequencies,frequencies,indexing='ij'),-1).reshape(-1,2)
    radius=np.max(abs(all_modes),axis=1)
    phase=np.clip((radius-passband)/(stopband-passband),0,1)
    windows=.5*(1+np.cos(np.pi*phase));windows[radius>=stopband]=0.
    selected=np.flatnonzero(windows);modes=all_modes[selected]
    ids=(modes[:,0]%nx)*nx+modes[:,1]%nx
    # Fourier projection first is equivalent and saves two large cubic tensors.
    original_modes=checkerboard_low_modes(original,modes)
    coarse_refined,_=refine_source_history(original_modes,times,factor=2)
    compact,new_times=refine_source_history(source[:,ids],times,factor=2)
    energy=0.;mean=np.zeros((len(selected),nv),complex)
    for frame in range(nt-1):
        check()
        residual=checkerboard_low_modes(observed[frame],modes)-coarse_refined[2*frame+1]
        correction=blend*windows[selected,None]*residual
        compact[2*frame+1]+=correction
        energy+=float(np.sum(abs(correction)**2));mean+=correction
    diagnostics={'method':'fixed checkerboard temporal residual; raw source at original and midpoint times',
        'coarse_nx':nc,'source_nx':2*nc,'spatial_fraction':.5,
        'passband':float(passband),'stopband':float(stopband),'blend':float(blend),
        'corrected_spatial_modes':len(selected),'inserted_unweighted_velocity_norm':float(np.sqrt(energy)),
        'inserted_temporal_mean_energy_fraction':float(np.sum(abs(mean)**2)/max((nt-1)*energy,1e-300)),
        'limitation':'diagonal half-grid aliases and full-grid aliases remain unresolved'}
    check()
    if prepared_refinement is None:
        refined,new_times=refine_source_history(source,times,factor=2)
    else:
        refined=prepared_refinement
        if refined.shape!=((nt-1)*2+1,)+source.shape[1:]:raise ValueError('compatible prepared refinement required')
    check()
    # All fallible computations and checks precede this bounded array copy.
    # A caller-supplied completed quarter-grid state stays intact on failure.
    refined[1::2,ids]=compact[1::2]
    return refined,new_times,diagnostics


def optional_checkerboard_refinement(source,times,coarse_original,*,source_nx,periods,
                                    relative_velocity,sample,source_rate,solve_clock,
                                    deadline,evolution_reserve,oom_errors=(),check=None,
                                    passband=3.,stopband=5.,
                                    quarter_passband=3.,quarter_stopband=5.,acquired_velocity_count=None):
    """Acquire quarter-grid midpoints, then a second grid when affordable.

    `coarse_original` contains the two parity grids produced by
    retain_checkerboard_original_samples. A legacy four-axis quarter-grid
    array is accepted and uses the original quarter-grid strategy. Every
    completed quarter-grid acquisition remains usable if the extra odd/odd
    acquisition cannot finish; neither source nor times is modified.
    """
    check=check or (lambda:None)
    # Keep a half-mode guard below the divisor-grid Nyquist. This permits
    # source20/24 refinements while preserving the existing source28 mask.
    stop_cap=float(source_nx)/4.-.5
    actual_stopband=min(float(stopband),stop_cap)
    actual_passband=min(float(passband),actual_stopband-.5)
    actual_quarter_stopband=min(float(quarter_stopband),stop_cap)
    actual_quarter_passband=min(float(quarter_passband),actual_quarter_stopband-.5)
    diagnostics={'status':'skipped','source_nx':int(source_nx),
        'upfront_cost_margin':1.15,'helper_reserve_seconds':10.,
        'evolution_reserve_seconds':float(evolution_reserve),
        'passband':actual_passband,'stopband':actual_stopband,
        'requested_passband':float(passband),'requested_stopband':float(stopband),
        'quarter_passband':actual_quarter_passband,'quarter_stopband':actual_quarter_stopband}
    recoverable=(TimeoutError,ValueError,FloatingPointError,MemoryError)+tuple(oom_errors)
    refined=None;new_times=None;even_observed=None;odd_observed=None
    original=None if coarse_original is None else np.asarray(coarse_original)
    eligible=(source_nx>=20 and source_nx%2==0 and original is not None)
    if not eligible:
        diagnostics['reason']='raw source samples on an eligible divisor subgrid unavailable'
    else:
        nc=source_nx//2;nv=source.shape[-1]
        checker_available=(original.ndim==5 and original.shape[1]==2)
        even_original=original[:,0] if checker_available else original
        acquired_nv=nv if acquired_velocity_count is None else int(acquired_velocity_count)
        if acquired_nv<1 or acquired_nv>nv or (acquired_velocity_count is not None and acquired_nv!=acquired_velocity_count):
            raise ValueError('acquired velocity count must be an integer in the tensor rule')
        points=(len(times)-1)*nc*nc*acquired_nv
        def estimated():return 1.15*float(source_rate())*points
        def can_acquire():return float(deadline-solve_clock())>estimated()+evolution_reserve+10.
        def optional_check():
            check()
            if solve_clock()+evolution_reserve+5.>=deadline:
                raise TimeoutError('optional checkerboard helper would consume core evolution reserve')
        def observe(parity):
            optional_check()
            midpoints=(np.asarray(times[:-1])+np.asarray(times[1:]))*.5
            xy=np.stack(np.meshgrid(*((2*np.arange(nc)+parity)*p/source_nx for p in periods),indexing='ij'),-1).reshape(-1,2)
            sites=np.column_stack((np.repeat(midpoints,nc*nc),np.tile(xy,(len(midpoints),1))))
            result=sample(sites,relative_velocity,'source').reshape(len(midpoints),nc,nc,nv)
            optional_check()
            return result
        diagnostics.update(coarse_nx=int(nc),quarter_source_points=int(points),
            acquired_velocity_count=acquired_nv,tensor_velocity_count=nv,
            estimated_quarter_acquisition_seconds=float(estimated()),
            remaining_seconds_at_decision=float(deadline-solve_clock()),
            checker_original_available=bool(checker_available))
        if not can_acquire():
            diagnostics['reason']='remaining shared budget cannot fund quarter-grid source and core evolution'
        else:
            diagnostics['status']='attempted';started=solve_clock()
            try:
                even_observed=observe(0)
                diagnostics['quarter_acquisition']='completed'
                diagnostics['completed_extra_source_points']=int(points)
                refined,new_times,detail=refine_same_grid_midpoints(source,times,even_original,
                    even_observed,passband=actual_quarter_passband,stopband=actual_quarter_stopband,check=optional_check)
                optional_check()
                diagnostics.update(status='applied',strategy='quarter_grid',correction=detail)
                if checker_available and can_acquire():
                    odd_started=solve_clock()
                    diagnostics['odd_acquisition']='attempted'
                    try:
                        odd_observed=observe(1)
                        diagnostics['odd_acquisition']='completed'
                        diagnostics['completed_extra_source_points']=int(2*points)
                    except recoverable as error:
                        diagnostics['odd_acquisition']='fallback'
                        diagnostics['odd_reason']=type(error).__name__+': '+str(error)
                    diagnostics['odd_elapsed_seconds']=float(solve_clock()-odd_started)
                else:
                    diagnostics['odd_acquisition']='skipped'
                    diagnostics['odd_reason']='extra grid unavailable or remaining shared budget insufficient'
                if odd_observed is not None:
                    try:
                        observed=np.stack((even_observed,odd_observed),axis=1)
                        refined,new_times,detail=_checkerboard_midpoint_update(source,times,original,
                            observed,passband=actual_passband,stopband=actual_stopband,check=optional_check,
                            prepared_refinement=refined)
                        diagnostics.update(status='applied',strategy='checkerboard',correction=detail)
                    except recoverable as error:
                        diagnostics['checker_helper_reason']=type(error).__name__+': '+str(error)
                    finally:
                        odd_observed=None
                        if 'observed' in locals():del observed
            except recoverable as error:
                if refined is None:
                    new_times=None
                    diagnostics.update(status='fallback',reason=type(error).__name__+': '+str(error))
                else:
                    diagnostics['preserved_quarter_reason']=type(error).__name__+': '+str(error)
            diagnostics['optional_elapsed_seconds']=float(solve_clock()-started)
    if refined is None:
        refined,new_times=refine_source_history(source,times,factor=2)
    return refined,new_times,diagnostics
