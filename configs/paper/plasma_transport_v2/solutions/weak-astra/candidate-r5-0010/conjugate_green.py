"""Exploit exact opposite Fourier wavevectors for a real source history.

Unpaired modes (including a one-sided Nyquist frequency) are evaluated directly.
Any failed source Hermitian check selects the complete original computation.
No averaging, truncation, or modification of source coefficients is performed.
"""
from __future__ import annotations
import numpy as np
from green_ops import green_response as _full_green_response


def hermitian_plan(source,wavevectors,*,relative_tolerance=1e-12,check=None):
    """Return representatives and conjugate destinations, or None on failure.

    Opposite waves must be exactly representable negatives, as produced by an
    FFT frequency axis. Approximate/duplicate keys are deliberately not merged.
    A single scale is used for roundoff checks across the source history.
    """
    source=np.asarray(source);waves=np.asarray(wavevectors,float)
    if source.ndim!=3 or waves.shape!=(source.shape[1],2):return None
    if not np.isfinite(waves).all():return None
    check=check or (lambda:None)
    lookup={}
    for index,wave in enumerate(waves):
        key=tuple(wave)
        if key in lookup:return None
        lookup[key]=index
    scale=0.
    for index in range(source.shape[1]):
        if index%32==0:check()
        scale=max(scale,float(np.max(np.abs(source[:,index]),initial=0.)))
    if not np.isfinite(scale):return None
    tolerance=relative_tolerance*scale
    representatives=[];destinations=[];origins=[];visited=set()
    for index,wave in enumerate(waves):
        if index%32==0:check()
        if index in visited:continue
        partner=lookup.get(tuple(-wave))
        slot=len(representatives)
        representatives.append(index);visited.add(index)
        if partner is None:continue
        difference=source[:,partner]-source[:,index].conjugate()
        if not np.isfinite(difference).all() or np.max(np.abs(difference),initial=0.)>tolerance:
            return None
        if partner!=index:
            visited.add(partner);destinations.append(partner);origins.append(slot)
    if not destinations:return None
    return tuple(np.asarray(item,dtype=int) for item in (representatives,destinations,origins))


def green_response(source_modes,source_times,velocity,integration,wavevectors,
                   bulk_center,nu,theta,*,density_only=False,mode_block=64,
                   quadrature_order=4,check=None,kernel_model=None,max_workers=None):
    """Evaluate one side of a verified Hermitian spectrum and reconstruct it."""
    check=check or (lambda:None);check()
    source=np.asarray(source_modes);waves=np.asarray(wavevectors,float)
    arguments=dict(source_times=source_times,velocity=velocity,integration=integration,
                   bulk_center=bulk_center,nu=nu,theta=theta,density_only=density_only,
                   mode_block=mode_block,quadrature_order=quadrature_order,check=check,
                   kernel_model=kernel_model,max_workers=max_workers)
    plan=hermitian_plan(source,waves,check=check)
    if plan is None:
        return _full_green_response(source_modes=source,wavevectors=waves,**arguments)
    representatives,destinations,origins=plan
    solved=_full_green_response(source_modes=source[:,representatives],
                                wavevectors=waves[representatives],**arguments)
    check()
    result=np.empty(solved.shape[:2]+(source.shape[1],),dtype=solved.dtype)
    result[:,:,representatives]=solved
    result[:,:,destinations]=solved[:,:,origins].conjugate()
    check();return result
