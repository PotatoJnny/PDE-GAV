"""Independent cubic segments as one B-spline source; no Oracle access."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline


def _cubic_as_quintic(times, values, block_size=512):
    """Exactly degree-elevate the ordinary not-a-knot cubic without collocation.

    Increasing each cubic knot multiplicity by two preserves its continuity
    when the degree increases from three to five. Each new coefficient is
    the quintic blossom of a local cubic polynomial. No ill-conditioned
    interpolation solve is involved in the change of basis.
    """
    count=len(times)
    cubic_knots=(np.r_[np.repeat(times[0],4),times[2:-2],np.repeat(times[-1],4)]
                 if count>=4 else np.r_[np.repeat(times[0],4),np.repeat(times[-1],4)])
    distinct,multiplicity=np.unique(cubic_knots,return_counts=True)
    knots=np.repeat(distinct,multiplicity+2)
    number=len(knots)-6
    # The middle blossom argument lies in a compatible polynomial interval;
    # C2 continuity makes either adjacent interval valid at a repeated knot.
    span=np.clip(np.searchsorted(times,knots[3:number+3],side='right')-1,0,count-2)
    delta=np.stack([knots[j:number+j] for j in range(1,6)],axis=1)-times[span,None]
    linear=np.sum(delta,axis=1)/5
    quadratic=sum(delta[:,i]*delta[:,j] for i in range(5) for j in range(i+1,5))/10
    cubic=sum(delta[:,i]*delta[:,j]*delta[:,k]
              for i in range(5) for j in range(i+1,5) for k in range(j+1,5))/10
    shape=values.shape[1:]
    data=values.reshape(count,-1)
    result=np.empty((number,data.shape[1]),dtype=np.result_type(values.dtype,float))
    for first in range(0,data.shape[1],block_size):
        part=slice(first,first+block_size)
        pp=CubicSpline(times,data[:,part],axis=0).c[:,span]
        result[:,part]=(pp[3]+linear[:,None]*pp[2]
                        +quadratic[:,None]*pp[1]+cubic[:,None]*pp[0])
    return BSpline.construct_fast(knots,result.reshape((number,)+shape),5,extrapolate=False)


def _bounded_quintic(times,values):
    """Fit quintic when its control polygon has bounded amplification.

    This is a numerical stability guard, not a shape/positivity constraint.
    The bound applies to deviations from each channel's first sample, so
    a constant channel cannot hide a large collocation cancellation error.
    """
    if len(times)<6:
        return _cubic_as_quintic(times,values)
    try:
        spline=make_interp_spline(times,values,k=5,axis=0)
        channels=values.reshape(len(values),-1)
        coefficients=spline.c.reshape(len(spline.c),-1)
        for first in range(0,channels.shape[1],512):
            part=slice(first,first+512)
            data=channels[:,part]
            offset=data[0]
            size=np.max(abs(data),axis=0)
            variation=np.max(abs(data-offset),axis=0)
            allowance=32*np.maximum(variation,256*np.finfo(float).eps*size)
            controls=coefficients[:,part]
            if (not np.isfinite(controls).all()
                    or np.any(np.max(abs(controls-offset),axis=0)>allowance)):
                return _cubic_as_quintic(times,values)
        return spline
    except (ValueError,ArithmeticError,np.linalg.LinAlgError):
        return _cubic_as_quintic(times,values)


def make_piecewise_cubic_spline(times, values, break_indices, *, join_multiplicity=4):
    """Fit independent ordinary cubic splines separated at supplied samples.

    Multiplicity four preserves each segment independently; adjacent segments
    interpolate the same endpoint value, so the represented source is C0.
    Multiplicity three shares that endpoint coefficient and is equivalent up
    to rounding. Two/three-sample pieces preserve CubicSpline's linear/quadratic
    special cases through explicit endpoint derivative boundary conditions.
    """
    times=np.asarray(times,dtype=float)
    values=np.asarray(values)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Source times must be finite and strictly increasing')
    if len(values)!=len(times):
        raise ValueError('Source values and times have different lengths')
    if join_multiplicity not in (3,4):
        raise ValueError('Cubic segment join multiplicity must be three or four')
    breaks=np.asarray(break_indices,dtype=int)
    if breaks.ndim!=1 or np.any(breaks<0) or np.any(breaks>=len(times)):
        raise ValueError('Break indices must lie within source samples')
    bounds=np.unique(np.r_[0,breaks,len(times)-1])
    coefficient_parts=[];knot_parts=[]
    for number,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
        local_times=times[left:right+1]
        local_values=values[left:right+1]
        if len(local_times)>=4:
            spline=make_interp_spline(local_times,local_values,k=3,axis=0)
        else:
            cubic=CubicSpline(local_times,local_values,axis=0)
            boundary=([(1,cubic(local_times[0],1))],[(1,cubic(local_times[-1],1))])
            spline=make_interp_spline(local_times,local_values,k=3,axis=0,bc_type=boundary)
        if number:
            # Drop the preceding clamped terminal knot block. The next piece's
            # initial block supplies exactly the requested join multiplicity.
            knot_parts[-1]=knot_parts[-1][:-4]
            knot_parts.append(spline.t if join_multiplicity==4 else spline.t[1:])
            if join_multiplicity==3:
                coefficient_parts[-1]=coefficient_parts[-1][:-1]
        else:
            knot_parts.append(spline.t)
        coefficient_parts.append(spline.c)
    return BSpline.construct_fast(np.concatenate(knot_parts),np.concatenate(coefficient_parts,axis=0),
                                  3,extrapolate=False)


def make_piecewise_quintic_spline(times, values, break_indices):
    """Quintic interpolation within independently resolved smooth pieces."""
    times=np.asarray(times,dtype=float)
    values=np.asarray(values)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Source times must be finite and strictly increasing')
    if len(values)!=len(times):
        raise ValueError('Source values and times have different lengths')
    breaks=np.asarray(break_indices,dtype=int)
    bounds=np.unique(np.r_[0,breaks,len(times)-1])
    coefficient_parts=[];knot_parts=[]
    for number,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
        local_times=times[left:right+1]
        local_values=values[left:right+1]
        spline=_bounded_quintic(local_times,local_values)
        if number:
            knot_parts[-1]=knot_parts[-1][:-6]
        knot_parts.append(spline.t)
        coefficient_parts.append(spline.c)
    return BSpline.construct_fast(np.concatenate(knot_parts),np.concatenate(coefficient_parts,axis=0),
                                  5,extrapolate=False)
