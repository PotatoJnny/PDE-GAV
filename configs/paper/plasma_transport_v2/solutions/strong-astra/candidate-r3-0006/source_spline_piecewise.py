"""Independent cubic segments as one B-spline source; no Oracle access."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline


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
        if len(local_times)>=6:
            spline=make_interp_spline(local_times,local_values,k=5,axis=0)
        else:
            cubic=CubicSpline(local_times,local_values,axis=0)
            boundary=([(1,cubic(local_times[0],1)),(2,cubic(local_times[0],2))],
                      [(1,cubic(local_times[-1],1)),(2,cubic(local_times[-1],2))])
            spline=make_interp_spline(local_times,local_values,k=5,axis=0,bc_type=boundary)
        if number:
            knot_parts[-1]=knot_parts[-1][:-6]
        knot_parts.append(spline.t)
        coefficient_parts.append(spline.c)
    return BSpline.construct_fast(np.concatenate(knot_parts),np.concatenate(coefficient_parts,axis=0),
                                  5,extrapolate=False)
