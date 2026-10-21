"""Piecewise cubic source via stable power-basis construction and conversion.

The not-a-knot B-spline coefficient formula is the cubic polar form of each
local polynomial, equivalent to the Marsden-identity conversion used by
SciPy's BSpline.from_power_basis, extended to vector/complex values. Unlike a
fresh collocation solve, it preserves constant data on highly unequal gaps.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline,CubicSpline


def _compact_cubic(times,values,channel_block=512,out=None):
    count=len(times);coefficient_count=max(4,count)
    dtype=np.complex128 if np.iscomplexobj(values) else np.float64
    shape=(coefficient_count,)+values.shape[1:]
    coefficients=np.empty(shape,dtype=dtype) if out is None else out
    if coefficients.shape!=shape or channel_block<1:raise ValueError('invalid coefficient buffer or block size')
    output_flat=coefficients.reshape(coefficient_count,-1)
    flat_values=values.reshape(count,-1) if values.flags.c_contiguous else None
    if count<4:
        knots=np.r_[np.repeat(times[0],4),np.repeat(times[-1],4)]
    else:
        knots=np.r_[np.repeat(times[0],4),times[2:-2],np.repeat(times[-1],4)]
        j=np.arange(count);piece=np.minimum(j,count-2)
        delta=knots[j[:,None]+np.arange(1,4)[None,:]]-times[piece,None]
        linear=delta.sum(axis=1)[:,None]/3
        quadratic=(delta[:,0]*delta[:,1]+delta[:,0]*delta[:,2]+delta[:,1]*delta[:,2])[:,None]/3
        cubic_scale=np.prod(delta,axis=1)[:,None]
    # Construct only a bounded channel block of four power coefficients.
    # The final compact coefficient buffer is allocated once by the caller.
    for first in range(0,output_flat.shape[1],channel_block):
        last=min(first+channel_block,output_flat.shape[1])
        if flat_values is None:
            if values.ndim==1:block=values[:,None]
            else:
                channels=np.unravel_index(np.arange(first,last),values.shape[1:])
                block=values[(slice(None),)+channels]
        else:block=flat_values[:,first:last]
        cubic=CubicSpline(times,block,axis=0)
        target=output_flat[:,first:last]
        if count<4:
            span=times[-1]-times[0]
            target[0]=block[0];target[1]=block[0]+span*cubic(times[0],1)/3
            target[2]=block[-1]-span*cubic(times[-1],1)/3;target[3]=block[-1]
        else:
            target[:]=cubic.c[3,piece]
            target+=linear*cubic.c[2,piece]
            target+=quadratic*cubic.c[1,piece]
            target+=cubic_scale*cubic.c[0,piece]
    return BSpline.construct_fast(knots,coefficients,3,extrapolate=False)


def make_piecewise_cubic_spline(times,values,break_indices,*,join_multiplicity=4):
    times=np.asarray(times,dtype=float);values=np.asarray(values)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Source times must be finite and strictly increasing')
    if len(values)!=len(times):raise ValueError('Source values and times have different lengths')
    if join_multiplicity not in (3,4):raise ValueError('Cubic segment join multiplicity must be three or four')
    breaks=np.asarray(break_indices,dtype=int)
    if breaks.ndim!=1 or np.any(breaks<0) or np.any(breaks>=len(times)):raise ValueError('Break indices must lie within source samples')
    bounds=np.unique(np.r_[0,breaks,len(times)-1]);knot_parts=[]
    segment_counts=np.maximum(4,np.diff(bounds)+1)
    coefficient_count=int(np.sum(segment_counts)-(len(segment_counts)-1)*(join_multiplicity==3))
    coefficients=np.empty((coefficient_count,)+values.shape[1:],dtype=np.complex128 if np.iscomplexobj(values) else np.float64)
    offset=0
    for number,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
        if number and join_multiplicity==3:offset-=1
        size=int(segment_counts[number])
        spline=_compact_cubic(times[left:right+1],values[left:right+1],out=coefficients[offset:offset+size])
        offset+=size
        if number:
            knot_parts[-1]=knot_parts[-1][:-4]
            knot_parts.append(spline.t if join_multiplicity==4 else spline.t[1:])
        else:knot_parts.append(spline.t)
    return BSpline.construct_fast(np.concatenate(knot_parts),coefficients,3,extrapolate=False)
