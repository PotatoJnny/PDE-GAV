"""Independent cubic segments as one B-spline source; no Oracle access."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline


def _cubic_as_quintic(times, values, block_size=512):
    """Elevate a stable ordinary cubic exactly, without collocation.

    An interior cubic knot gains multiplicity two under degree elevation.
    The degree-five blossom of each cubic polynomial supplies its control
    coefficients. Data channels are blocked to bound auxiliary memory.
    """
    count = len(times)
    data = values.reshape(count, -1)
    inner = times[2:-2] if count >= 4 else np.empty(0)
    knots = np.r_[np.repeat(times[0], 6), np.repeat(inner, 3),
                  np.repeat(times[-1], 6)]
    coefficient_count = len(knots) - 6
    spans = np.clip(np.searchsorted(times, knots[np.arange(coefficient_count)+3],
                                   side='right')-1, 0, count-2)
    offsets = np.stack([knots[j:j+coefficient_count] for j in range(1, 6)], axis=1)
    offsets = offsets-times[spans, None]
    linear = offsets.sum(axis=1)/5
    quadratic = sum(offsets[:, i]*offsets[:, j]
                    for i in range(5) for j in range(i+1, 5))/10
    cubic = sum(offsets[:, i]*offsets[:, j]*offsets[:, k]
                for i in range(5) for j in range(i+1, 5)
                for k in range(j+1, 5))/10
    result = np.empty((coefficient_count, data.shape[1]),
                      dtype=np.result_type(values.dtype, float))
    for first in range(0, data.shape[1], block_size):
        part = slice(first, first+block_size)
        pp = CubicSpline(times, data[:, part], axis=0).c[:, spans]
        result[:, part] = (pp[3]+linear[:, None]*pp[2]
                          +quadratic[:, None]*pp[1]+cubic[:, None]*pp[0])
    return BSpline.construct_fast(knots, result.reshape((coefficient_count,)+values.shape[1:]),
                                  5, extrapolate=False)


def _bounded_control_polygon(spline, values, factor=64., block_size=512):
    """Reject amplification per channel, including complex Fourier data."""
    data = values.reshape(len(values), -1)
    control = spline.c.reshape(len(spline.c), -1)
    for first in range(0, data.shape[1], block_size):
        part = slice(first, first+block_size)
        scale = np.max(abs(data[:, part]), axis=0)
        magnitude = np.max(abs(control[:, part]), axis=0)
        if not np.isfinite(magnitude).all() or np.any(magnitude > factor*scale):
            return False
    return True


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
    """Quintics on ordinary pieces; exact cubic elevation on unsafe pieces.

    Short/strongly clustered collocation problems can amplify even constant
    data by many orders of magnitude. These pieces use the stable cubic in
    degree-five representation; well-conditioned ordinary paths are unchanged.
    """
    times=np.asarray(times,dtype=float)
    values=np.asarray(values)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Source times must be finite and strictly increasing')
    if values.ndim<1 or len(values)!=len(times):
        raise ValueError('Source values and times have different lengths')
    breaks=np.asarray(break_indices,dtype=int)
    if breaks.ndim!=1 or np.any(breaks<0) or np.any(breaks>=len(times)):
        raise ValueError('Break indices must lie within source samples')
    bounds=np.unique(np.r_[0,breaks,len(times)-1])
    coefficient_parts=[];knot_parts=[]
    for number,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
        local_times=times[left:right+1]
        local_values=values[left:right+1]
        widths=np.diff(local_times)
        safe_geometry=(len(local_times)>=6 and np.min(widths)>=1.e-4*np.max(widths))
        if safe_geometry:
            try:
                spline=make_interp_spline(local_times,local_values,k=5,axis=0)
                if not _bounded_control_polygon(spline,local_values):
                    spline=_cubic_as_quintic(local_times,local_values)
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                spline=_cubic_as_quintic(local_times,local_values)
        else:
            spline=_cubic_as_quintic(local_times,local_values)
        if number:
            knot_parts[-1]=knot_parts[-1][:-6]
        knot_parts.append(spline.t)
        coefficient_parts.append(spline.c)
    return BSpline.construct_fast(np.concatenate(knot_parts),np.concatenate(coefficient_parts,axis=0),
                                  5,extrapolate=False)
