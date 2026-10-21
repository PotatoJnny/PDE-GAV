"""Independent cubic segments as one B-spline source; no Oracle access."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline
from scipy.linalg import solve_banded
from concurrent.futures import ThreadPoolExecutor


def _stable_cubic(times, values, block_size=512):
    """Convert a stable power spline analytically to compact B-spline form."""
    count = len(times)
    interior = times[2:-2] if count >= 4 else times[1:-1]
    knots = np.r_[np.repeat(times[0], 4), interior,
                  np.repeat(times[-1], 4)]
    a, b, c = knots[1:-3], knots[2:-2], knots[3:-1]
    first = ((a - b) + (c - b)) / 3
    second = (a - b) * (c - b) / 6
    flat = np.asarray(values).reshape(count, -1)
    coefficients = np.empty((len(knots)-4, flat.shape[1]),
                            dtype=np.result_type(values, np.float64))
    for start in range(0, flat.shape[1], block_size):
        stop = min(start + block_size, flat.shape[1])
        power = CubicSpline(times, flat[:, start:stop], axis=0)
        coefficients[:, start:stop] = (power(b)
            + first[:, None] * power(b, 1)
            + second[:, None] * power(b, 2))
    return BSpline.construct_fast(knots,
        coefficients.reshape((len(knots)-4,) + values.shape[1:]),
        3, extrapolate=False)


def _direct_cubic(times, values, block_size=512, workers=8):
    """Solve not-a-knot first derivatives and form cubic blossom coefficients."""
    times = np.asarray(times, float)
    values = np.asarray(values, dtype=np.result_type(values, np.float64))
    n = len(times)
    if n < 4:
        return _stable_cubic(times, values)
    h = np.diff(times)
    system = np.zeros((3, n))
    system[1, 1:-1] = 2 * (h[:-1] + h[1:])
    system[0, 2:] = h[:-1]
    system[2, :-2] = h[1:]
    system[1, 0] = h[1]
    system[0, 1] = times[2] - times[0]
    system[1, -1] = h[-2]
    system[2, -2] = times[-1] - times[-3]
    knots = np.r_[np.repeat(times[0], 4), times[2:-2], np.repeat(times[-1], 4)]
    aa, bb, cc = knots[1:-3], knots[2:-2], knots[3:-1]
    first = ((aa - bb) + (cc - bb)) / 3
    second = (aa - bb) * (cc - bb) / 6
    flat = values.reshape(n, -1)
    coefficients = np.empty_like(flat)

    def block(start):
        stop = min(start + block_size, flat.shape[1])
        value = flat[:, start:stop]
        slopes = np.diff(value, axis=0) / h[:, None]
        rhs = np.empty_like(value)
        rhs[1:-1] = 3 * (h[1:, None] * slopes[:-1] + h[:-1, None] * slopes[1:])
        d = times[2] - times[0]
        rhs[0] = ((h[0] + 2*d)*h[1]*slopes[0] + h[0]**2*slopes[1]) / d
        d = times[-1] - times[-3]
        rhs[-1] = (h[-1]**2*slopes[-2] + (2*d + h[-1])*h[-2]*slopes[-1]) / d
        derivative = solve_banded((1, 1), system, rhs, overwrite_ab=False,
                                  overwrite_b=True, check_finite=False)
        co = np.array(value, copy=True)
        co[1] = value[0] + first[1]*derivative[0]
        co[-2] = value[-1] + first[-2]*derivative[-1]
        if n > 4:
            second_derivative = 2*(3*slopes[2:-1] - 2*derivative[2:-2]
                                    - derivative[3:-1]) / h[2:-1, None]
            co[2:-2] += (first[2:-2, None]*derivative[2:-2]
                         + second[2:-2, None]*second_derivative)
        coefficients[:, start:stop] = co

    starts = range(0, flat.shape[1], block_size)
    if flat.shape[1] > block_size and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(block, starts))
    else:
        for start in starts:
            block(start)
    return BSpline.construct_fast(knots, coefficients.reshape(values.shape),
                                  3, extrapolate=False)


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
        spline = _direct_cubic(local_times, local_values)
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
