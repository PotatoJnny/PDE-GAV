"""Per-interval/channel guard for cubics on severely graded time grids.

Ordinary stable cubic coefficients are unchanged unless a channel's control
polygon exceeds its sampled amplitude by the stated bound. Only offending
channels receive shape-preserving interpolation. Other channels retain the
original represented cubic, converted algebraically to Bernstein pieces.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator


def _shape_derivatives(times, data):
    if not np.iscomplexobj(data):
        return PchipInterpolator(times, data, axis=0)(times, 1)
    result = np.empty_like(data)
    real_columns = np.all(data.imag == 0, axis=0)
    if np.any(real_columns):
        result[:, real_columns] = PchipInterpolator(times, data[:, real_columns].real, axis=0)(times, 1)
    if np.all(real_columns):
        return result
    complex_columns = ~real_columns
    data = data[:, complex_columns]
    # Fix a channel's complex orientation before the nonlinear real PCHIP
    # operation. A global Fourier phase then rotates the result by that same
    # phase. Selecting the first nearly maximal entry makes magnitude ties
    # insensitive to a few floating-point bits under complex rotation.
    magnitude = abs(data)
    maximum = magnitude.max(axis=0)
    index = np.argmax(magnitude >= maximum[None, :]*(1.-64*np.finfo(float).eps), axis=0)
    anchor = data[index, np.arange(data.shape[1])]
    phase = np.divide(anchor, abs(anchor), out=np.ones_like(anchor), where=abs(anchor)>0)
    oriented = data*phase.conj()
    derivative = (PchipInterpolator(times, oriented.real, axis=0)(times, 1)
                  +1j*PchipInterpolator(times, oriented.imag, axis=0)(times, 1))
    result[:, complex_columns] = derivative*phase
    return result


def _interval_control_mask(cubic, times, data):
    data = np.asarray(data).reshape(len(times), -1)
    width = np.diff(times)[:, None]
    pp = cubic.c.reshape(4, len(times)-1, -1)
    first = data[:-1]+width*pp[2]/3
    last_derivative = (3*pp[0]*width+2*pp[1])*width+pp[2]
    second = data[1:]-width*last_derivative/3
    allowed = (64.*np.maximum(abs(data[:-1]), abs(data[1:]))
               +1.e-12*np.max(abs(data), axis=0)+1.e-12)
    return (~np.isfinite(first) | ~np.isfinite(second)
            | (abs(first)>allowed) | (abs(second)>allowed))


def _cubic_block_coefficients(times, values, block_size=512, *, inspect_controls=False):
    # This is the incumbent's stable construction, retained verbatim so that
    # a segment with no rejected channels has identical knots/coefficients.
    n = len(times)
    shape = values.shape[1:]
    data = values.reshape(n, -1)
    rejected = np.zeros((n-1, data.shape[1]), dtype=bool) if inspect_controls else None
    if n == 2:
        h = times[1]-times[0]
        # CubicSpline promotes input before forming differences. Preserve
        # that behavior for float32/complex64 and integer research inputs.
        linear_data=np.asarray(data,dtype=np.result_type(values.dtype,float))
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            slope=(linear_data[1]-linear_data[0])/h
        # Extreme spans or slopes can overflow nominally zero curvature or
        # endpoint-evaluation powers. Preserve that legacy behavior,
        # including exceptions, by using the original path in those cases.
        ordinary=(np.finfo(float).tiny<=h<=np.sqrt(np.finfo(float).max)/4
                  and linear_data.dtype in (np.dtype(float),np.dtype(complex))
                  and np.isfinite(slope).all()
                  and np.all(abs(slope)<=np.finfo(float).max/4))
        if ordinary:
            knots=np.r_[np.repeat(times[0],4),np.repeat(times[1],4)]
            result=np.empty((4,data.shape[1]),dtype=linear_data.dtype)
            result[0]=data[0]
            result[1]=linear_data[0]+(h/3)*slope
            result[2]=linear_data[1]-(h/3)*slope
            result[3]=data[1]
            if inspect_controls:
                # The guard deliberately has its own legacy multiply/divide
                # order, distinct from the Bernstein coefficient formula.
                first=data[0]+h*slope/3
                second=data[1]-h*slope/3
                allowed=(64.*np.maximum(abs(data[0]),abs(data[1]))
                         +1.e-12*np.max(abs(data),axis=0)+1.e-12)
                rejected[0]=(~np.isfinite(first)|~np.isfinite(second)
                             |(abs(first)>allowed)|(abs(second)>allowed))
            answer=(knots,result.reshape((4,)+shape))
            return answer+(rejected,) if inspect_controls else answer
    if n == 3 and block_size == 512:
        block_size = 4096
    def inspect(cubic, part):
        if rejected is None:
            return
        rejected[:, part] = _interval_control_mask(cubic, times, data[:, part])
    if n < 4:
        knots = np.r_[np.repeat(times[0], 4), np.repeat(times[-1], 4)]
        result = np.empty((4, data.shape[1]), dtype=np.result_type(values.dtype, float))
        for first in range(0, data.shape[1], block_size):
            part = slice(first, first + block_size)
            cubic = CubicSpline(times, data[:, part], axis=0)
            inspect(cubic, part)
            h = times[-1] - times[0]
            result[0, part] = data[0, part]
            result[1, part] = data[0, part] + (h / 3) * cubic(times[0], 1)
            result[2, part] = data[-1, part] - (h / 3) * cubic(times[-1], 1)
            result[3, part] = data[-1, part]
    else:
        knots = np.r_[np.repeat(times[0], 4), times[2:-2], np.repeat(times[-1], 4)]
        result = np.empty((n, data.shape[1]), dtype=np.result_type(values.dtype, float))
        spans = np.minimum(np.arange(n), n - 2)
        d = np.stack([knots[1:n+1], knots[2:n+2], knots[3:n+3]], axis=1) - times[spans, None]
        linear = d.sum(axis=1) / 3
        quadratic = (d[:, 0]*d[:, 1] + d[:, 0]*d[:, 2] + d[:, 1]*d[:, 2]) / 3
        cubic_term = d.prod(axis=1)
        for first in range(0, data.shape[1], block_size):
            part = slice(first, first + block_size)
            cubic = CubicSpline(times, data[:, part], axis=0)
            inspect(cubic, part)
            pp = cubic.c[:, spans]
            result[:, part] = (pp[3] + linear[:, None]*pp[2]
                              + quadratic[:, None]*pp[1] + cubic_term[:, None]*pp[0])
    answer = (knots, result.reshape((len(result),) + shape))
    return answer+(rejected,) if inspect_controls else answer


def _insertion_rows(original_knots, endpoints):
    """Sparse scalar Boehm refinement, without copying coefficient fields."""
    knots = list(np.asarray(original_knots, float))
    rows = [{i: 1.} for i in range(len(knots)-4)]
    for value in np.unique(endpoints):
        multiplicity = knots.count(float(value))
        for _ in range(4-multiplicity):
            slot = int(np.searchsorted(knots, value, side='right')-1)
            repeated = knots.count(float(value))
            blended = []
            for i in range(slot-2, slot-repeated+1):
                alpha = (value-knots[i])/(knots[i+3]-knots[i])
                row = {}
                for old, weight in rows[i-1].items():
                    contribution = (1.-alpha)*weight
                    if contribution:
                        row[old] = contribution
                for old, weight in rows[i].items():
                    contribution = alpha*weight
                    if contribution:
                        row[old] = row.get(old, 0.)+contribution
                blended.append(row)
            rows = rows[:slot-2]+blended+rows[slot-repeated:]
            knots.insert(slot+1, float(value))
    return np.asarray(knots), rows


def _bounded_segment(times, values, *, block_size=512, fallback='pchip'):
    knots, coefficients, rejected = _cubic_block_coefficients(
        times, values, block_size, inspect_controls=True)
    data = values.reshape(len(times), -1)
    flat = coefficients.reshape(len(coefficients), -1)
    if not np.any(rejected):
        return knots, coefficients, rejected

    intervals = np.flatnonzero(np.any(rejected, axis=1))
    endpoints = times[np.unique(np.r_[intervals, intervals+1])]
    refined_knots, rows = _insertion_rows(knots, endpoints)
    output = np.empty((len(rows), data.shape[1]), dtype=flat.dtype)
    for i, row in enumerate(rows):
        entries = list(row.items())
        source, weight = entries[0]
        if len(entries) == 1 and weight == 1.:
            output[i] = flat[source]
        else:
            output[i] = weight*flat[source]
            for source, weight in entries[1:]:
                output[i] += weight*flat[source]

    # Only flagged intervals have clamped four-control Bernstein pieces.
    # Other knots and all unaffected coefficient rows remain untouched.
    flagged_channels = np.flatnonzero(np.any(rejected, axis=0))
    for first in range(0, len(flagged_channels), block_size):
        channels = flagged_channels[first:first+block_size]
        local_data = data[:, channels]
        derivative = (_shape_derivatives(times, local_data) if fallback=='pchip' else None)
        for interval in intervals:
            local = np.flatnonzero(rejected[interval, channels])
            if not len(local):
                continue
            selected = channels[local]
            y0, y1 = local_data[interval, local], local_data[interval+1, local]
            if fallback == 'linear':
                b1, b2 = (2*y0+y1)/3, (y0+2*y1)/3
            else:
                width = times[interval+1]-times[interval]
                b1 = y0+width*derivative[interval, local]/3
                b2 = y1-width*derivative[interval+1, local]/3
            offset = int(np.searchsorted(refined_knots, times[interval], side='right')-4)
            output[offset, selected] = y0
            output[offset+1, selected] = b1
            output[offset+2, selected] = b2
            output[offset+3, selected] = y1
    return refined_knots, output.reshape((len(output),)+values.shape[1:]), rejected


def make_piecewise_cubic_spline(times, values, break_indices, *, join_multiplicity=4,
                                fallback='pchip'):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Source times must be finite and strictly increasing')
    if values.ndim < 1 or len(values) != len(times) or not np.isfinite(values).all():
        raise ValueError('Source values must be finite and agree with the times')
    if join_multiplicity not in (3, 4):
        raise ValueError('Cubic segment join multiplicity must be three or four')
    if fallback not in ('pchip', 'linear'):
        raise ValueError('Cubic guard fallback must be pchip or linear')
    breaks = np.asarray(break_indices, dtype=int)
    if breaks.ndim != 1 or np.any(breaks < 0) or np.any(breaks >= len(times)):
        raise ValueError('Break indices must lie within source samples')
    bounds = np.unique(np.r_[0, breaks, len(times)-1])
    coefficient_parts, knot_parts = [], []
    diagnostics = []
    for number, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        knots, coefficients, rejected = _bounded_segment(times[left:right+1], values[left:right+1],
                                                        fallback=fallback)
        diagnostics.append(dict(left=int(left), right=int(right),
            rejected_channels=int(np.any(rejected, axis=0).sum()),
            rejected_interval_channels=int(rejected.sum()), channels=int(rejected.shape[1]),
            intervals=int(rejected.shape[0])))
        if number:
            knot_parts[-1] = knot_parts[-1][:-4]
            if join_multiplicity == 3:
                knots = knots[1:]
                coefficient_parts[-1] = coefficient_parts[-1][:-1]
        knot_parts.append(knots)
        coefficient_parts.append(coefficients)
    result = BSpline.construct_fast(np.concatenate(knot_parts),
        np.concatenate(coefficient_parts, axis=0), 3, extrapolate=False)
    result.guard_diagnostics = diagnostics
    return result


class GuardedPiecewiseCubic:
    """Compatible density-helper interpolant, unchanged on ordinary pieces.

    This can replace pilot_density.PiecewiseCubic consistently for its coarse,
    residual, fitted and anchor curves. Accepted parts keep the original
    CubicSpline object, so there is no representation rounding on that path.
    """
    def __init__(self, times, values, corners=(), *, fallback='pchip'):
        self.times = np.asarray(times, float)
        self.values = np.asarray(values)
        joins = np.flatnonzero(np.isin(self.times, corners))
        indices = np.unique(np.r_[0, joins, len(self.times)-1])
        self.ends = self.times[indices[1:]]
        self.parts = []
        self.guard_diagnostics = []
        for left, right in zip(indices[:-1], indices[1:]):
            t = self.times[left:right+1]
            y = self.values[left:right+1]
            original = CubicSpline(t, y, axis=0)
            bad = _interval_control_mask(original, t, y)
            if np.any(bad):
                part = make_piecewise_cubic_spline(t, y, [], fallback=fallback)
            else:
                part = original
            self.parts.append(part)
            self.guard_diagnostics.append(dict(left=int(left), right=int(right),
                rejected_channels=int(np.any(bad, axis=0).sum()),
                rejected_interval_channels=int(bad.sum())))

    def __call__(self, times):
        t = np.atleast_1d(np.asarray(times, float))
        slots = np.minimum(np.searchsorted(self.ends, t, side='right'), len(self.parts)-1)
        out = np.empty((len(t),)+self.values.shape[1:], dtype=np.result_type(self.values, float))
        for i, part in enumerate(self.parts):
            mask = slots == i
            if np.any(mask):
                out[mask] = part(t[mask])
        return out
