"""Stable independent cubic splines in a compact B-spline representation.

The scalar cubic is solved in the usual derivative formulation. Its power
basis is converted analytically by polynomial blossoming, avoiding the
ill-conditioned B-spline collocation system on strongly clustered knots.
Complex vector-valued data are handled in bounded channel blocks.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline, CubicSpline


def _cubic_block_coefficients(times, values, block_size=512):
    n = len(times)
    shape = values.shape[1:]
    data = values.reshape(n, -1)
    if n < 4:
        knots = np.r_[np.repeat(times[0], 4), np.repeat(times[-1], 4)]
        result = np.empty((4, data.shape[1]), dtype=np.result_type(values.dtype, float))
        for first in range(0, data.shape[1], block_size):
            part = slice(first, first + block_size)
            cubic = CubicSpline(times, data[:, part], axis=0)
            h = times[-1] - times[0]
            result[0, part] = data[0, part]
            result[1, part] = data[0, part] + (h / 3) * cubic(times[0], 1)
            result[2, part] = data[-1, part] - (h / 3) * cubic(times[-1], 1)
            result[3, part] = data[-1, part]
    else:
        knots = np.r_[np.repeat(times[0], 4), times[2:-2], np.repeat(times[-1], 4)]
        result = np.empty((n, data.shape[1]), dtype=np.result_type(values.dtype, float))
        # Each coefficient is the blossom of a local cubic at three knots.
        # The final two coefficients use the last polynomial interval.
        spans = np.minimum(np.arange(n), n - 2)
        d = np.stack([knots[1:n+1], knots[2:n+2], knots[3:n+3]], axis=1) - times[spans, None]
        linear = d.sum(axis=1) / 3
        quadratic = (d[:, 0]*d[:, 1] + d[:, 0]*d[:, 2] + d[:, 1]*d[:, 2]) / 3
        cubic_term = d.prod(axis=1)
        for first in range(0, data.shape[1], block_size):
            part = slice(first, first + block_size)
            pp = CubicSpline(times, data[:, part], axis=0).c[:, spans]
            result[:, part] = (pp[3] + linear[:, None]*pp[2]
                              + quadratic[:, None]*pp[1] + cubic_term[:, None]*pp[0])
    return knots, result.reshape((len(result),) + shape)


def make_piecewise_cubic_spline(times, values, break_indices, *, join_multiplicity=4):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Source times must be finite and strictly increasing')
    if values.ndim < 1 or len(values) != len(times):
        raise ValueError('Source values and times have different lengths')
    if join_multiplicity not in (3, 4):
        raise ValueError('Cubic segment join multiplicity must be three or four')
    breaks = np.asarray(break_indices, dtype=int)
    if breaks.ndim != 1 or np.any(breaks < 0) or np.any(breaks >= len(times)):
        raise ValueError('Break indices must lie within source samples')
    bounds = np.unique(np.r_[0, breaks, len(times)-1])
    coefficient_parts, knot_parts = [], []
    for number, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        knots, coefficients = _cubic_block_coefficients(times[left:right+1], values[left:right+1])
        if number:
            knot_parts[-1] = knot_parts[-1][:-4]
            if join_multiplicity == 3:
                knots = knots[1:]
                coefficient_parts[-1] = coefficient_parts[-1][:-1]
        knot_parts.append(knots)
        coefficient_parts.append(coefficients)
    return BSpline.construct_fast(np.concatenate(knot_parts),
                                  np.concatenate(coefficient_parts, axis=0),
                                  3, extrapolate=False)
