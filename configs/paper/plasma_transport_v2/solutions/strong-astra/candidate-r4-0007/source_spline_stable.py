"""Stable independent cubic splines in a compact B-spline representation.

The scalar cubic is solved in the usual derivative formulation. Its power
basis is converted analytically by polynomial blossoming, avoiding the
ill-conditioned B-spline collocation system on strongly clustered knots.
Complex vector-valued data are handled in bounded channel blocks.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline, CubicSpline


def _cubic_block_coefficients(times, values, block_size=None, out=None):
    n = len(times)
    shape = values.shape[1:]
    data = values.reshape(n, -1)
    if block_size is None:
        block_size = 8192 if n <= 12 else 2048 if n <= 32 else 512
    required = 4 if n < 4 else n
    result = (np.empty((required, data.shape[1]), dtype=np.result_type(values.dtype, float))
              if out is None else out.reshape(required, data.shape[1]))
    if n == 2:
        # CubicSpline's two-node not-a-knot special case is exactly linear.
        # Its Bernstein coefficients need no polynomial construction or solve.
        knots = np.r_[np.repeat(times[0], 4), np.repeat(times[-1], 4)]
        first = np.asarray(data[0], dtype=result.dtype)
        last = np.asarray(data[1], dtype=result.dtype)
        result[0] = data[0]
        result[1] = first + (last - first) / 3
        result[2] = last - (last - first) / 3
        result[3] = data[1]
        return knots, result.reshape((required,) + shape)
    if n < 4:
        knots = np.r_[np.repeat(times[0], 4), np.repeat(times[-1], 4)]
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
    segment_sizes = np.maximum(np.diff(bounds) + 1, 4)
    segment_count = len(segment_sizes)
    overlap = int(join_multiplicity == 3)
    coefficient_count = int(segment_sizes.sum() - overlap * (segment_count - 1))
    coefficients = np.empty((coefficient_count,) + values.shape[1:],
                            dtype=np.result_type(values.dtype, float))
    all_knots = np.empty(coefficient_count + 4, dtype=float)
    coefficient_offset = knot_offset = 0
    for number, (left, right, size) in enumerate(zip(bounds[:-1], bounds[1:], segment_sizes)):
        # For join multiplicity three, the final coefficient of each nonfinal
        # part is written into the next part's first slot and then overwritten.
        # This is exactly the former coefficient_parts[-1][:-1] rule.
        target = coefficients[coefficient_offset:coefficient_offset+size]
        knots, _ = _cubic_block_coefficients(times[left:right+1], values[left:right+1], out=target)
        start = overlap if number else 0
        stop = -4 if number < segment_count-1 else None
        part = knots[start:stop]
        all_knots[knot_offset:knot_offset+len(part)] = part
        knot_offset += len(part)
        coefficient_offset += int(size) - (overlap if number < segment_count-1 else 0)
    return BSpline.construct_fast(all_knots, coefficients, 3, extrapolate=False)
