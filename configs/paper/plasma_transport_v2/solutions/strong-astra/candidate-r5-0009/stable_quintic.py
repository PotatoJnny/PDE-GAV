"""Bound quintic interpolation while preserving ordinary accepted splines.

Fallbacks are exactly degree-elevated stable cubic splines.  Only numerical
source interpolation changes; there is no Oracle or query access here.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator, make_interp_spline


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


def elevate_cubic(times, values, block_size=512):
    """Represent the ordinary not-a-knot cubic exactly as a quintic B-spline.

    Raising degree from three to five adds two to every knot multiplicity.
    The quintic blossom of a cubic polynomial consists of its elementary
    symmetric polynomials, divided by the corresponding binomial factors.
    This construction avoids a second, ill-conditioned interpolation solve.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values)
    n = len(times)
    interior = times[2:-2] if n >= 4 else np.empty(0)
    knots = np.r_[np.repeat(times[0], 6), np.repeat(interior, 3),
                  np.repeat(times[-1], 6)]
    count = len(knots) - 6
    arguments = np.stack([knots[j:j + count] for j in range(1, 6)], axis=1)
    # At repeated knots either adjacent cubic is equivalent.  The median
    # argument chooses a nearby interval and limits cancellation.
    spans = np.clip(np.searchsorted(times, arguments[:, 2], side='right') - 1,
                    0, n - 2)
    offsets = arguments - times[spans, None]
    first = offsets.sum(axis=1) / 5.
    second = sum(offsets[:, i] * offsets[:, j]
                 for i, j in itertools.combinations(range(5), 2)) / 10.
    third = sum(offsets[:, i] * offsets[:, j] * offsets[:, k]
                for i, j, k in itertools.combinations(range(5), 3)) / 10.
    flat = values.reshape(n, -1)
    result = np.empty((count, flat.shape[1]), dtype=np.result_type(values, float))
    for begin in range(0, flat.shape[1], block_size):
        part = slice(begin, begin + block_size)
        pp = CubicSpline(times, flat[:, part], axis=0).c[:, spans]
        result[:, part] = (pp[3] + first[:, None] * pp[2]
                          + second[:, None] * pp[1] + third[:, None] * pp[0])
    return BSpline.construct_fast(knots, result.reshape((count,) + values.shape[1:]),
                                  5, extrapolate=False)


def _bounded(spline, values, amplification=64.):
    """Reject nonfinite or grossly amplified control polygons per channel."""
    if not np.isfinite(spline.c).all():
        return False
    flat = np.asarray(values).reshape(len(values), -1)
    coeff = np.asarray(spline.c).reshape(len(spline.c), -1)
    amplitude = np.max(abs(flat), axis=0)
    # A small absolute noise allowance avoids treating exact zeros specially.
    allowance = amplification * np.maximum(amplitude, 1.e-14)
    return bool(np.all(np.max(abs(coeff), axis=0) <= allowance))


def elevate_shape_preserving(times, values, block_size=512):
    """A bounded final fallback when clustered input noise destabilizes cubic.

    PCHIP handles real and imaginary components independently. Its cubic
    Bernstein controls are degree elevated algebraically. Exact supplied
    frames and conjugate symmetry are preserved.
    """
    times, values = np.asarray(times), np.asarray(values)
    n = len(times)
    knots = np.r_[np.repeat(times[0], 6), np.repeat(times[1:-1], 5),
                  np.repeat(times[-1], 6)]
    flat = values.reshape(n, -1)
    result = np.empty((5*(n-1)+1, flat.shape[1]), dtype=np.result_type(values, float))
    h = np.diff(times)[:, None]
    # Degree-three Bernstein to degree-five Bernstein elevation matrix.
    transform = np.array([[1, 0, 0, 0], [.4, .6, 0, 0], [.1, .6, .3, 0],
                          [0, .3, .6, .1], [0, 0, .6, .4], [0, 0, 0, 1.]])
    for begin in range(0, flat.shape[1], block_size):
        part = slice(begin, begin + block_size)
        data = flat[:, part]
        derivatives = _shape_derivatives(times, data)
        bernstein = np.stack([data[:-1], data[:-1]+h*derivatives[:-1]/3,
                              data[1:]-h*derivatives[1:]/3, data[1:]], axis=1)
        elevated = np.einsum('ij,sjc->sic', transform, bernstein)
        result[:-1, part] = elevated[:, :5].reshape(-1, len(range(*part.indices(flat.shape[1]))))
        result[-1, part] = data[-1]
    return BSpline.construct_fast(knots, result.reshape((len(result),)+values.shape[1:]),
                                  5, extrapolate=False)


def make_piecewise_quintic_spline(times, values, break_indices):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Source times must be finite and strictly increasing')
    if values.ndim < 1 or len(values) != len(times):
        raise ValueError('Source values and times have different lengths')
    breaks = np.asarray(break_indices, dtype=int)
    if (breaks.ndim != 1 or np.any(breaks < 0)
            or np.any(breaks >= len(times))):
        raise ValueError('Break indices must lie within source samples')
    bounds = np.unique(np.r_[0, breaks, len(times) - 1])
    coefficient_parts, knot_parts = [], []
    for number, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        local_times, local_values = times[left:right + 1], values[left:right + 1]
        spline = None
        if len(local_times) >= 6:
            try:
                candidate = make_interp_spline(local_times, local_values, k=5, axis=0)
                if _bounded(candidate, local_values):
                    spline = candidate
            except (ValueError, np.linalg.LinAlgError):
                pass
        if spline is None:
            spline = elevate_cubic(local_times, local_values)
            if not _bounded(spline, local_values):
                spline = elevate_shape_preserving(local_times, local_values)
        if number:
            knot_parts[-1] = knot_parts[-1][:-6]
        knot_parts.append(spline.t)
        coefficient_parts.append(spline.c)
    return BSpline.construct_fast(np.concatenate(knot_parts),
                                  np.concatenate(coefficient_parts, axis=0),
                                  5, extrapolate=False)
