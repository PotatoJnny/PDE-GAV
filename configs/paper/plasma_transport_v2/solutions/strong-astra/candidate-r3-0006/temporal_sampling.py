"""Progressive source-time sampling from solver-owned public pilot values.

The pilot moments use normalized probabilists' Hermites with variance theta,
ordered as (00,10,01,20,11,02,30,21,12,03).  No Oracle or solution query is
performed here.  Returned indices are in acquisition order, not time order;
sort an acquired prefix before constructing the final source interpolant.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _nearest_index(times, value):
    right = min(int(np.searchsorted(times, value)), len(times)-1)
    left = max(right-1, 0)
    return left if value-times[left] <= times[right]-value else right


def _coarse_seed(times):
    """Endpoints first, followed by progressively finer uniform dyadic nodes."""
    result, seen = [], set()
    for fraction in [0., 1.]+[odd/(2**level) for level in range(1, 5)
                              for odd in range(1, 2**level, 2)]:
        index = _nearest_index(times, times[0]+fraction*(times[-1]-times[0]))
        if index not in seen:
            result.append(index)
            seen.add(index)
    return result


def _channel_map(theta, density_weight):
    """Approximate source-to-output gains, not an estimate of the score."""
    matrix = np.zeros((10, 8), dtype=np.float64)
    scale = np.sqrt(theta)
    matrix[0, 0] = density_weight/1.0041253663782175
    matrix[1, 1] = matrix[2, 2] = scale/.4788345668393978
    matrix[0, 3] = matrix[0, 5] = theta/2.930763214320111
    matrix[3, 3] = matrix[4, 4] = matrix[5, 5] = theta*np.sqrt(2)/2.930763214320111
    heat = .5*scale**3/.9900774471118267
    matrix[6, 6], matrix[8, 6], matrix[1, 6] = heat*np.sqrt(6), heat*np.sqrt(2), 4*heat
    matrix[9, 7], matrix[7, 7], matrix[2, 7] = heat*np.sqrt(6), heat*np.sqrt(2), 4*heat
    return matrix


def select_temporal_knot_order(times, values, theta, nu, target_count,
                               density_weight=4.0):
    """Return a bounded progressive order of useful pilot-time indices.

    Seed at most 17 nearest uniform times, endpoints first, and then all pilot
    times in the first 1/256 of the interval, subject to the requested count.
    The remaining indices maximize cubic interpolation defects weighted by
    the local Ornstein--Uhlenbeck response and macroscopic output scales.

    Density receives an extra factor (default 4) to acknowledge its electric
    field coupling.  This heuristic does not reconstruct the scoring field
    error or replace an actual solution comparison.  The target is clipped to
    the number of pilot times; even counts smaller than the seed set are safe.
    A single target returns the first endpoint.  Zero returns an empty array.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("pilot times must be a finite nonempty vector")
    if np.any(np.diff(times) <= 0):
        raise ValueError("pilot times must be strictly increasing")
    if (values.ndim != 3 or values.shape[0] != len(times) or values.shape[1] < 1
            or values.shape[2] != 10 or not np.isfinite(values).all()):
        raise ValueError("pilot values must have finite shape (time, probes, 10)")
    if (not np.isfinite(theta) or theta <= 0 or not np.isfinite(nu) or nu < 0
            or not np.isfinite(density_weight) or density_weight <= 0):
        raise ValueError("positive theta/density weight and nonnegative nu required")
    if isinstance(target_count, (bool, np.bool_)) or not isinstance(target_count, (int, np.integer)) or target_count < 0:
        raise ValueError("target_count must be a nonnegative integer")
    count = min(int(target_count), len(times))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if len(times) == 1:
        return np.array([0], dtype=np.int64)

    order = _coarse_seed(times)[:count]
    chosen = np.zeros(len(times), dtype=bool)
    chosen[order] = True
    relative = (times-times[0])/(times[-1]-times[0])
    early = np.flatnonzero((relative < 1/256) & ~chosen)
    # Farthest-point order in logarithmic time gives useful prefixes even if
    # a particularly dense initial scout mesh exceeds the acquisition budget.
    if len(early) and len(order) < count:
        positive = relative[relative > 0]
        log_times = np.log(np.maximum(relative, positive[0]*.5))
        while len(early) and len(order) < count:
            distance = np.min(np.abs(log_times[early, None]-log_times[np.array(order)][None, :]), axis=1)
            index = int(early[np.argmax(distance)])
            order.append(index)
            chosen[index] = True
            early = early[early != index]

    degree = np.array([0., 1., 1., 2., 2., 2., 3., 3., 3., 3.])
    rates = nu*degree
    channel_map = _channel_map(theta, density_weight)
    # Floating point residuals of an exactly cubic pilot must not drive the
    # selection: use broad temporal coverage when defects reach roundoff.
    roundoff = (64*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
                * max(1., float(np.max(np.abs(channel_map)))))
    while len(order) < count:
        selected = np.flatnonzero(chosen)
        estimated = CubicSpline(times[selected], values[selected], axis=0)(times)
        interval = np.clip(np.searchsorted(times[selected], times, side="right")-1, 0, len(selected)-2)
        widths = times[selected[interval+1]]-times[selected[interval]]
        transfer = np.broadcast_to(widths[:, None], (len(times), 10)).copy()
        np.divide(-np.expm1(-widths[:, None]*rates[None, :]), rates[None, :],
                  out=transfer, where=rates[None, :] > 0)
        error = (estimated-values)*transfer[:, None, :]
        channels = error @ channel_map
        indicator = np.max(np.mean(channels*channels, axis=1), axis=-1)
        indicator[chosen] = -1.
        if np.max(indicator) <= (roundoff*np.max(widths))**2:
            distance = np.min(np.abs(times[:, None]-times[selected][None, :]), axis=1)
            distance[chosen] = -1.
            index = int(np.argmax(distance))
        else:
            index = int(np.argmax(indicator))
        order.append(index)
        chosen[index] = True
    return np.asarray(order, dtype=np.int64)
