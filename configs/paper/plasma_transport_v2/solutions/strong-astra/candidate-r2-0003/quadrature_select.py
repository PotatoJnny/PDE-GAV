"""Bounded full-velocity quadrature refinement from public source probes."""
from __future__ import annotations

import math
import numpy as np


def _hermite(z, degree):
    values = np.empty((degree + 1, len(z)), dtype=float)
    values[0] = 1.
    if degree:
        values[1] = z
    for k in range(1, degree):
        values[k + 1] = (z * values[k] - math.sqrt(k) * values[k - 1]) / math.sqrt(k + 1)
    return values


def select_quadrature(get, *, periods, t_interval, theta, variance,
                      base_order, tolerance=5.e-5, degree=12):
    """Choose base_order, base_order+4, or base_order+8 using degree-12 moments.

    ``get("source", points)`` owns request splitting and the shared Case clock.
    All coordinates are fixed before preparation receives any scoring query.
    """
    if (isinstance(base_order, bool) or int(base_order) != base_order
            or base_order < 2 or not math.isfinite(theta) or theta <= 0
            or not math.isfinite(variance) or variance <= 0):
        raise ValueError("positive quadrature order, temperature, and basis variance required")
    base_order = int(base_order)
    periods = np.asarray(periods, dtype=float)
    interval = np.asarray(t_interval, dtype=float)
    if (periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0)
            or interval.shape != (2,) or not np.isfinite(interval).all()
            or interval[1] <= interval[0]):
        raise ValueError("positive periods and an ordered public time interval required")
    fractions = np.array([0., 1.e-7, 1.e-5, .001, .01, .05, .2, .5, .8, 1.])
    times = interval[0] + (interval[1] - interval[0]) * fractions
    index = np.arange(1, 9)
    xy = np.column_stack((np.mod(index * .6180339887498949, 1.),
                          np.mod(index * .4142135623730951, 1.))) * periods
    tx = np.column_stack((np.repeat(times, len(xy)), np.tile(xy, (len(times), 1))))
    powers = [(a, d - a) for d in range(degree+1) for a in range(d, -1, -1)]

    def plan(order):
        z, w = np.polynomial.hermite_e.hermegauss(order)
        velocity = z * math.sqrt(theta)
        weights = w * np.exp(z * z / 2) * math.sqrt(theta)
        h = _hermite(velocity / math.sqrt(variance), degree)
        basis = np.array([(h[a, :, None] * h[b, None, :]
                           * weights[:, None] * weights[None, :]).ravel()
                          for a, b in powers])
        vx, vy = np.meshgrid(velocity, velocity, indexing="ij")
        points = np.empty((len(tx), order * order, 5), dtype=float)
        points[:, :, :3] = tx[:, None, :]
        points[:, :, 3] = vx.ravel()
        points[:, :, 4] = vy.ravel()
        return points.reshape(-1, 5), basis

    def moments(values, basis):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(tx) * basis.shape[1],) or not np.isfinite(values).all():
            raise ValueError("public source probe must return one finite value per point")
        projected = values.reshape(len(tx), -1) @ basis.T
        if not np.isfinite(projected).all():
            raise ArithmeticError("nonfinite source quadrature projection")
        return projected

    def discrepancy(coarse, fine):
        numerator = np.max(np.abs(coarse - fine), axis=0)
        denominator = 1. + np.max(np.abs(fine), axis=0)
        return float(np.max(numerator / denominator))

    coarse_points, coarse_basis = plan(base_order)
    fine_points, fine_basis = plan(base_order + 4)
    values = np.asarray(get("source", np.concatenate((coarse_points, fine_points))), dtype=float)
    split = len(coarse_points)
    coarse = moments(values[:split], coarse_basis)
    fine = moments(values[split:], fine_basis)
    if discrepancy(coarse, fine) <= tolerance:
        return base_order
    finer_points, finer_basis = plan(base_order + 8)
    finer = moments(get("source", finer_points), finer_basis)
    return base_order + (4 if discrepancy(fine, finer) <= tolerance else 8)
