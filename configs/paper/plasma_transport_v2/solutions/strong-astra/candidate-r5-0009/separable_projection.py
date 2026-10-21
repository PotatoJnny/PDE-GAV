"""Portable NumPy tensor projection for normalized probabilists' Hermites.

Velocity tensors are ordered (vx, vy), with vy varying fastest when flattened.
The result uses total-degree order (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),...
Only summation order differs from a dense tensor-basis matrix product.
"""
from __future__ import annotations
import math
import numpy as np


def weighted_hermite_rule(order, theta, variance, degree):
    """Return 1D velocity nodes and weighted orthonormal Hermite polynomials.

The weights integrate ordinary dv, including sqrt(theta) and cancellation
of the probabilists' Gauss-Hermite weight exp(-z**2/2). The basis uses the
independently chosen positive variance, not necessarily theta.
"""
    order, degree = int(order), int(degree)
    if order < 1 or degree < 0 or not np.isfinite(theta) or not np.isfinite(variance) or theta <= 0 or variance <= 0:
        raise ValueError('positive order, theta and variance, nonnegative degree required')
    z, w = np.polynomial.hermite_e.hermegauss(order)
    velocity = z * math.sqrt(theta)
    weight = w * np.exp(z*z/2) * math.sqrt(theta)
    argument = velocity / math.sqrt(variance)
    h = np.empty((degree+1, order)); h[0] = 1.
    if degree: h[1] = argument
    for n in range(1, degree):
        h[n+1] = (argument*h[n] - math.sqrt(n)*h[n-1]) / math.sqrt(n+1)
    return velocity, h * weight[None, :]


class SeparableHermiteProjector:
    """Reusable full-square contraction followed by triangular degree gather.

``values`` may have shape (..., q, q) or (..., q*q). Leading batch axes are
preserved. Signed and complex input values are supported. This helper does
not normalize mass, alter density, or select a quadrature rule.
"""
    def __init__(self, weighted_h, degree=None):
        h = np.asarray(weighted_h)
        if h.ndim != 2 or min(h.shape) < 1 or not np.issubdtype(h.dtype, np.number):
            raise ValueError('weighted_h must be a nonempty numeric matrix')
        degree = h.shape[0]-1 if degree is None else int(degree)
        if degree < 0 or degree >= h.shape[0]:
            raise ValueError('degree exceeds supplied Hermite rows')
        self.weighted_h = np.ascontiguousarray(h[:degree+1])
        self.degree = degree
        self.order = h.shape[1]
        indices = np.array([(a, d-a) for d in range(degree+1)
                            for a in range(d, -1, -1)], dtype=np.intp)
        self.x_degree, self.y_degree = indices.T

    def project(self, values):
        values = np.asarray(values)
        q = self.order
        if values.ndim >= 2 and values.shape[-2:] == (q, q):
            tensor = values
        elif values.ndim >= 1 and values.shape[-1] == q*q:
            tensor = values.reshape(values.shape[:-1] + (q, q))
        else:
            raise ValueError('values must end in (q,q) or (q*q,)')
        square = self.weighted_h @ tensor @ self.weighted_h.T
        return square[..., self.x_degree, self.y_degree]
