"""Shared differentiable one-dimensional bases for continuous NS fields.

Coordinates and knots use [0,1]. Derivatives returned here are with respect to
that normalized coordinate; callers apply physical coordinate scale factors.
B-spline endpoint values/derivatives use the interior one-sided polynomial.
"""
from __future__ import annotations

import numpy as np


def open_uniform_knots(n_basis: int, degree: int) -> tuple[float, ...]:
    if (isinstance(n_basis, bool) or int(n_basis) != n_basis
            or isinstance(degree, bool) or int(degree) != degree
            or degree < 0 or n_basis < degree + 1):
        raise ValueError('a B-spline requires integer n_basis >= degree+1')
    interior = np.linspace(0., 1., int(n_basis)-int(degree)+1)[1:-1]
    return (0.,) * (int(degree)+1) + tuple(float(x) for x in interior) + (1.,) * (int(degree)+1)


def validate_bspline_spec(knots, *, n_basis: int, degree: int) -> tuple[float, ...]:
    if isinstance(degree, bool) or int(degree) != degree or degree < 0:
        raise ValueError('B-spline degree must be a nonnegative integer')
    if isinstance(n_basis, bool) or int(n_basis) != n_basis or n_basis < degree+1:
        raise ValueError('B-spline basis count must be at least degree+1')
    values = tuple(float(x) for x in knots)
    if len(values) != int(n_basis)+int(degree)+1 or not np.isfinite(values).all():
        raise ValueError('B-spline knot count or finite values disagree with the declared basis')
    p = int(degree)
    if values[:p+1] != (0.,)*(p+1) or values[-p-1:] != (1.,)*(p+1):
        raise ValueError('B-spline knots must be clamped to [0,1] with degree+1 endpoint repeats')
    interior = values[p+1:-p-1]
    if (any(not 0. < x < 1. for x in interior)
            or any(b <= a for a, b in zip(interior, interior[1:]))):
        raise ValueError('B-spline interior knots must be simple and strictly increasing')
    return values


def _bspline_jet(coordinates, *, knots, degree, order):
    import torch
    if isinstance(order, bool) or int(order) != order or order < 0:
        raise ValueError('derivative order must be a nonnegative integer')
    degree, order = int(degree), int(order)
    knots = validate_bspline_spec(knots, n_basis=len(knots)-degree-1, degree=degree)
    x = torch.as_tensor(coordinates)
    if not x.is_floating_point():
        x = x.to(dtype=torch.float64)
    t = x.new_tensor(knots)
    count = len(knots)-1
    # The last positive interval owns x=1. Recursion then evaluates the left
    # polynomial and its true endpoint derivatives, without epsilon clipping.
    endpoint_interval = torch.tensor([a < b and b == 1. for a, b in zip(knots, knots[1:])], device=x.device)
    support = ((x[..., None] >= t[:-1]) & (x[..., None] < t[1:])) | ((x[..., None] == 1.) & endpoint_interval)
    table = {(0, 0): support.to(x.dtype) + x[..., None]*0.}
    for r in range(1, order+1):
        table[0, r] = table[0, 0]*0.
    for p in range(1, degree+1):
        count -= 1
        left_den = [knots[i+p]-knots[i] for i in range(count)]
        right_den = [knots[i+p+1]-knots[i+1] for i in range(count)]
        left = x.new_tensor([1./d if d > 0 else 0. for d in left_den])
        right = x.new_tensor([1./d if d > 0 else 0. for d in right_den])
        previous = table[p-1, 0]
        table[p, 0] = ((x[..., None]-t[:count])*left*previous[..., :count]
                       + (t[p+1:p+1+count]-x[..., None])*right*previous[..., 1:count+1])
        for r in range(1, order+1):
            previous = table[p-1, r-1]
            table[p, r] = p*(left*previous[..., :count]-right*previous[..., 1:count+1])
    return tuple(table[degree, r] for r in range(order+1))


def bspline_basis(coordinates, *, knots, degree: int, derivative_order: int = 0):
    """Return [...,n_basis] values or an exact B-spline derivative in Torch."""
    return _bspline_jet(coordinates, knots=knots, degree=degree, order=derivative_order)[derivative_order]


def axis_basis_derivatives(coordinates, *, config, axis: str, order: int = 0):
    """Return (B, dB, ..., d^order B), for normalized x, y or t coordinates."""
    import torch
    if axis not in ('x', 'y', 't'):
        raise ValueError('basis axis must be x, y, or t')
    if isinstance(order, bool) or int(order) != order or order < 0:
        raise ValueError('derivative order must be a nonnegative integer')
    order = int(order)
    x = torch.as_tensor(coordinates)
    if not x.is_floating_point():
        x = x.to(dtype=torch.float64)
    temporal = axis == 't'
    kind = config.temporal_basis if temporal else config.spatial_basis
    if kind == 'bspline':
        return _bspline_jet(x, knots=(config.temporal_knots if temporal else getattr(config, f'spatial_knots_{axis}')),
                            degree=config.temporal_degree if temporal else config.spatial_degree, order=order)
    if not temporal and kind == 'cosine':
        count = int(getattr(config, f'spatial_mode_{axis}'))+1
        k = torch.arange(count, dtype=x.dtype, device=x.device)
        extension = float(config.spatial_extension)
        width = 1. + 2.*extension
        argument = torch.pi * ((x[..., None]+extension)/width) * k
        omega = torch.pi*k/width
        cosine, sine = torch.cos(argument), torch.sin(argument)
        factors = (cosine, -sine, -cosine, sine)
        return tuple(factors[r % 4] * omega**r for r in range(order+1))
    if temporal and kind == 'chebyshev':
        z = 2.*x-1.
        zero = x*0.
        columns = [[torch.ones_like(x) if r == 0 else zero] for r in range(order+1)]
        if int(config.time_order):
            for r in range(order+1):
                columns[r].append(z if r == 0 else torch.ones_like(x)*2. if r == 1 else zero)
        for degree in range(2, int(config.time_order)+1):
            for r in range(order, -1, -1):
                value = 2.*z*columns[r][degree-1]-columns[r][degree-2]
                if r:
                    value = value+4.*r*columns[r-1][degree-1]
                columns[r].append(value)
        return tuple(torch.stack(values, dim=-1) for values in columns)
    raise ValueError(f'unsupported {axis} basis {kind!r}')


def spatial_basis_with_derivatives(normalized_coords, *, config):
    x = axis_basis_derivatives(normalized_coords[..., 0], config=config, axis='x', order=1)
    y = axis_basis_derivatives(normalized_coords[..., 1], config=config, axis='y', order=1)
    shape = (*normalized_coords.shape[:-1], config.spatial_coefficient_count)
    product = lambda a, b: (a[..., :, None]*b[..., None, :]).reshape(shape)
    return product(x[0], y[0]), product(x[1], y[0]), product(x[0], y[1])
