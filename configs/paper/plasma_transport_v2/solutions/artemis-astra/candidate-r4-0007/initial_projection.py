"""Gauss12 initial-data quadrature projected into the existing DG space."""
from __future__ import annotations

import numpy as np

QUADRATURE_ORDER = 12


def projection_matrix(state_order):
    """Reference-cell L2 projection from quadrature values to state nodes."""
    high_nodes, high_weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    state_nodes, _ = np.polynomial.legendre.leggauss(state_order)
    high_basis = np.polynomial.legendre.legvander(high_nodes, state_order-1)
    state_basis = np.polynomial.legendre.legvander(state_nodes, state_order-1)
    inverse_mass = (2*np.arange(state_order)+1)/2
    projection = ((state_basis*inverse_mass[None, :])@high_basis.T)*high_weights[None, :]
    return high_nodes, projection


def sample_initial(x, edges, state_order, query):
    """Query only f0 at Gauss12 nodes and return projected DG nodal values.

    The affine cell Jacobian cancels in the projection. With six state nodes,
    the state's quadrature moments0--5 equal the Gauss12 input moments.
    """
    high_nodes, projection = projection_matrix(state_order)
    widths = np.diff(edges)
    velocity = ((edges[:-1, None]+edges[1:, None])/2
                +widths[:, None]*high_nodes/2).ravel()
    points = np.column_stack((np.repeat(x, len(velocity)), np.tile(velocity, len(x))))
    values = query(points, initial=True).reshape(len(x), len(widths), QUADRATURE_ORDER)
    return np.einsum('xcq,nq->xcn', values, projection).reshape(len(x), len(widths)*state_order)
