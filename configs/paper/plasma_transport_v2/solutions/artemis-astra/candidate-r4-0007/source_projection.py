"""Exact L2 projection of piecewise polynomial source elements.

No queries: maps the existing source Gauss samples to state DG nodal values.
"""
import importlib.util
from pathlib import Path
import numpy as np

_path=Path(__file__).with_name('source_geometry.py')
_spec=importlib.util.spec_from_file_location('_projection_geometry',_path)
geometry=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)


def interpolation_matrix(source_edges,source_order,targets):
    indices,weights=geometry.element_interpolation(source_edges,source_order,targets)
    out=np.zeros((len(targets),(len(source_edges)-1)*source_order))
    out[np.arange(len(targets))[:,None],indices]=weights
    return out


def state_quadrature_projection(source_edges,state_edges,state_order=6,source_order=12,
                                quadrature_order=12):
    """Unsplit Gauss projection; exact only absent interior source boundaries."""
    q,w=np.polynomial.legendre.leggauss(quadrature_order)
    z,_=np.polynomial.legendre.leggauss(state_order)
    basis=np.polynomial.legendre.legvander(z,state_order-1)
    qbasis=np.polynomial.legendre.legvander(q,state_order-1)
    projection=((basis*(2*np.arange(state_order)+1)[None,:]/2)@qbasis.T)*w
    v=((state_edges[:-1]+state_edges[1:])[:,None]/2
       +np.diff(state_edges)[:,None]*q/2).ravel()
    samples=interpolation_matrix(source_edges,source_order,v)
    return np.einsum('nq,cqs->cns',projection,
        samples.reshape(len(state_edges)-1,quadrature_order,-1)).reshape(
        state_order*(len(state_edges)-1),-1)


def exact_element_projection(source_edges,state_edges,state_order=6,source_order=12,
                             quadrature_order=None):
    """Integrate over every source/state cell intersection.

    A source polynomial has degree source_order-1 and the highest state basis
    has degree state_order-1. Gauss ceil((source_order+state_order-1)/2)
    therefore exactly integrates each overlap, apart from roundoff.
    """
    if quadrature_order is None:
        quadrature_order=(source_order+state_order)//2
    q,w=np.polynomial.legendre.leggauss(quadrature_order)
    z,_=np.polynomial.legendre.leggauss(state_order)
    state_basis=np.polynomial.legendre.legvander(z,state_order-1)
    ns=(len(source_edges)-1)*source_order
    matrix=np.zeros(((len(state_edges)-1)*state_order,ns))
    for c,(lo,hi) in enumerate(zip(state_edges[:-1],state_edges[1:])):
        cuts=np.r_[lo,source_edges[(source_edges>lo)&(source_edges<hi)],hi]
        v=((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*q/2).ravel()
        weight=(np.diff(cuts)[:,None]*w/2).ravel()
        local_z=2*(v-lo)/(hi-lo)-1
        qbasis=np.polynomial.legendre.legvander(local_z,state_order-1)
        modal=(qbasis.T*weight[None,:])*(2*np.arange(state_order)+1)[:,None]/(hi-lo)
        matrix[c*state_order:(c+1)*state_order]=state_basis@modal@interpolation_matrix(
            source_edges,source_order,v)
    return matrix
