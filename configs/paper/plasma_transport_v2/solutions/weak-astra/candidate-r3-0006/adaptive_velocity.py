"""Local Gauss--Hermite node plans from already prepared initial moments."""
from __future__ import annotations
import numpy as np


def local_hermite(raw, order=32, scale=.75, rotate=True):
    """Return relative velocities[r,n*n,2] and integration weights[r,n*n].

    raw is degree-ordered relative velocity moments with shape(moments,r).
    No Gaussian shape is assumed for the integrand. The Gaussian quadrature
    weight is divided out; the result integrates arbitrary sampled source.
    Its accuracy still requires independent numerical validation.
    """
    raw = np.asarray(raw, float)
    if (isinstance(order, bool) or int(order) != order or order < 2
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('integer order>=2 and finite positive scale required')
    if raw.ndim != 2 or raw.shape[0] < 6 or np.any(raw[0] <= 0) or not np.isfinite(raw).all():
        raise ValueError('finite initial moment matrix with positive density required')
    r = raw[0]
    mean = raw[1:3].T/r[:, None]
    covariance = np.empty((len(r), 2, 2))
    covariance[:, 0, 0] = raw[3]/r-mean[:, 0]**2
    covariance[:, 0, 1] = covariance[:, 1, 0] = raw[4]/r-mean[:, 0]*mean[:, 1]
    covariance[:, 1, 1] = raw[5]/r-mean[:, 1]**2
    if rotate:
        eigen, vectors = np.linalg.eigh(covariance)
        if np.min(eigen) <= 0:
            raise ValueError('positive initial central velocity covariance required')
        transforms = vectors*np.sqrt(2*eigen)[:, None, :]*scale
    else:
        diagonal = np.diagonal(covariance, axis1=1, axis2=2)
        if np.min(diagonal) <= 0:
            raise ValueError('positive initial diagonal velocity covariance required')
        transforms = np.zeros_like(covariance)
        transforms[:, 0, 0] = np.sqrt(2*diagonal[:, 0])*scale
        transforms[:, 1, 1] = np.sqrt(2*diagonal[:, 1])*scale
    node, weight = np.polynomial.hermite.hermgauss(int(order))
    normal_nodes = np.stack(np.meshgrid(node, node, indexing='ij'), -1).reshape(-1, 2)
    flat_weight = np.outer(weight*np.exp(node**2), weight*np.exp(node**2)).ravel()
    velocity = mean[:, None, :]+np.einsum('rij,nj->rni', transforms, normal_nodes)
    integration = abs(np.linalg.det(transforms))[:, None]*flat_weight
    return velocity, integration
