"""Unfitted anisotropic OU-displacement covariance for a mixing hypothesis.

This is a source-free spatial-averaging model, not a full VPFP solver. It can
replace D*t**2*I in the existing Gaussian spatial filter without more Oracle
queries. Use one common multiplier for rho, E, j and raw stress.
"""
from __future__ import annotations

import numpy as np


def initial_thermal_covariance(moments):
    """Density-weighted mean local central covariance from raw (...,8)."""
    m = np.asarray(moments, dtype=float)
    if m.shape[-1:] != (8,) or not np.isfinite(m).all() or np.any(m[...,0] <= 0):
        raise ValueError('finite raw moments with positive density required')
    r, x, y = m[...,0], m[...,1], m[...,2]
    p = np.stack((m[...,3]-x*x/r,m[...,4]-x*y/r,m[...,5]-y*y/r),axis=-1)
    p = p.reshape(-1,3).mean(axis=0)/r.mean()
    covariance = np.asarray([[p[0],p[1]],[p[1],p[2]]])
    # Roundoff-only projection; significant negative eigenvalues indicate
    # an invalid input state and must not silently become a mixing scale.
    eigen, vectors = np.linalg.eigh(covariance)
    if np.min(eigen) < -1e-10*max(1.,np.max(eigen)):
        raise ValueError('initial mean central pressure is not positive semidefinite')
    return (vectors*np.maximum(eigen,0.))@vectors.T


def displacement_covariance(elapsed, nu, theta, initial_covariance):
    """Covariance of integrated OU thermal velocity, including nu=0."""
    t = np.asarray(elapsed, dtype=float)
    c = np.asarray(initial_covariance, dtype=float)
    nu, theta = float(nu), float(theta)
    if (not np.isfinite(t).all() or np.any(t<0) or c.shape != (2,2)
            or not np.isfinite(c).all() or not np.isfinite([nu,theta]).all()
            or nu < 0 or theta <= 0):
        raise ValueError('finite t>=0, nu>=0, theta>0 and covariance (2,2) required')
    if not np.allclose(c,c.T,atol=1e-12) or np.min(np.linalg.eigvalsh(c)) < -1e-12:
        raise ValueError('initial covariance must be symmetric positive semidefinite')
    if nu == 0:
        return t[...,None,None]**2*c
    z = nu*t
    a = -np.expm1(-z)/nu
    # Series avoids cancellation of O(z) and O(z**2) terms in the O(z**3)
    # Brownian displacement contribution as collision frequency tends to zero.
    noise_polynomial = z**3*(2/3+z*(-.5+z*(7/30+z*(-1/12+z*(31/1260)))))
    noise_direct = 2*z+4*np.expm1(-z)-np.expm1(-2*z)
    noise = theta/nu**2*np.where(z<1e-3,noise_polynomial,noise_direct)
    return a[...,None,None]**2*c+noise[...,None,None]*np.eye(2)


if __name__ == '__main__':
    # Independent quadrature of the integrated OU stochastic convolution.
    nodes,weights = np.polynomial.legendre.leggauss(96)
    c = np.asarray([[1.7,.3],[.3,.8]])
    maximum = 0.
    for nu in [0.,1e-8,.003,.03,2.]:
        for t in [0.,1e-4,.1,1.,3.]:
            got = displacement_covariance(t,nu,1.2,c)
            if nu == 0:
                expected = t*t*c
            else:
                a = -np.expm1(-nu*t)/nu
                s = t*(nodes+1)/2
                kernel = -np.expm1(-nu*s)/nu
                noise = 2*nu*1.2*(t/2)*np.dot(weights,kernel**2)
                expected = a*a*c+noise*np.eye(2)
            maximum = max(maximum,float(np.max(np.abs(got-expected)/(1+np.abs(expected)))))
            assert np.min(np.linalg.eigvalsh(got)) >= -1e-15
    assert maximum < 3e-11,maximum
    print({'ou_displacement_independent_quadrature_max':maximum})

