"""Bounded initial-g0 quadrature with independent order/scale checks.

Only g0 is queried; source quadrature is deliberately unaffected. ``get`` owns
request splitting and the shared Case clock, and its exceptions propagate.
"""
from __future__ import annotations

import math
import numpy as np


def _hermite(z, degree):
    h=np.empty((degree+1,len(z))); h[0]=1.
    if degree: h[1]=z
    for k in range(1,degree): h[k+1]=(z*h[k]-math.sqrt(k)*h[k-1])/math.sqrt(k+1)
    return h


def select_initial_quadrature(get, *, periods, theta, variance, degree=14,
                              orders=(28,36,48,64,80,96), broaden=True,
                              tolerances=(1.e-8,1.e-6,1.e-5),
                              normalization_tolerance=1.e-6, diagnostics=None):
    """Return ``(order, quadrature_variance)`` for the initial g0 projection.

    First establish a reference at quadrature variance theta by comparing
    successive orders. A half-tolerance margin is used for order convergence.
    The selected coarse rule must also integrate normalized g0 to within the
    supplied normalization tolerance at every probe site. If no pair passes,
    return the highest theta order, mark it unconverged, and do not broaden.

    When this reference needs more than the lowest order, test midpoint and
    basis variances for a cheaper rule. A broadened rule must agree with its
    next order AND with the finer theta reference through every retained
    degree; its fine rule must agree with that reference too. Choose the
    smallest accepted order, breaking ties toward the narrower quadrature.

    This finite probe is a safeguard, not an accuracy certificate outside its
    eight deterministic spatial sites. The final order remains a bounded
    fallback when a Gaussian component is too narrow for the order budget.
    """
    periods=np.asarray(periods,dtype=float)
    if (periods.shape!=(2,) or not np.isfinite(periods).all() or np.any(periods<=0)
            or not math.isfinite(theta) or theta<=0 or not math.isfinite(variance) or variance<=0
            or isinstance(degree,(bool,np.bool_)) or not isinstance(degree,(int,np.integer)) or degree<0):
        raise ValueError('positive domain, theta, variance, and integer nonnegative degree required')
    if len(orders)<2 or any(isinstance(n,(bool,np.bool_)) or int(n)!=n or n<2 for n in orders):
        raise ValueError('at least two increasing integer quadrature orders required')
    orders=tuple(map(int,orders))
    if any(a>=b for a,b in zip(orders,orders[1:])):
        raise ValueError('quadrature orders must increase strictly')
    tolerances=np.asarray(tolerances,dtype=float)
    if tolerances.shape!=(3,) or not np.isfinite(tolerances).all() or np.any(tolerances<=0):
        raise ValueError('three positive finite degree-band tolerances required')
    if not math.isfinite(normalization_tolerance) or normalization_tolerance<=0:
        raise ValueError('positive normalization tolerance required')
    powers=[(a,d-a) for d in range(degree+1) for a in range(d,-1,-1)]
    ds=np.array([a+b for a,b in powers]); bands=np.where(ds<=3,0,np.where(ds<=8,1,2))
    thresholds=tolerances[bands]
    index=np.arange(1,9)
    xy=np.column_stack((np.mod(index*.6180339887498949,1.),
                        np.mod(index*.4142135623730951,1.)))*periods
    cache={}; tested=[]; points_count=0

    def sample(order,quad_variance):
        nonlocal points_count
        key=(order,float(quad_variance))
        if key in cache: return cache[key]
        z,w=np.polynomial.hermite_e.hermegauss(order)
        velocity=z*math.sqrt(quad_variance)
        weights=w*np.exp(z*z/2)*math.sqrt(quad_variance)
        h=_hermite(velocity/math.sqrt(variance),degree)
        basis=np.array([(h[a,:,None]*h[b,None,:]*weights[:,None]*weights[None,:]).ravel()
                        for a,b in powers])
        vx,vy=np.meshgrid(velocity,velocity,indexing='ij')
        points=np.empty((len(xy),order*order,4)); points[:,:,:2]=xy[:,None,:]
        points[:,:,2]=vx.ravel(); points[:,:,3]=vy.ravel()
        points=points.reshape(-1,4)
        values=np.asarray(get('g0',points),dtype=float)
        if values.shape!=(len(points),) or not np.isfinite(values).all():
            raise ValueError('public g0 probe must return one finite value per point')
        moments=values.reshape(len(xy),-1)@basis.T
        if not np.isfinite(moments).all(): raise ArithmeticError('nonfinite initial quadrature projection')
        cache[key]=moments; points_count+=len(points)
        tested.append(dict(order=order,quadrature_variance=float(quad_variance),
                           normalization_error=float(np.max(np.abs(moments[:,0]-1)))))
        return moments

    def discrepancy(coarse,fine):
        return np.max(np.abs(coarse-fine),axis=0)/(1+np.max(np.abs(fine),axis=0))

    def agrees(coarse,fine,factor=1.):
        return bool(np.all(discrepancy(coarse,fine)<=factor*thresholds))

    def normalized(moments):
        return bool(np.max(np.abs(moments[:,0]-1))<=normalization_tolerance)

    selected=orders[-1]; reference_order=orders[-1]; converged=False
    coarse=sample(orders[0],theta)
    for n,fine_n in zip(orders,orders[1:]):
        fine=sample(fine_n,theta)
        if normalized(coarse) and normalized(fine) and agrees(coarse,fine,.5):
            selected=n; reference_order=fine_n; converged=True; break
        coarse=fine
    quad_variance=float(theta)
    reference=sample(reference_order,theta)
    candidates=[dict(order=selected,quadrature_variance=quad_variance)]
    if broaden and converged and selected>orders[0] and variance>theta*(1+1.e-12):
        scales=sorted(set([float(.5*(theta+variance)),float(variance)]))
        for candidate_scale in scales:
            for n,fine_n in zip(orders,orders[1:]):
                if n>=selected: break
                coarse=sample(n,candidate_scale); fine=sample(fine_n,candidate_scale)
                if (normalized(coarse) and normalized(fine) and agrees(coarse,fine,.5)
                        and agrees(coarse,reference) and agrees(fine,reference)):
                    selected=n; quad_variance=candidate_scale
                    candidates.append(dict(order=selected,quadrature_variance=quad_variance))
                    break
    if diagnostics is not None:
        diagnostics.update(dict(converged=converged,selected_order=selected,
                                quadrature_variance=quad_variance,
                                theta_reference_order=reference_order,
                                thresholds=tolerances.tolist(),tested=tested,
                                probe_points=points_count,accepted_candidates=candidates,
                                reason='converged' if converged else 'bounded_theta_fallback'))
    return selected,quad_variance
