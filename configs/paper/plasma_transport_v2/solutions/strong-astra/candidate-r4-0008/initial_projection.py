"""Independent initial velocity quadrature and oversampled Fourier projection.

Only public g0/rho0 evaluations are required. The caller owns the shared
Case clock and request splitting through ``get``. The source quadrature
need not change when this initial rule is refined.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.fft import fft2


def _rule(order, theta, variance, degree):
    z,w=np.polynomial.hermite_e.hermegauss(order)
    v=z*math.sqrt(theta); weights=w*np.exp(z*z/2)*math.sqrt(theta)
    h=np.empty((degree+1,order));h[0]=1.
    if degree:h[1]=v/math.sqrt(variance)
    for k in range(1,degree):
        h[k+1]=(v/math.sqrt(variance)*h[k]-math.sqrt(k)*h[k-1])/math.sqrt(k+1)
    powers=[(a,d-a) for d in range(degree+1) for a in range(d,-1,-1)]
    basis=np.array([(h[a,:,None]*h[b,None,:]*weights[:,None]*weights[None,:]).ravel() for a,b in powers])
    vx,vy=np.meshgrid(v,v,indexing='ij')
    return np.c_[vx.ravel(),vy.ravel()],basis


def initial_quadrature_order(get, *, periods, theta, variance, degree, base_order=24):
    """Refine g0 independently of source, retaining a noise-tolerant guard.

    Sixteen deterministic spatial scouts precede preparation. A known
    normalization check prevents false convergence of badly truncated rules.
    This remains a finite-site heuristic, not an arbitrary-data certificate.
    """
    index=np.arange(1,17)
    xy=np.c_[index*.6180339887498949%1,index*.4142135623730951%1]*np.asarray(periods)
    degrees=np.concatenate([np.full(d+1,d) for d in range(degree+1)])
    tolerances=np.where(degrees<=3,2.e-6,np.where(degrees<=8,8.e-6,3.e-5))
    diagnostics=[];previous=None;previous_mass_error=None
    orders=sorted(set([int(base_order),int(base_order)+8,48,64,80,96]))
    orders=[q for q in orders if q>=base_order]
    for order in orders:
        vel,basis=_rule(order,theta,variance,degree)
        points=np.empty((len(xy),order*order,4))
        points[:,:,:2]=xy[:,None,:];points[:,:,2:]=vel
        raw=np.asarray(get('g0',points.reshape(-1,4)),dtype=float)
        moments=raw.reshape(len(xy),-1)@basis.T
        if not np.isfinite(moments).all():raise ArithmeticError('nonfinite initial quadrature')
        mass_error=float(np.max(abs(moments[:,0]-1.)))
        error=(None if previous is None else float(np.max(
            abs(moments-previous)/(tolerances[None,:]*(1.+abs(moments))))))
        diagnostics.append({'order':order,'normalization_error':mass_error,'relative_guard_error':error})
        if (previous is not None and mass_error<=2.e-6
                and previous_mass_error<=2.e-6 and error<=1.):
            # Preserve the source order if the first independent g0 check
            # passes. If refinement was necessary, retain the fine rule.
            selected=(int(base_order) if len(diagnostics)==2 else order)
            return selected,{'converged':True,'probes':diagnostics}
        previous=moments
        previous_mass_error=mass_error
    return orders[-1],{'converged':False,'probes':diagnostics}


def project_initial(get, *, periods, state_n, theta, variance, degree,
                    order, sample_n=None, max_points=262144, normalize=False):
    """Sample initial moments on a fine mesh and restrict to the state modes.

    Memory is bounded by one public-request spatial block; ``get`` still
    owns the cooperative deadline. Exact density replaces g0's mass moment.
    Normalizing other moments is optional and disabled pending evidence.
    """
    n=int(state_n)
    default_m=(3*n)//2
    if default_m%2==0:default_m+=1
    m=int(sample_n or default_m)
    if n%2!=1 or m%2!=1 or m<n:raise ValueError('ordered odd grids required')
    vel,basis=_rule(order,theta,variance,degree)
    xx,yy=np.meshgrid(np.arange(m)*periods[0]/m,np.arange(m)*periods[1]/m,indexing='ij')
    xy=np.c_[xx.ravel(),yy.ravel()]
    rho=np.asarray(get('rho0',xy),dtype=float)
    moments=np.empty((m*m,basis.shape[0]));block=max(1,int(max_points)//(order*order))
    mass_error=0.
    for first in range(0,len(xy),block):
        part=xy[first:first+block]
        points=np.empty((len(part),order*order,4))
        points[:,:,:2]=part[:,None,:];points[:,:,2:]=vel
        raw=np.asarray(get('g0',points.reshape(-1,4)),dtype=float)
        projected=raw.reshape(len(part),-1)@basis.T
        mass_error=max(mass_error,float(np.max(abs(projected[:,0]-1.))))
        if normalize:
            if np.any(projected[:,0]<=0):raise ArithmeticError('nonpositive g0 normalization')
            projected/=projected[:,0,None]
        moments[first:first+len(part)]=projected*rho[first:first+len(part),None]
    moments[:,0]=rho
    modes=fft2(moments.T.reshape(-1,m,m),axes=(-2,-1))
    if m!=n:
        k=np.rint(np.fft.fftfreq(n)*n).astype(int)%m
        modes=modes[:,k[:,None],k[None,:]]*(n/m)**2
    modes[0,0,0]=n*n
    return modes,{'sample_n':m,'quadrature_order':order,'max_normalization_error':mass_error}


def prepare_initial_state(get, *, periods, state_n, theta, variance, degree,
                          source_order, sample_n=None, max_points=262144):
    """Drop-in state initializer, independent from source spatial/velocity rules."""
    order,quadrature=initial_quadrature_order(
        get,periods=periods,theta=theta,variance=variance,degree=degree,
        base_order=source_order)
    state,projection=project_initial(
        get,periods=periods,state_n=state_n,theta=theta,variance=variance,
        degree=degree,order=order,sample_n=sample_n,max_points=max_points)
    return state,{'quadrature':quadrature,'projection':projection}
