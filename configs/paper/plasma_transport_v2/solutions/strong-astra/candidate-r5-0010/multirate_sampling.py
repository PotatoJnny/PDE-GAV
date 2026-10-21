"""Guarded lower-degree source quadrature; no solution observations."""
from __future__ import annotations
import math
import numpy as np


def rule(order, theta, variance, degree=6):
    z,w=np.polynomial.hermite_e.hermegauss(order)
    v=z*math.sqrt(theta)
    weights=w*np.exp(z*z/2)*math.sqrt(theta)
    h=np.empty((degree+1,order));h[0]=1.;h[1]=v/math.sqrt(variance)
    for k in range(1,degree):
        h[k+1]=(v/math.sqrt(variance)*h[k]-math.sqrt(k)*h[k-1])/math.sqrt(k+1)
    powers=[(a,d-a) for d in range(degree+1) for a in range(d,-1,-1)]
    basis=np.array([(h[a,:,None]*h[b,None,:]*weights[:,None]*weights[None,:]).ravel()
                    for a,b in powers])
    vx,vy=np.meshgrid(v,v,indexing='ij')
    return np.c_[vx.ravel(),vy.ravel()],basis


def choose_low_rule(get, pilot_times, pilot_xy, pilot_bank, *, theta, variance,
                    full_order, degree=6, tolerance=5.e-6):
    fractions=np.array([0.,1.e-7,1.e-5,.001,.01,.05,.2,.5,.8,1.])
    targets=pilot_times[0]+fractions*(pilot_times[-1]-pilot_times[0])
    selected=np.unique([np.argmin(abs(pilot_times-t)) for t in targets])
    times=pilot_times[selected];xy=pilot_xy[:8]
    count=(degree+1)*(degree+2)//2
    reference=np.array([pilot_bank[float(t)][:len(xy),:count] for t in times])
    reference=reference.reshape(-1,count)
    def probe(order):
        vel,basis=rule(order,theta,variance,degree)
        p=np.empty((len(times),len(xy),len(vel),5))
        p[:,:,:,0]=times[:,None,None];p[:,:,:,1:3]=xy[None,:,None,:]
        p[:,:,:,3:]=vel
        raw=get('source',p.reshape(-1,5)).reshape(len(times)*len(xy),len(vel))
        return vel,basis,raw@basis.T
    _,_,anchor=probe(full_order+4)
    noise=np.max(abs(anchor-reference),axis=0)
    scale=1+np.maximum(np.max(abs(reference),axis=0),np.max(abs(anchor),axis=0))
    powers=np.repeat(np.arange(degree+1),np.arange(1,degree+2))
    floor=np.where(powers==0,2.e-7,np.where(powers<=3,5.e-7,2.e-6))
    allowed=np.minimum(np.maximum(3*noise,floor),tolerance*scale)
    diagnostics=[{'anchor_order':full_order+4,'noise_max':float(np.max(noise)),
                  'density_noise':float(noise[0]),'density_allowed':float(allowed[0])}]
    for order in sorted(set([min(18,full_order),min(20,full_order),min(22,full_order),full_order])):
        if order==full_order:
            vel,basis=rule(order,theta,variance,degree)
            return order,vel,basis,diagnostics
        vel,basis,value=probe(order)
        defect=np.maximum(np.max(abs(value-reference),axis=0),np.max(abs(value-anchor),axis=0))
        error=float(np.max(defect/allowed))
        diagnostics.append({'order':order,'guard_ratio':error,'density_defect':float(defect[0])})
        if np.isfinite(error) and error<=1.:
            return order,vel,basis,diagnostics
    raise ArithmeticError('empty low quadrature plan')
