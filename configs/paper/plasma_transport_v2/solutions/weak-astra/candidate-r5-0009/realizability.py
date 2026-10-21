"""Necessary degree4 moment realizability checks, independent of any candidate."""
from __future__ import annotations
import numpy as np

def powers(degree):
    return tuple((d-b,b) for d in range(degree+1) for b in range(d+1))

PAIRS=powers(4)
IDS={pair:i for i,pair in enumerate(PAIRS)}
PSI=powers(2)
GRAM_INDEX=np.asarray([[IDS[a+c,b+d] for c,d in PSI] for a,b in PSI])


def gram(raw,normalized=False):
    """Moment axis first; return (...,6,6) for psi=(1,x,y,x²,xy,y²)."""
    raw=np.asarray(raw,float)
    if raw.shape[0]<15:raise ValueError('degree4 raw moments required')
    value=np.moveaxis(raw[GRAM_INDEX],(0,1),(-2,-1))
    if normalized:
        diagonal=np.diagonal(value,axis1=-2,axis2=-1)
        floor=1e-14*np.maximum(np.abs(raw[0]),1.)[...,None]
        scale=np.sqrt(np.maximum(np.abs(diagonal),floor))
        value=value/scale[...,None,:]/scale[...,:,None]
    return value


def diagnose(raw,tolerance=1e-10):
    raw=np.asarray(raw,float)
    eig=np.linalg.eigvalsh(gram(raw));normalized_eig=np.linalg.eigvalsh(gram(raw,True))
    r=raw[0];denom=np.maximum(r,1e-15)
    pxx=raw[3]-raw[1]**2/denom;pxy=raw[4]-raw[1]*raw[2]/denom;pyy=raw[5]-raw[2]**2/denom
    pressure_min=.5*(pxx+pyy-np.sqrt((pxx-pyy)**2+4*pxy**2))
    return {'raw_min_eigenvalue':float(eig.min()),'normalized_min_eigenvalue':float(normalized_eig.min()),
            'negative_gram_fraction':float(np.mean(normalized_eig[...,0]<-tolerance)),
            'min_density':float(r.min()),'min_pressure_eigenvalue':float(pressure_min.min()),
            'states':int(r.size)}


def maximal_safe_blend(prior,delta,tolerance=1e-10,iterations=40):
    """Global raw-state blend, only if the supplied prior is already PSD."""
    prior=np.asarray(prior,float);delta=np.asarray(delta,float)
    if prior.shape!=delta.shape:raise ValueError('matching raw moment shapes required')
    if np.min(np.linalg.eigvalsh(gram(prior,True))) < -tolerance:
        raise ValueError('cannot use an indefinite prior as a realizability anchor')
    if np.min(np.linalg.eigvalsh(gram(prior+delta,True)))>=-tolerance:return 1.
    lower=0.;upper=1.
    for _ in range(iterations):
        mid=.5*(lower+upper)
        if np.min(np.linalg.eigvalsh(gram(prior+mid*delta,True)))>=-tolerance:lower=mid
        else:upper=mid
    return lower
