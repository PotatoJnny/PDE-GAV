"""Analytic OU response of a low-order Gaussian/Hermite velocity lift.

Coefficients are integrals against orthonormal probabilists' Hermites.
The lift's covariance is a numerical closure choice, not a fitted truth model.
"""
from __future__ import annotations
import math
import numpy as np
from green_ops import ou_parameters, PAIRS

SOURCE_PAIRS=tuple((degree-b,b) for degree in range(5) for b in range(degree+1))
SOURCE_INDEX={pair:i for i,pair in enumerate(SOURCE_PAIRS)}


def _product(left,right):
    answer={}
    for (a,b),x in left.items():
        for (c,d),y in right.items():answer[a+c,b+d]=answer.get((a+c,b+d),0.)+x*y
    return answer


class GaussianHermiteOUKernel:
    """Exact degree-three macro response to degree-four Hermite sources."""
    def __init__(self,covariance,physical_center,bulk_center,nu,theta):
        self.covariance=np.asarray(covariance,float)
        self.center=np.asarray(physical_center,float)
        self.bulk=np.asarray(bulk_center,float)
        self.nu=float(nu);self.theta=float(theta)
        eigen,rotation=np.linalg.eigh(self.covariance)
        if (np.min(eigen)<=0 or self.center.shape!=(2,) or self.bulk.shape!=(2,)
                or nu<0 or theta<=0):raise ValueError('positive Gaussian covariance required')
        self.transform=rotation*np.sqrt(eigen)[None,:]
        self.inverse=np.linalg.inv(self.transform)
        self.normalization=np.asarray([math.sqrt(math.factorial(a)*math.factorial(b)) for a,b in SOURCE_PAIRS])
        origin=self.center-self.bulk
        polynomials=[]
        for row in self.inverse:
            linear={(0,0):-float(row@origin),(1,0):row[0],(0,1):row[1]}
            polys=[{(0,0):1.},linear]
            for degree in range(1,3):
                nxt=_product(linear,polys[-1])
                for power,value in polys[-2].items():nxt[power]=nxt.get(power,0.)-np.sqrt(degree)*value
                polys.append({power:value/np.sqrt(degree+1) for power,value in nxt.items()})
            polynomials.append(polys)
        self.raw_to_hermite=np.asarray([[_product(polynomials[0][a],polynomials[1][b]).get(pair,0.)
                                         for pair in PAIRS] for a,b in PAIRS])
        # Used only by green_response's temporal-phase quadrature planner.
        # It is not an integration rule; the exact kernel integrates velocity.
        angles=2*np.pi*np.arange(len(SOURCE_PAIRS))/len(SOURCE_PAIRS)
        self.phase_probes=self.center+6*np.stack((np.cos(angles),np.sin(angles)),1)@self.transform.T

    def lift(self,raw):
        return np.einsum('ab,b...->a...',self.raw_to_hermite,np.asarray(raw)[:10],optimize=True)

    def force(self,hermite,field):
        """Hermite coefficients of -field.grad_v(lift), with + force sign."""
        hermite=np.asarray(hermite);field=np.asarray(field)
        acceleration=np.einsum('ab,b...->a...',self.inverse,field,optimize=True)
        out=np.zeros((len(SOURCE_PAIRS),)+hermite.shape[1:],dtype=np.result_type(hermite,field))
        for row,(a,b) in enumerate(PAIRS):
            out[SOURCE_INDEX[a+1,b]]+=np.sqrt(a+1)*acceleration[0]*hermite[row]
            out[SOURCE_INDEX[a,b+1]]+=np.sqrt(b+1)*acceleration[1]*hermite[row]
        return out

    def iter_kernel_terms(self,tau,wavevectors,*,density_only=False):
        tau=np.atleast_1d(np.asarray(tau,float));k=np.asarray(wavevectors,float)
        r,A,V,C,W=ou_parameters(tau,self.nu,self.theta)
        shape=(len(tau),len(k))
        rotated=k@self.transform
        zmean=-1j*A[:,None,None]*rotated[None]
        covk=k@self.covariance
        wmean=r[:,None,None]*self.center[None,None]-self.bulk[None,None]
        wmean=wmean-1j*(r*A)[:,None,None]*covk[None]-1j*C[:,None,None]*k[None]
        covw=r[:,None,None]**2*self.covariance[None]+V[:,None,None]*np.eye(2)[None]
        covzw=r[:,None,None]*self.transform.T[None]
        phase=np.exp(-1j*A[:,None]*(k@self.center)[None]
                     -.5*A[:,None]**2*np.sum(covk*k,axis=1)[None]
                     -.5*W[:,None]*np.sum(k*k,axis=1)[None])
        cache={}
        def moment(a,b,p,q):
            key=(a,b,p,q)
            if key in cache:return cache[key]
            if p==q==0:out=zmean[:,:,0]**a*zmean[:,:,1]**b
            elif p:
                out=wmean[:,:,0]*moment(a,b,p-1,q)
                if p>1:out=out+(p-1)*covw[:,0,0,None]*moment(a,b,p-2,q)
                if q:out=out+q*covw[:,0,1,None]*moment(a,b,p-1,q-1)
                if a:out=out+a*covzw[:,0,0,None]*moment(a-1,b,p-1,q)
                if b:out=out+b*covzw[:,1,0,None]*moment(a,b-1,p-1,q)
            else:
                out=wmean[:,:,1]*moment(a,b,0,q-1)
                if q>1:out=out+(q-1)*covw[:,1,1,None]*moment(a,b,0,q-2)
                if a:out=out+a*covzw[:,0,1,None]*moment(a-1,b,0,q-1)
                if b:out=out+b*covzw[:,1,1,None]*moment(a,b-1,0,q-1)
            cache[key]=out
            return out
        for p,q in PAIRS[:1 if density_only else 10]:
            yield phase[:,:,None]*np.stack([moment(a,b,p,q)/normalization
                            for (a,b),normalization in zip(SOURCE_PAIRS,self.normalization)],axis=-1)

    def kernel(self,tau,wavevectors,*,density_only=False):
        return np.stack(list(self.iter_kernel_terms(tau,wavevectors,density_only=density_only)))
