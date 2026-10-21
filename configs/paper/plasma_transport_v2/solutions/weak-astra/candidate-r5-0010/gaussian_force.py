"""Analytic low-degree Gaussian lift for optional nonlinear electric response.

This module has no numerical Oracle or scoring access. It evolves only macro
corrections and uses the measured physical thermal covariance for its closure.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from green_ops import ou_parameters
from moments import project, evaluate_grid, resample_modes, electric_modes

PAIRS=tuple((degree-b,b) for degree in range(5) for b in range(degree+1))
INDEX={pair:i for i,pair in enumerate(PAIRS)}


def _product(left,right):
    result={}
    for (a,b),x in left.items():
        for (c,d),y in right.items():result[a+c,b+d]=result.get((a+c,b+d),0.)+x*y
    return result


class GaussianLift:
    """A degree-three normalized Hermite lift with exact force/OU integration."""
    def __init__(self,covariance,origin_relative,bulk_center,nu,theta):
        self.bulk=np.asarray(bulk_center,float)
        self.origin=np.asarray(origin_relative,float)
        self.mean=self.origin+self.bulk
        self.covariance=np.asarray(covariance,float)
        eigen,self.rotation=np.linalg.eigh(self.covariance)
        if np.min(eigen)<=0 or not np.isfinite(eigen).all():
            raise ValueError('positive Gaussian-lift covariance required')
        self.sigma=np.sqrt(eigen)
        self.basis=self.rotation*self.sigma[None,:]
        self.nu=float(nu);self.theta=float(theta)
        polynomials=[]
        for axis in range(2):
            direction=self.rotation[:,axis]/self.sigma[axis]
            linear={(0,0):-float(direction@self.origin),(1,0):direction[0],(0,1):direction[1]}
            polys=[{(0,0):1.},linear]
            for degree in range(1,3):
                poly=_product(linear,polys[-1])
                for power,coefficient in polys[-2].items():
                    poly[power]=poly.get(power,0.)-np.sqrt(degree)*coefficient
                polys.append({power:coefficient/np.sqrt(degree+1) for power,coefficient in poly.items()})
            polynomials.append(polys)
        self.transform=np.asarray([[_product(polynomials[0][a],polynomials[1][b]).get(pair,0.)
                                     for pair in PAIRS[:10]] for a,b in PAIRS[:10]])

    def coefficients(self,raw):
        """Raw relative moments -> normalized degree-three Hermite coefficients."""
        return np.einsum('ab,b...->a...',self.transform,np.asarray(raw)[:10],optimize=True)

    def force_coefficients(self,coefficients,field):
        """Coefficients of -field.grad_v(lift), through degree four."""
        alpha=np.asarray(coefficients)
        acceleration=np.einsum('ij,i...->j...',self.rotation,np.asarray(field))/self.sigma.reshape((2,)+(1,)*(alpha.ndim-1))
        beta=np.zeros((15,)+alpha.shape[1:],dtype=np.result_type(alpha,field))
        for row,(a,b) in enumerate(PAIRS[:10]):
            beta[INDEX[a+1,b]]+=np.sqrt(a+1)*acceleration[0]*alpha[row]
            beta[INDEX[a,b+1]]+=np.sqrt(b+1)*acceleration[1]*alpha[row]
        return beta

    def kernels(self,tau,wavevectors):
        """OU macro adjoint against each normalized Gaussian Hermite basis.

        Shape is (10,nt,nk,15). Joint Gaussian identities integrate velocity
        continuously, including anisotropy, drift, and mixed third moments.
        """
        tau=np.atleast_1d(np.asarray(tau,float));k=np.asarray(wavevectors,float)
        e,a,var,cross,disp=ou_parameters(tau,self.nu,self.theta)
        kcov=k@self.covariance
        phase=np.exp(-1j*a[:,None]*(k@self.mean)[None]
                     -.5*a[:,None]**2*np.sum(kcov*k,axis=1)[None]
                     -.5*disp[:,None]*np.sum(k*k,axis=1)[None])
        ell=-1j*a[:,None,None]*(k@self.basis)[None]
        mu=(e[:,None,None]*self.mean-self.bulk
            -1j*(e*a)[:,None,None]*kcov[None]-1j*cross[:,None,None]*k[None])
        covariance=e[:,None,None]**2*self.covariance+var[:,None,None]*np.eye(2)
        coupling=e[:,None,None]*self.basis
        raw={}
        zeroth=np.empty((len(tau),len(k),15),complex)
        for row,(p,q) in enumerate(PAIRS):
            zeroth[...,row]=phase*ell[...,0]**p*ell[...,1]**q/math.sqrt(math.factorial(p)*math.factorial(q))
        raw[0,0]=zeroth
        for aout,bout in PAIRS[1:10]:
            axis=0 if aout else 1
            aold,bold=(aout-1,bout) if axis==0 else (aout,bout-1)
            old=raw[aold,bold]
            result=mu[...,axis,None]*old
            if aold:result+=aold*covariance[:,None,axis,0,None]*raw[aold-1,bold]
            if bold:result+=bold*covariance[:,None,axis,1,None]*raw[aold,bold-1]
            for row,(p,q) in enumerate(PAIRS):
                if p:result[...,row]+=np.sqrt(p)*coupling[:,None,axis,0]*old[...,INDEX[p-1,q]]
                if q:result[...,row]+=np.sqrt(q)*coupling[:,None,axis,1]*old[...,INDEX[p,q-1]]
            raw[aout,bout]=result
        return np.stack([raw[pair] for pair in PAIRS[:10]])

    def response(self,source_modes,times,wavevectors,check=lambda:None,*,mode_block=128,quadrature_order=6):
        """Continuous OU convolution of piecewise-linear Hermite source modes.

        source_modes has shape (time,mode,15); output is (time,10,mode).
        Time quadrature and initial half-hat removal match the main Green core.
        """
        source=np.asarray(source_modes);times=np.asarray(times,float);wave=np.asarray(wavevectors,float)
        nt,nm,nv=source.shape
        if nv!=15 or times.shape!=(nt,) or wave.shape!=(nm,2):raise ValueError('invalid Gaussian source axes')
        dt=float(times[1]-times[0])
        if dt<=0 or not np.allclose(np.diff(times),dt):raise ValueError('uniform increasing source times required')
        result=np.zeros((nt,10,nm),complex);nfft=next_fast_len(2*nt-1);lags=np.arange(nt)
        for start in range(0,nm,mode_block):
            check();stop=min(nm,start+mode_block);k=wave[start:stop]
            phase=dt*float(np.max(abs(k@self.mean)))
            order=max(quadrature_order,int(np.ceil(phase/2))+2)
            roots,weights=np.polynomial.legendre.leggauss(order);offsets=(roots+1)/2;weights=weights/2
            source_fft=fft(source[:,start:stop],n=nfft,axis=0,workers=1)
            hats=np.zeros((10,nt,stop-start,15),complex)
            extra=np.zeros((10,nt,stop-start),complex)
            for right in (True,False):
                for offset,weight in zip(offsets,weights):
                    check();tau=(lags+offset if right else np.maximum(lags-1+offset,0))*dt
                    factor=np.full(nt,dt*weight*((1-offset) if right else offset))
                    if not right:factor[0]=0.
                    weighted=self.kernels(tau,k)*factor[None,:,None,None]
                    hats+=weighted
                    if right:extra+=np.einsum('mtkv,kv->mtk',weighted,source[0,start:stop],optimize=True)
            for m in range(10):
                check();kernel_fft=fft(hats[m],n=nfft,axis=0,workers=1)
                answer=ifft(np.einsum('tkv,tkv->tk',kernel_fft,source_fft,optimize=True),axis=0,workers=1)[:nt]
                result[:,m,start:stop]=answer-extra[m]
        result[0]=0.
        if not np.isfinite(result).all():raise FloatingPointError('nonfinite Gaussian force response')
        check();return result


def force_source_modes(lift,raw_initial_modes,delta_modes,used_density_modes,periods,check=lambda:None):
    """Omitted nonlinear force plus coherent repair of linear density feedback.

    ``used_density_modes`` is the delta density that generated the baseline's
    last -deltaE.grad(f0) source. If None, use the current delta density so only
    the omitted force is added. Products are dealiased on a double spatial mesh.
    """
    delta_modes=np.asarray(delta_modes);nt,_,n,_=delta_modes.shape
    initial=evaluate_grid(raw_initial_modes[:10],2*n)
    alpha0=lift.coefficients(initial)
    field0=evaluate_grid(electric_modes(raw_initial_modes[0],periods),2*n)
    forcing=np.empty((nt,n*n,15),complex)
    for frame in range(nt):
        check();delta=evaluate_grid(delta_modes[frame],2*n)
        alpha=lift.coefficients(delta)
        field=evaluate_grid(electric_modes(delta_modes[frame,0],periods),2*n)
        beta=lift.force_coefficients(alpha,field0+field)
        if used_density_modes is not None:
            difference=delta_modes[frame,0]-used_density_modes[frame]
            beta+=lift.force_coefficients(alpha0,evaluate_grid(electric_modes(difference,periods),2*n))
        forcing[frame]=resample_modes(project(beta),n).reshape(15,-1).T
    return forcing


def nonlinear_force_correction(lift,raw_initial_modes,base_delta_modes,used_density_modes,
                               times,periods,taper,check=lambda:None,*,iterations=2):
    """Picard correct the completed weak-field macro state without new samples.

    ``base_delta_modes`` should represent the completed, guarded baseline.
    Return a fresh correction; callers must retain their completed handle when
    the optional check raises. This function never mutates caller-owned arrays.
    """
    base=np.asarray(base_delta_modes);nt,_,n,_=base.shape
    kx,ky=np.meshgrid(*(2*np.pi*np.fft.fftfreq(n,d=l/n) for l in periods),indexing='ij')
    wave=np.stack((kx,ky),-1).reshape(-1,2)
    active=np.flatnonzero(np.asarray(taper).ravel()>=1e-12)
    updated=base.copy();diagnostics=[]
    for iteration in range(iterations):
        check();source=force_source_modes(lift,raw_initial_modes,updated,used_density_modes,periods,check)
        response=lift.response(source[:,active],times,wave[active],check)
        extra=np.zeros((nt,10,n*n),complex);extra[:,:,active]=response
        extra=extra.reshape(base.shape)*taper
        extra[:,0,0,0]=0.;extra[...,n//2,:]=0.;extra[...,:,n//2]=0.
        updated=base+extra
        diagnostics.append({'iteration':iteration+1,'rms_added_mode':np.sqrt(np.mean(abs(extra)**2,axis=(0,2,3))).tolist()})
    check();return updated,diagnostics
