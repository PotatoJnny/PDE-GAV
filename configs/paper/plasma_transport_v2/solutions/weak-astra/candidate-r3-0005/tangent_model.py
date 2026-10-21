"""Exact VPFP initial moment tangent and bandwise bounded transport forecast.

All moments use w=v-c with one constant public bulk center c. No Oracle access
occurs in this module. The forecast is an initial-value model, not a PDE solve.
"""
from __future__ import annotations

import numpy as np

PAIRS = tuple((d-j, j) for d in range(5) for j in range(d+1))
INDEX = {pair:k for k,pair in enumerate(PAIRS)}


def resize_source_grid(source,shape):
    """Continuous Fourier projection onto a target spatial lattice.

    Source Nyquist planes are discarded because their off-grid sine/cosine
    continuation is ambiguous. Retained normalized coefficients are copied.
    """
    a=np.asarray(source);old=a.shape[-2:];shape=tuple(map(int,shape))
    if old==shape:return a
    modes=np.fft.fft2(a)/np.prod(old);out=np.zeros(a.shape[:-2]+shape,complex)
    fx=np.rint(np.fft.fftfreq(old[0])*old[0]).astype(int)
    fy=np.rint(np.fft.fftfreq(old[1])*old[1]).astype(int)
    for ix,kx in enumerate(fx):
        if (old[0]%2==0 and abs(kx)==old[0]//2) or abs(kx)>=(shape[0]+1)//2:continue
        for iy,ky in enumerate(fy):
            if (old[1]%2==0 and abs(ky)==old[1]//2) or abs(ky)>=(shape[1]+1)//2:continue
            out[...,kx%shape[0],ky%shape[1]]=modes[...,ix,iy]
    return np.fft.ifft2(out*np.prod(shape)).real


def triangular_weights(velocity, degree=4):
    v=np.asarray(velocity, float)
    dx=float(v[1]-v[0])
    vx,vy=np.meshgrid(v,v,indexing='ij')
    return np.asarray([(vx**i*vy**j).ravel() for i,j in PAIRS if i+j<=degree])*dx*dx


def waves(shape, periods):
    return np.meshgrid(*(2*np.pi*np.fft.fftfreq(n,d=l/n) for n,l in zip(shape,periods)),indexing='ij')


def derivative(a,periods,axis):
    k=waves(a.shape[-2:],periods)[axis]
    n=a.shape[-2+axis]
    if n%2==0:
        if axis==0:k[n//2,:]=0.
        else:k[:,n//2]=0.
    return np.fft.ifft2(1j*k*np.fft.fft2(a)).real


def electric(rho,periods):
    kx,ky=waves(rho.shape,periods);k2=kx*kx+ky*ky
    phi=np.divide(np.fft.fft2(rho),k2,out=np.zeros(rho.shape,complex),where=k2>0)
    if rho.shape[0]%2==0:kx[rho.shape[0]//2,:]=0.
    if rho.shape[1]%2==0:ky[:,rho.shape[1]//2]=0.
    return np.asarray([np.fft.ifft2(-1j*k*phi).real for k in (kx,ky)])


def raw_tangent(moments,source,periods,center,nu,theta):
    """Return the degree<=3 moment tangent from degree<=4 initial moments."""
    m=np.asarray(moments,float);s=np.asarray(source,float)
    if m.shape[0]!=15 or s.shape!=(10,)+m.shape[1:]:raise ValueError('triangular degree4 moments and degree3 source required')
    e=electric(m[0],periods);zero=np.zeros_like(m[0]);out=[]
    def get(a,b):return zero if min(a,b)<0 else m[INDEX[a,b]]
    for k,(a,b) in enumerate(PAIRS[:10]):
        mm=m[k]
        value=s[k]-derivative(get(a+1,b)+center[0]*mm,periods,0)-derivative(get(a,b+1)+center[1]*mm,periods,1)
        value=value+a*e[0]*get(a-1,b)+b*e[1]*get(a,b-1)
        value=value-nu*((a+b)*mm+a*center[0]*get(a-1,b)+b*center[1]*get(a,b-1))
        value=value+nu*theta*(a*(a-1)*get(a-2,b)+b*(b-1)*get(a,b-2))
        out.append(value)
    return np.asarray(out)


def central_macros(m,center):
    """Map degree3 raw moments to laboratory j, central P and central heat."""
    r,jx,jy,xx,xy,yy,xxx,xxy,xyy,yyy=m[:10]
    ux=jx/r;uy=jy/r
    pxx=xx-jx*ux;pxy=xy-jx*uy;pyy=yy-jy*uy
    qx=.5*(xxx+xyy)-ux*(1.5*xx+.5*yy)-uy*xy+r*ux*(ux*ux+uy*uy)
    qy=.5*(xxy+yyy)-uy*(.5*xx+1.5*yy)-ux*xy+r*uy*(ux*ux+uy*uy)
    return np.asarray([jx+center[0]*r,jy+center[1]*r,pxx,pxy,pyy,qx,qy])


def central_tangent(m,dm,center):
    """Complex-step application of the exact rational central-moment Jacobian."""
    return central_macros(np.asarray(m[:10],complex)+1e-30j*np.asarray(dm),center).imag/1e-30


class BandForecast:
    def __init__(self,moments,source,periods,center,nu,theta,*,bandwise=True,heat=False,tangent=None):
        self.periods=np.asarray(periods,float);self.shape=moments.shape[-2:]
        initial=central_macros(moments,center)
        raw_derivative=raw_tangent(moments,source,periods,center,nu,theta) if tangent is None else np.asarray(tangent,float)
        if raw_derivative.shape!=(10,)+self.shape or not np.isfinite(raw_derivative).all():raise ValueError('finite compatible raw tangent required')
        tangent=central_tangent(moments,raw_derivative,center)
        self.initial_modes=np.fft.fft2(initial)/np.prod(self.shape)
        self.tangent_modes=np.fft.fft2(tangent)/np.prod(self.shape)
        for axis,n in enumerate(self.shape):
            if n%2==0:
                if axis==0:
                    self.initial_modes[:,n//2,:]=0.;self.tangent_modes[:,n//2,:]=0.
                else:
                    self.initial_modes[:,:,n//2]=0.;self.tangent_modes[:,:,n//2]=0.
        k=np.meshgrid(*(np.abs(np.fft.fftfreq(n)*n) for n in self.shape),indexing='ij')
        radial=np.maximum(*k)
        bounds=[0,2,4,8,np.inf] if bandwise else [0,np.inf]
        self.tau=np.ones((7,)+self.shape)
        self.diagnostics=[]
        for group,metric in (([0,1],[1,1]),([2,3,4],[1,2,1]),([5,6],[1,1])):
            weights=np.asarray(metric)[:,None,None]
            original=self.initial_modes[group];slope=self.tangent_modes[group]
            nonzero=radial>0
            global_energy=float(np.sum(weights*np.abs(original)**2*nonzero))
            global_slope=float(np.sum(weights*np.abs(slope)**2*nonzero))
            global_tau=np.clip(np.sqrt(global_energy/max(global_slope,1e-30)),.1,1.)
            records=[]
            for lower,upper in zip(bounds[:-1],bounds[1:]):
                mask=(radial>lower)&(radial<=upper)
                e=float(np.sum(weights*np.abs(original)**2*mask));d=float(np.sum(weights*np.abs(slope)**2*mask))
                # Weak group-scale ridge avoids unstable ratios in empty shells.
                ridge=max(1e-4*global_energy,1e-30)
                tau=np.clip(np.sqrt((e+ridge)/max(d+ridge/global_tau**2,1e-30)),.1,1.)
                self.tau[group]=np.where(mask,tau,self.tau[group])
                records.append({'band':[lower,upper if np.isfinite(upper) else None],'energy':e,'slope_energy':d,'tau':float(tau)})
            self.diagnostics.append({'group':group,'global_tau':float(global_tau),'bands':records})
        self.heat=bool(heat)
        if heat:
            self.tangent_modes[5:,radial>4]=0.
            # Heat is persistent in the source-forced family; only its slope
            # receives the bounded envelope, rather than damping initial heat.
        self.wave=[2*np.pi*np.fft.fftfreq(n,d=l/n) for n,l in zip(self.shape,periods)]

    def project(self,txy,t0):
        p=np.asarray(txy,float);elapsed=p[:,0]-t0
        bx=np.exp(1j*p[:,1,None]*self.wave[0]);by=np.exp(1j*p[:,2,None]*self.wave[1])
        factor=np.exp(-.5*(elapsed[:,None,None,None]/self.tau[None])**2)
        out=factor*(self.initial_modes[None]+elapsed[:,None,None,None]*self.tangent_modes[None])
        # Means are not spatial fluctuations. Give their initial tangent a
        # bounded envelope while keeping the initial mean exactly.
        out[:,:,0,0]=self.initial_modes[:,0,0]+factor[:,:,0,0]*elapsed[:,None]*self.tangent_modes[:,0,0]
        if self.heat:
            out[:,5:]=self.initial_modes[None,5:]+factor[:,5:]*elapsed[:,None,None,None]*self.tangent_modes[None,5:]
        return np.einsum('ncij,ni,nj->nc',out,bx,by).real

    def heat_increment(self,txy,t0):
        """Only the bounded prepared heat tangent, for adding to a prior."""
        p=np.asarray(txy,float);elapsed=p[:,0]-t0
        bx=np.exp(1j*p[:,1,None]*self.wave[0]);by=np.exp(1j*p[:,2,None]*self.wave[1])
        factor=np.exp(-.5*(elapsed[:,None,None,None]/self.tau[None,5:])**2)
        modes=factor*elapsed[:,None,None,None]*self.tangent_modes[None,5:]
        return np.einsum('ncij,ni,nj->nc',modes,bx,by).real
