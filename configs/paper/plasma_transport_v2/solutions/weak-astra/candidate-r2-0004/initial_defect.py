"""Well-balanced initial-distribution source defect, with no Oracle access.

If L(f)=-v.grad_x f-E[f].grad_v f+nu div_v(vf+theta grad_v f), define
S0=-L(f0). Use S_eff(t)=S(t)-S0_fine(x,v)+S0_discrete_coarse(x,v).
The fine source is evaluated as a continuous spatial Fourier polynomial plus
a pointwise force product; it is never low-pass truncated to the coarse grid.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.fft import ifft2
from kinetic_evolution import KineticEvolution


def _project(a):
    nx,ny=a.shape[:2]
    if nx%2==0:a[nx//2]=0
    if ny%2==0:a[:,ny//2]=0
    return a


def fourier_grid(modes,shape):
    """Evaluate EVERY fine Fourier frequency on a coarser uniform mesh.

    Frequencies fold modulo the requested grid size. This is point evaluation,
    deliberately different from Fourier low-pass resizing.
    """
    modes=np.asarray(modes);nx,ny=modes.shape[:2];mx,my=map(int,shape)
    kx=np.rint(np.fft.fftfreq(nx)*nx).astype(int)
    ky=np.rint(np.fft.fftfreq(ny)*ny).astype(int)
    out=np.zeros((mx,my)+modes.shape[2:],dtype=modes.dtype)
    for ix,k in enumerate(kx):
        for iy,l in enumerate(ky):out[k%mx,l%my]+=modes[ix,iy]
    return (ifft2(out,axes=(0,1))*(mx*my)).real


class InitialDefect:
    """Fine initial Fourier representation and its stationary source.

    CPU derivatives are intentional: inputs and public source samples are host
    arrays. No CUDA context or Oracle session is created here.
    """
    def __init__(self,f0,periods,velocity,center,nu,theta,*,dtype=torch.float32,
                 deadline_check=None):
        self.check=deadline_check or (lambda:None);self.check()
        self.periods=np.asarray(periods,float);self.velocity=np.asarray(velocity,float)
        self.center=np.asarray(center,float);self.nu=float(nu);self.theta=float(theta)
        self.dtype=dtype;self.coarse_cache={}
        if len(self.velocity)<4 or not np.allclose(np.diff(self.velocity),np.diff(self.velocity)[0]):raise ValueError('uniform velocity grid required')
        self.dv=float(self.velocity[1]-self.velocity[0]);nv=len(self.velocity)
        f=torch.as_tensor(f0,dtype=dtype,device='cpu')
        if f.ndim!=4 or f.shape[2:]!=(nv,nv) or not bool(torch.isfinite(f).all()):raise ValueError('finite f0[nx,ny,nv,nv] required')
        nx,ny=f.shape[:2];self.shape=(nx,ny)
        h=_project(torch.fft.fft2(f,dim=(0,1)))
        kx=2*math.pi*torch.fft.fftfreq(nx,d=self.periods[0]/nx,dtype=dtype)
        ky=2*math.pi*torch.fft.fftfreq(ny,d=self.periods[1]/ny,dtype=dtype)
        k2=kx[:,None]**2+ky[None,:]**2
        inv=torch.where(k2>0,1/torch.clamp(k2,min=1e-30),torch.zeros_like(k2))
        if nx%2==0:kx[nx//2]=0
        if ny%2==0:ky[ny//2]=0
        v=torch.as_tensor(self.velocity,dtype=dtype)
        vx=v[None,None,:,None]+float(center[0]);vy=v[None,None,None,:]+float(center[1])
        q=2*math.pi*torch.fft.fftfreq(nv,d=self.dv,dtype=dtype);q2=q[:,None]**2+q[None,:]**2
        q1=q.clone()
        if nv%2==0:q1[nv//2]=0
        hv=torch.fft.fft2(h,dim=(-2,-1))
        dx=torch.fft.ifft2(1j*q1[None,None,:,None]*hv,dim=(-2,-1))
        dy=torch.fft.ifft2(1j*q1[None,None,None,:]*hv,dim=(-2,-1))
        collision=self.nu*torch.fft.ifft2(1j*q1[None,None,:,None]*torch.fft.fft2(vx*h,dim=(-2,-1))
             +1j*q1[None,None,None,:]*torch.fft.fft2(vy*h,dim=(-2,-1))-self.theta*q2[None,None]*hv,dim=(-2,-1))
        linear=1j*(kx[:,None,None,None]*vx+ky[None,:,None,None]*vy)*h-collision
        rho=h.sum(dim=(-2,-1))*self.dv**2
        electric=torch.stack([-1j*kx[:,None]*inv*rho,-1j*ky[None,:]*inv*rho],dim=-1)
        scale=1./(nx*ny)
        self.initial_modes=(h*scale).contiguous().numpy()
        self.linear_modes=(_project(linear)*scale).contiguous().numpy()
        self.gradient_modes=tuple((_project(a)*scale).contiguous().numpy() for a in (dx,dy))
        self.electric_modes=(electric*scale).contiguous().numpy()
        self.wave=(2*np.pi*np.fft.fftfreq(nx,d=self.periods[0]/nx),2*np.pi*np.fft.fftfreq(ny,d=self.periods[1]/ny))
        self.check()

    def _basis(self,xy):
        bx=np.exp(1j*xy[:,0,None]*self.wave[0]);by=np.exp(1j*xy[:,1,None]*self.wave[1])
        return (bx[:,:,None]*by[:,None,:]).reshape(len(xy),-1).astype(self.linear_modes.dtype)

    def evaluate_sites(self,sites,batch=16):
        """Return S0 at xy or txy sites, on the complete velocity lattice."""
        sites=np.asarray(sites,float)
        if sites.ndim!=2 or sites.shape[1] not in (2,3) or not np.isfinite(sites).all():raise ValueError('finite xy or txy rows required')
        xy=sites[:,-2:];nv=len(self.velocity);out=[]
        lin=self.linear_modes.reshape(-1,nv*nv)
        gx=self.gradient_modes[0].reshape(-1,nv*nv);gy=self.gradient_modes[1].reshape(-1,nv*nv)
        ee=self.electric_modes.reshape(-1,2)
        for start in range(0,len(xy),batch):
            self.check();b=self._basis(xy[start:start+batch]);e=(b@ee).real
            value=(b@lin).real+e[:,0,None]*(b@gx).real+e[:,1,None]*(b@gy).real
            out.append(value.reshape(-1,nv,nv))
        return np.concatenate(out) if out else np.empty((0,nv,nv),self.linear_modes.real.dtype)

    def source_grid(self,nx,ny=None):
        """Full point-sampled S0; all fine frequencies contribute by folding."""
        if ny is None:ny=nx
        self.check();shape=(nx,ny)
        linear=fourier_grid(self.linear_modes,shape)
        e=fourier_grid(self.electric_modes,shape)
        linear+=e[:,:,0,None,None]*fourier_grid(self.gradient_modes[0],shape)
        linear+=e[:,:,1,None,None]*fourier_grid(self.gradient_modes[1],shape)
        self.check();return linear

    def initial_grid(self,nx,ny=None):
        """Continuous point samples of the fine initial representation."""
        if ny is None:ny=nx
        self.check();return fourier_grid(self.initial_modes,(nx,ny))

    def coarse(self,nx,ny=None):
        """Cached matching initial state, continuous source, discrete source."""
        if ny is None:ny=nx
        shape=(int(nx),int(ny))
        self.check()
        if shape not in self.coarse_cache:
            initial=self.initial_grid(*shape)
            fine_source=self.source_grid(*shape)
            discrete_source=coarse_stationary_source(initial,self.periods,self.velocity,
                self.center,self.nu,self.theta,dtype=self.dtype,deadline_check=self.check)
            self.coarse_cache[shape]=(initial,fine_source,discrete_source)
        self.check();return self.coarse_cache[shape]


def coarse_stationary_source(initial,periods,velocity,center,nu,theta,*,
                             dtype=torch.float32,deadline_check=None):
    """Exactly negate the existing coarse kinetic operator, including force.

    This uses its actual Nyquist projection and3/2-dealiased force product.
    The supplied coarse initial distribution is not replaced or altered here.
    """
    f=np.asarray(initial);zero=np.zeros((2,)+f.shape,dtype=np.float32 if dtype==torch.float32 else np.float64)
    solver=KineticEvolution(f,zero,[0.,1.],periods,velocity,center,nu,theta,
        device='cpu',dtype=dtype,deadline_check=deadline_check)
    if deadline_check is not None:deadline_check()
    sourceh=-(solver.linear*solver.h+solver.nonlinear(solver.h,0.))
    result=torch.fft.ifft2(sourceh,dim=(0,1)).real.numpy()
    if deadline_check is not None:deadline_check()
    return result

