"""Torch Fourier kinetic evolution from prepared continuous-source samples.

No Oracle or query access.  The state is Fourier in periodic laboratory x,y
and collocated on a solver-owned uniform relative-velocity box w=v-center.
ETDRK4 treats laboratory free streaming exactly and advances conservative
spectral force/Ornstein--Uhlenbeck terms explicitly.  Stationary cancellation
between streaming and a manufactured source is preserved by ETDRK4.
"""
from __future__ import annotations
import math
import numpy as np
import torch


def _project_spatial(a):
    nx,ny=a.shape[:2]
    if nx%2==0:a[nx//2]=0
    if ny%2==0:a[:,ny//2]=0
    return a


def _resize_spatial(a,shape):
    nx,ny=a.shape[:2];mx,my=shape
    out=torch.zeros((mx,my)+a.shape[2:],dtype=a.dtype,device=a.device)
    px=min((nx+1)//2,(mx+1)//2);nxneg=min((nx-1)//2,(mx-1)//2)
    py=min((ny+1)//2,(my+1)//2);nyneg=min((ny-1)//2,(my-1)//2)
    out[:px,:py]=a[:px,:py]
    if nyneg:out[:px,-nyneg:]=a[:px,-nyneg:]
    if nxneg:out[-nxneg:,:py]=a[-nxneg:,:py]
    if nxneg and nyneg:out[-nxneg:,-nyneg:]=a[-nxneg:,-nyneg:]
    out*=float(mx*my)/(nx*ny)
    return out


def _phi(z,order):
    """Stable entire phi_j, evaluated in complex128 before coefficient cast."""
    safe=torch.where(torch.abs(z)<.25,torch.ones_like(z),z)
    numerator=torch.expm1(z)
    for j in range(1,order):numerator-=z**j/math.factorial(j)
    direct=numerator/safe**order
    series=torch.zeros_like(z)
    term=torch.full_like(z,1/math.factorial(order))
    for j in range(18):
        series+=term
        term*=z/(order+j+1)
    return torch.where(torch.abs(z)<.25,series,direct)


class KineticEvolution:
    def __init__(self,initial_f,source,times,periods,velocity,center,nu,theta,
                 *,device='cpu',dtype=None,dealias=True,conserve_mean_mass=True,
                 deadline_check=None):
        self.device=torch.device(device)
        if dtype is None:dtype=torch.float32 if self.device.type=='cuda' else torch.float64
        self.dtype=dtype;self.cdtype=torch.complex64 if dtype==torch.float32 else torch.complex128
        self.check=deadline_check or (lambda:None);self.check()
        self.times=np.asarray(times,float)
        if self.times.ndim!=1 or len(self.times)<2 or np.any(np.diff(self.times)<=0):raise ValueError('ordered source times required')
        self.nu=float(nu);self.theta=float(theta)
        if self.nu<0 or self.theta<=0:raise ValueError('nonnegative nu and positive theta required')
        if isinstance(velocity,tuple):vx,vy=velocity
        else:vx=vy=velocity
        vx=np.asarray(vx,float);vy=np.asarray(vy,float)
        if min(len(vx),len(vy))<4:raise ValueError('at least four velocity nodes per axis')
        dx=float(vx[1]-vx[0]);dy=float(vy[1]-vy[0])
        if dx<=0 or dy<=0 or not np.allclose(np.diff(vx),dx) or not np.allclose(np.diff(vy),dy):raise ValueError('uniform ordered velocity axes required')
        self.dv=dx*dy;self.center=np.asarray(center,float)
        ff=torch.as_tensor(initial_f,dtype=dtype,device=self.device)
        if ff.ndim!=4 or ff.shape[-2:]!=(len(vx),len(vy)):raise ValueError('initial_f must have shape nx,ny,nvx,nvy')
        self.nx,self.ny,self.nvx,self.nvy=ff.shape
        if not bool(torch.isfinite(ff).all()):raise ValueError('nonfinite initial distribution')
        self.h=_project_spatial(torch.fft.fft2(ff,dim=(0,1)))
        self.check()
        st=torch.as_tensor(source,dtype=dtype,device=self.device)
        if st.shape!=(len(self.times),)+tuple(ff.shape) or not bool(torch.isfinite(st).all()):raise ValueError('source shape or finite-value failure')
        self.source=torch.fft.fft2(st,dim=(1,2)).to(self.cdtype)
        if self.nx%2==0:self.source[:,self.nx//2]=0
        if self.ny%2==0:self.source[:,:,self.ny//2]=0
        del st,ff
        self.wx=torch.as_tensor(vx,dtype=dtype,device=self.device)[None,None,:,None]
        self.wy=torch.as_tensor(vy,dtype=dtype,device=self.device)[None,None,None,:]
        self.vx=self.wx+float(center[0]);self.vy=self.wy+float(center[1])
        kx=2*math.pi*torch.fft.fftfreq(self.nx,d=float(periods[0])/self.nx,device=self.device,dtype=dtype)
        ky=2*math.pi*torch.fft.fftfreq(self.ny,d=float(periods[1])/self.ny,device=self.device,dtype=dtype)
        k2=kx[:,None]**2+ky[None,:]**2
        self.invk2=torch.where(k2>0,1/torch.clamp(k2,min=1e-30),torch.zeros_like(k2))
        self.kx=kx.clone();self.ky=ky.clone()
        if self.nx%2==0:self.kx[self.nx//2]=0
        if self.ny%2==0:self.ky[self.ny//2]=0
        self.linear=(-1j*(kx[:,None,None,None]*self.vx+ky[None,:,None,None]*self.vy)).to(self.cdtype)
        qx=2*math.pi*torch.fft.fftfreq(self.nvx,d=dx,device=self.device,dtype=dtype)
        qy=2*math.pi*torch.fft.fftfreq(self.nvy,d=dy,device=self.device,dtype=dtype)
        self.q2=qx[:,None]**2+qy[None,:]**2
        self.qx=qx.clone();self.qy=qy.clone()
        if self.nvx%2==0:self.qx[self.nvx//2]=0
        if self.nvy%2==0:self.qy[self.nvy//2]=0
        self.qmax=(float(torch.max(torch.abs(qx))),float(torch.max(torch.abs(qy))))
        self.velocity_bound=(float(torch.max(torch.abs(self.vx))),float(torch.max(torch.abs(self.vy))))
        self.dealias=bool(dealias);self.padshape=(int(math.ceil(1.5*self.nx)),int(math.ceil(1.5*self.ny)))
        self.initial_raw=self.raw_moments(self.h)
        self.mass_removed=[]
        if conserve_mean_mass:
            profile=self.h[0,0].real/(self.nx*self.ny)
            profile=profile/(profile.sum()*self.dv)
            for it in range(len(self.times)):
                mass=self.source[it,0,0].sum()*self.dv
                self.mass_removed.append(float((mass/(self.nx*self.ny)).real))
                self.source[it,0,0]-=mass*profile
        self.coefficients={};self.check()

    def electric_modes(self,h):
        rho=h.sum(dim=(-2,-1))*self.dv
        return (-1j*self.kx[:,None]*self.invk2*rho,
                -1j*self.ky[None,:]*self.invk2*rho)

    def source_at(self,t):
        # Four-point local cubic interpolation, including one-sided endpoints.
        n=min(4,len(self.times));left=int(np.searchsorted(self.times,t,side='right')-1)
        start=min(max(left-1,0),len(self.times)-n);indices=range(start,start+n)
        out=torch.zeros_like(self.h)
        for i in indices:
            weight=1.
            for j in indices:
                if i!=j:weight*=(t-self.times[j])/(self.times[i]-self.times[j])
            if weight:out.add_(self.source[i],alpha=float(weight))
        return out

    def nonlinear(self,h,t):
        fv=torch.fft.fft2(h,dim=(-2,-1))
        dx=torch.fft.ifft2(1j*self.qx[None,None,:,None]*fv,dim=(-2,-1))
        dy=torch.fft.ifft2(1j*self.qy[None,None,None,:]*fv,dim=(-2,-1))
        if self.nu:
            # Conservative divergence form preserves all local velocity mass.
            fx=torch.fft.fft2(self.vx*h,dim=(-2,-1))
            fy=torch.fft.fft2(self.vy*h,dim=(-2,-1))
            collision=self.nu*torch.fft.ifft2(1j*self.qx[None,None,:,None]*fx+1j*self.qy[None,None,None,:]*fy-self.theta*self.q2[None,None]*fv,dim=(-2,-1))
        else:collision=torch.zeros_like(h)
        ex,ey=self.electric_modes(h)
        if self.dealias:
            xp=torch.fft.ifft2(_resize_spatial(dx,self.padshape),dim=(0,1)).real
            yp=torch.fft.ifft2(_resize_spatial(dy,self.padshape),dim=(0,1)).real
            ep=torch.fft.ifft2(_resize_spatial(torch.stack([ex,ey],dim=-1),self.padshape),dim=(0,1)).real
            force=torch.fft.fft2(ep[:,:,0,None,None]*xp+ep[:,:,1,None,None]*yp,dim=(0,1))
            force=_resize_spatial(force,(self.nx,self.ny))
        else:
            xp=torch.fft.ifft2(dx,dim=(0,1)).real;yp=torch.fft.ifft2(dy,dim=(0,1)).real
            ex=torch.fft.ifft2(ex).real;ey=torch.fft.ifft2(ey).real
            force=torch.fft.fft2(ex[:,:,None,None]*xp+ey[:,:,None,None]*yp,dim=(0,1))
        return _project_spatial(collision-force+self.source_at(t))

    def operators(self,dt):
        key=round(float(dt),14)
        if key not in self.coefficients:
            z=(float(dt)*self.linear).to(torch.complex128)
            p1=_phi(z,1);p2=_phi(z,2);p3=_phi(z,3)
            coeff=(torch.exp(z),torch.exp(z/2),float(dt)*.5*_phi(z/2,1),
                   float(dt)*(p1-3*p2+4*p3),float(dt)*(p2-2*p3),float(dt)*(-p2+4*p3))
            self.coefficients[key]=tuple(v.to(self.cdtype) for v in coeff)
            # Only a few step sizes are useful; bound memory if adaptive counts vary.
            if len(self.coefficients)>4:self.coefficients.pop(next(iter(self.coefficients)))
        return self.coefficients[key]

    def step(self,h,t,dt):
        ee,e2,q,f1,f2,f3=self.operators(dt)
        n0=self.nonlinear(h,t)
        a=e2*h+q*n0;na=self.nonlinear(a,t+dt/2)
        b=e2*h+q*na;nb=self.nonlinear(b,t+dt/2)
        c=e2*a+q*(2*nb-n0);nc=self.nonlinear(c,t+dt)
        return _project_spatial(ee*h+f1*n0+2*f2*(na+nb)+f3*nc)

    def raw_moments(self,h):
        wx=self.wx if hasattr(self,'wx') else None
        if wx is None:raise RuntimeError('velocity weights not initialized')
        wy=self.wy;wx=wx[0,0];wy=wy[0,0]
        weights=torch.stack(torch.broadcast_tensors(torch.ones_like(wx+wy),wx,wy,wx*wx,wx*wy,wy*wy,.5*wx*(wx*wx+wy*wy),.5*wy*(wx*wx+wy*wy)))
        mh=torch.einsum('xyij,cij->cxy',h,weights.to(h.dtype))*self.dv
        return torch.fft.ifft2(mh).real

    def stable_step(self,h):
        ex,ey=self.electric_modes(h)
        emax=(float(torch.fft.ifft2(ex).real.abs().max()),float(torch.fft.ifft2(ey).real.abs().max()))
        qx,qy=self.qmax;vx,vy=self.velocity_bound
        rate=self.nu*(vx*qx+vy*qy+self.theta*(qx*qx+qy*qy))+emax[0]*qx+emax[1]*qy
        return 1./max(rate,1e-12)

    def run(self,output_times=None,max_step=None,reject_invalid=False):
        if output_times is None:output_times=np.linspace(self.times[0],self.times[-1],129)
        output_times=np.asarray(output_times,float)
        if output_times[0]!=self.times[0] or output_times[-1]!=self.times[-1] or np.any(np.diff(output_times)<=0):raise ValueError('output times must span full source domain')
        if max_step is None:max_step=(self.times[-1]-self.times[0])/128
        h=self.h;t=float(self.times[0]);snapshots=[self.initial_raw.cpu().numpy()]
        min_density=float(self.initial_raw[0].min());min_pressure=float('inf');steps=0
        boundary_fraction=0.;minimum_f=float('inf')
        for target in output_times[1:]:
            step_limit=min(float(max_step),self.stable_step(h))
            count=max(1,int(math.ceil((float(target)-t)/step_limit-1e-12)))
            dt=(float(target)-t)/count
            for j in range(count):
                self.check();h=self.step(h,t,dt);t+=dt;steps+=1
            t=float(target);raw=self.raw_moments(h)
            if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(h).all()):raise FloatingPointError('nonfinite full kinetic state')
            rho=raw[0];safe=torch.clamp(rho,min=1e-12)
            pxx=raw[3]-raw[1]**2/safe;pxy=raw[4]-raw[1]*raw[2]/safe;pyy=raw[5]-raw[2]**2/safe
            eig=.5*(pxx+pyy-torch.sqrt((pxx-pyy)**2+4*pxy**2))
            min_density=min(min_density,float(rho.min()));min_pressure=min(min_pressure,float(eig.min()))
            if reject_invalid and (float(rho.min())<=0 or float(eig.min())<0):raise FloatingPointError('nonrealizable kinetic macro snapshot')
            snapshots.append(raw.cpu().numpy())
        self.check()
        physical=torch.fft.ifft2(h,dim=(0,1)).real
        minimum_f=float(physical.min())
        absolute=physical.abs();total=float(absolute.sum())
        edge=absolute[:,:,0,:].sum()+absolute[:,:,-1,:].sum()+absolute[:,:,:,0].sum()+absolute[:,:,:,-1].sum()
        boundary_fraction=float(edge)/max(total,1e-30)
        moments=np.asarray(snapshots)
        correction=np.fft.fft2(moments-moments[0:1])/(self.nx*self.ny)
        return {'times':output_times,'moments':moments,'correction_modes':correction,
                'diagnostics':{'steps':steps,'minimum_density':min_density,
                  'minimum_pressure_eigenvalue':min_pressure,'minimum_distribution_final':minimum_f,
                  'final_boundary_l1_fraction':boundary_fraction,'removed_source_mean_mass':self.mass_removed,
                  'device':str(self.device),'dtype':str(self.dtype)}}


def evolve(initial_f,source,times,periods,velocity,center,nu,theta,*,device=None,dtype=None,
           output_times=None,max_step=None,dealias=True,conserve_mean_mass=True,
           deadline_check=None,reject_invalid=False):
    if device is None:device='cuda' if torch.cuda.is_available() else 'cpu'
    solver=KineticEvolution(initial_f,source,times,periods,velocity,center,nu,theta,
              device=device,dtype=dtype,dealias=dealias,conserve_mean_mass=conserve_mean_mass,
              deadline_check=deadline_check)
    return solver.run(output_times=output_times,max_step=max_step,reject_invalid=reject_invalid)


