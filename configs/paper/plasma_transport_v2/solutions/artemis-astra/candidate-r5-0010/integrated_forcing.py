"""Carrier-exact fourth-order Taylor convolution for accepted source rows."""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import BSpline


class IntegratedCarrier:
    def __init__(self,coefficients,frequency,spline,kx,velocity,tmin,phi):
        self.cell_carriers=frequency.ndim==2
        self.indices=((frequency!=0) if frequency.ndim==1 else (frequency!=0).any(-1)).nonzero().ravel()
        self.frequency=frequency[self.indices]
        self.nt,self.nx,self.nv=coefficients.shape
        self.mask=(self.frequency!=0 if self.cell_carriers else
                   torch.ones((len(self.indices),self.nv),device=coefficients.device,dtype=torch.bool))
        self.rows=coefficients[:,self.indices].reshape(self.nt,-1)
        self.spline=spline
        self.kx=np.asarray(kx)[self.indices.cpu().numpy()]
        self.velocity=np.asarray(velocity)
        self.tmin=float(tmin)
        self.phi=phi
        self.plans={}

    def plan(self,level,h,nstep,starts):
        if level in self.plans:return self.plans[level]
        device=self.rows.device
        begin=self.tmin+np.arange(nstep)*h
        first=np.asarray(starts)[::2][:nstep]
        support=first[:,None]+np.arange(8)[None,:]
        basis=BSpline(self.spline.t,np.eye(self.nt),self.spline.k)
        derivatives=np.stack([basis(begin,nu=d)[np.arange(nstep)[:,None],support]
                              for d in range(4)],axis=1)
        local=torch.as_tensor(derivatives,dtype=torch.complex128,device=device)
        omega=self.frequency.cpu().numpy()
        if omega.ndim==1:
            omega=omega[:,None]
        linear=-1j*(self.kx[:,None]*self.velocity[None,:]+omega)
        half_phi=torch.as_tensor(np.stack([(h/2)**(r+1)*self.phi(linear*h/2,r+1)
                                           for r in range(4)]),device=device)
        full_phi=torch.as_tensor(np.stack([h**(r+1)*self.phi(linear*h,r+1)
                                           for r in range(4)]),device=device)
        if self.cell_carriers:
            # A full (steps,rows,velocities) phase panel can exceed memory.
            # Store the two fixed factors and form the current phase in at().
            phase_half=torch.as_tensor(np.exp(1j*h/2*omega),device=device)
            phase_full=torch.as_tensor(np.exp(1j*h*omega),device=device)
        else:
            phase_half=torch.as_tensor(np.exp(1j*(begin[:,None,None]-self.tmin+h/2)*omega[None]),device=device)
            phase_full=torch.as_tensor(np.exp(1j*(begin[:,None,None]-self.tmin+h)*omega[None]),device=device)
        self.plans[level]=(first,local,half_phi,full_phi,phase_half,phase_full)
        return self.plans[level]

    def at(self,level,h,nstep,starts,tick):
        first,local,half_phi,full_phi,phase_half,phase_full=self.plan(level,h,nstep,starts)
        start=first[tick]
        derivative=(local[tick]@self.rows[start:start+8]).reshape(4,-1,self.nv)
        if self.cell_carriers:
            phase=torch.exp(1j*(tick*h)*self.frequency)
            half=(derivative*half_phi).sum(0)*phase*phase_half
            full=(derivative*full_phi).sum(0)*phase*phase_full
        else:
            half=(derivative*half_phi).sum(0)*phase_half[tick]
            full=(derivative*full_phi).sum(0)*phase_full[tick]
        if self.cell_carriers:
            half=half*self.mask
            full=full*self.mask
        return half,full
