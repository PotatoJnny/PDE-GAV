"""Exact local-piece convolution of a degree-seven source spline.

Each crossed knot starts a new polynomial integral. Composing those bounded
local integrals avoids cancellation between a long polynomial extrapolation
and large derivative-jump corrections. Planning storage is bounded in time.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.interpolate import BSpline


class IntegratedCarrier:
    CHUNK_STEPS = 256

    def __init__(self,coefficients,frequency,spline,kx,velocity,tmin,phi,all_rows=False):
        self.indices=(torch.arange(coefficients.shape[1],device=coefficients.device)
                      if all_rows else (frequency!=0).nonzero().ravel())
        self.frequency=frequency[self.indices]
        self.nt,self.nx,self.nv=coefficients.shape
        self.rows=(coefficients if all_rows else coefficients[:,self.indices]).reshape(self.nt,-1)
        self.spline=spline
        self.kx=np.asarray(kx)[self.indices.cpu().numpy()]
        self.velocity=np.asarray(velocity)
        self.tmin=float(tmin)
        self.phi=phi
        self.linear=torch.as_tensor(-1j*(self.kx[:,None]*self.velocity[None,:]
                    +self.frequency.cpu().numpy()[:,None]),device=self.rows.device)
        self.has_carrier=bool(torch.any(self.frequency!=0).item())
        self.knot_times=np.asarray(self.spline.t[8:-8])
        self.basis=BSpline(self.spline.t,np.eye(self.nt),self.spline.k)
        self.knot_first,self.knot_local=self._local_basis(self.knot_times)
        self.plans={}

    def _local_basis(self,times):
        times=np.asarray(times)
        if not len(times):
            return np.empty(0,dtype=int),torch.empty((0,8,8),dtype=torch.complex128,device=self.rows.device)
        first=BSpline.design_matrix(times,self.spline.t,self.spline.k).indices.reshape(-1,8)[:,0]
        support=first[:,None]+np.arange(8)[None,:]
        local=np.stack([self.basis(times,nu=d)[np.arange(len(times))[:,None],support]
                        for d in range(8)],axis=1)
        return first,torch.as_tensor(local,dtype=torch.complex128,device=self.rows.device)

    @staticmethod
    def phi8(z):
        """Stable entire series for the supported phase magnitude at most4."""
        value=torch.full_like(z,1./math.factorial(39))
        for power in range(30,-1,-1):
            value=value*z+1./math.factorial(8+power)
        return value

    def scaled_phi(self,duration):
        z=self.linear*duration
        value=self.phi8(z)
        values=[duration**8*value]
        for order in range(7,0,-1):
            value=z*value+1./math.factorial(order)
            values.append(duration**order*value)
        return torch.stack(values[::-1])

    def plan(self,level,h,nstep,starts):
        if level not in self.plans:
            # Levels only increase in the solver. Keeping one level also
            # bounds storage when a caller revisits earlier levels.
            self.plans.clear()
            self.plans[level]=dict(h=float(h),nstep=int(nstep),
                half_phi=self.scaled_phi(h/2),full_phi=self.scaled_phi(h),chunk=None)
        return self.plans[level]

    def _chunk(self,plan,tick):
        chunk=plan['chunk']
        if chunk is None or not chunk['start']<=tick<chunk['stop']:
            start=(int(tick)//self.CHUNK_STEPS)*self.CHUNK_STEPS
            stop=min(start+self.CHUNK_STEPS,plan['nstep'])
            begin=self.tmin+np.arange(start,stop)*plan['h']
            first,local=self._local_basis(begin)
            chunk=dict(start=start,stop=stop,begin=begin,first=first,local=local,
                lower=np.searchsorted(self.knot_times,begin,side='right'),
                upper=np.searchsorted(self.knot_times,begin+plan['h'],side='left'))
            plan['chunk']=chunk
        return chunk

    def _derivatives(self,first,local):
        return (local@self.rows[first:first+8]).reshape(8,-1,self.nv)

    def _integral(self,derivative,duration):
        return (derivative*self.scaled_phi(duration)).sum(0)

    def at(self,level,h,nstep,starts,tick):
        plan=self.plan(level,h,nstep,starts)
        chunk=self._chunk(plan,int(tick))
        offset=int(tick)-chunk['start']
        begin=chunk['begin'][offset]
        first=chunk['first'][offset]
        derivative=self._derivatives(first,chunk['local'][offset])
        lower,upper=int(chunk['lower'][offset]),int(chunk['upper'][offset])
        if lower==upper:
            half=(derivative*plan['half_phi']).sum(0)
            full=(derivative*plan['full_phi']).sum(0)
        else:
            # Integrate the envelope with shifted streaming eigenvalue A.
            # For each piece [a,b], compose F(b)=exp(A*(b-a))*F(a)+J(a,b).
            # The half state is evaluated inside its actual spline piece.
            middle=begin+h/2
            end=begin+h
            accumulated=torch.zeros_like(self.linear)
            half=None
            a=begin
            for knot_index in range(lower,upper+1):
                b=self.knot_times[knot_index] if knot_index<upper else end
                duration=b-a
                if half is None and middle<=b:
                    partial=middle-a
                    half=torch.exp(self.linear*partial)*accumulated+self._integral(derivative,partial)
                accumulated=(torch.exp(self.linear*duration)*accumulated
                             +self._integral(derivative,duration))
                a=b
                if knot_index<upper:
                    derivative=self._derivatives(self.knot_first[knot_index],self.knot_local[knot_index])
            full=accumulated
            if half is None:
                raise RuntimeError('Half-step forcing segment was not located')
        if self.has_carrier:
            half *= torch.exp(1j*self.frequency*(begin-self.tmin+h/2))[:,None]
            full *= torch.exp(1j*self.frequency*(begin-self.tmin+h))[:,None]
        return half,full
