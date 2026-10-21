"""Per-frame public-pilot validation of cheaper late source quadrature.

The full source rule remains the reference. A cheaper spatial/velocity plan
must match independent full-rule pilots at every acquired time, using the
unchanged spatial guard. Failed attempts fall back to the ordinary sampler.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import fft2


class LateSourcePlan:
    def __init__(self, get, *, periods, theta, variance, degree, state_n,
                 full_order, pilot_xy, reference_modes, reference_values,
                 reference_time, guard_factory, rule_factory, projector_factory,
                 pad_modes):
        self.get=get;self.periods=np.asarray(periods);self.n=state_n
        self.pad_modes=pad_modes;self.plan=None;self.attempts=0
        self.guard=guard_factory(state_n,pilot_xy,periods,degree,
                                  reference_modes,reference_values)
        self.calibration=[]
        if not self.guard.enabled:return
        orders=sorted(set([20,22,24,full_order]))
        orders=[q for q in orders if q<=full_order]
        meshes=sorted(set([min(29,state_n),min(33,state_n),state_n]))
        plans=sorted((q*q*m*m,q,m) for q in orders for m in meshes
                     if q<full_order)
        # Do not exceed the incumbent's secondary reduced spatial mesh.
        plans=[p for p in plans if p[0]<full_order**2*min(33,state_n)**2]
        for cost,q,m in plans:
            velocity,h=rule_factory(q,theta,variance,degree)
            projector=projector_factory(h)
            x,y=np.meshgrid(np.arange(m)*periods[0]/m,
                            np.arange(m)*periods[1]/m,indexing='ij')
            vx,vy=np.meshgrid(velocity,velocity,indexing='ij')
            points=np.empty((m*m,q*q,5))
            points[:,:,1:3]=np.c_[x.ravel(),y.ravel()][:,None,:]
            points[:,:,3]=vx.ravel();points[:,:,4]=vy.ravel()
            plan=(q,m,points.reshape(-1,5),projector)
            modes=self._sample(reference_time,plan)
            # Calibration rejects do not disable subsequent, finer plans.
            probe_guard=guard_factory(state_n,pilot_xy,periods,degree,
                                       reference_modes,reference_values)
            passed=probe_guard.accept(modes,reference_values)
            self.calibration.append(dict(order=q,mesh=m,accepted=bool(passed)))
            if passed:
                self.plan=plan
                break

    def _sample(self,time,plan):
        q,m,points,projector=plan;points[:,0]=time
        raw=np.asarray(self.get('source',points)).reshape(m*m,q*q)
        moments=projector.project(raw).T.reshape(-1,m,m)
        modes=self.pad_modes(fft2(moments,axes=(-2,-1)),self.n)
        modes[0,0,0]=0.
        return modes

    def sample(self,time,pilot_values):
        if self.plan is None or not self.guard.enabled:return None
        self.attempts+=1
        modes=self._sample(time,self.plan)
        return modes if self.guard.accept(modes,pilot_values) else None

    def diagnostics(self):
        return dict(calibration=self.calibration,attempts=self.attempts,
                    selected=None if self.plan is None else self.plan[:2],
                    guard=self.guard.diagnostics())
