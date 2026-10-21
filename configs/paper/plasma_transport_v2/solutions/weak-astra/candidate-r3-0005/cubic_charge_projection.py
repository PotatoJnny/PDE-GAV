"""Standalone charge-only local cubic time projection; no solve or Oracle.

The four-point Lagrange stencil is identical for density and both field
components, so interpolation commutes with the linear Poisson projection.
Query-time density/finite-value fallback matches charge_engine.Prepared.
"""
from __future__ import annotations
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView

CHARGE_FIELDS=('rho','electric_x','electric_y')


def lagrange_stencil(times,query_times):
    """Local polynomial indices/weights on ordered, possibly irregular knots."""
    times=np.asarray(times,dtype=float);query_times=np.asarray(query_times,dtype=float)
    if (times.ndim!=1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0)):
        raise ValueError('finite strictly increasing time knots required')
    if query_times.ndim!=1 or not np.isfinite(query_times).all():
        raise ValueError('finite one-dimensional query times required')
    count=min(4,len(times))
    left=np.searchsorted(times,query_times,side='right')-1
    start=np.clip(left-1,0,len(times)-count)
    indices=start[:,None]+np.arange(count)
    knots=times[indices];weights=np.ones(knots.shape,dtype=float)
    for i in range(count):
        for j in range(count):
            if i!=j:
                weights[:,i]*=(query_times-knots[:,j])/(knots[:,i]-knots[:,j])
    return indices,weights


class CubicChargePrepared:
    def __init__(self,baseline,times=None,modes=None):
        self.baseline=baseline;self.times=times;self.modes=modes
        self.charge_modes=None if modes is None else np.asarray(modes)[:,[0,8,9]]

    def evaluate(self,query):
        fallback=self.baseline.evaluate(query)
        if self.charge_modes is None:return fallback
        p=np.asarray([(v['t'],v['x'],v['y']) for v in query.points],dtype=float)
        indices,weights=lagrange_stencil(self.times,p[:,0])
        fm=np.zeros((len(p),)+self.charge_modes.shape[1:],dtype=self.charge_modes.dtype)
        for j in range(indices.shape[1]):
            fm+=weights[:,j,None,None,None]*self.charge_modes[indices[:,j]]
        bx=np.exp(1j*p[:,1,None]*self.baseline.wave[0])
        by=np.exp(1j*p[:,2,None]*self.baseline.wave[1])
        m=np.einsum('ncij,ni,nj->nc',fm,bx,by).real
        values=dict(zip(CHARGE_FIELDS,(np.maximum(m[:,0],1e-8),m[:,1],m[:,2])))
        bad=(m[:,0]<.1)|(~np.isfinite(m).all(axis=1))
        out={}
        for key in query.fields:
            if key not in CHARGE_FIELDS:
                out[key]=fallback.fields[key]
                continue
            prediction=values[key].copy();base=np.asarray(fallback.fields[key])
            prediction[bad]=base[bad]
            if not np.isfinite(prediction).all():prediction=base
            out[key]=tuple(map(float,prediction))
        return AnswerView(kind=query.kind,fields=out,points=query.points,shape=(len(p),))

    def close(self):self.baseline.close()
