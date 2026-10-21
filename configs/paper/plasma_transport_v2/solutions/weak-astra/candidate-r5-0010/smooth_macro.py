"""Positive cubic-in-time projection of a prepared continuous macro state.

Every cubic Bernstein control is a realizable raw state. Convex Bernstein and
bilinear spatial weights therefore preserve the density/pressure cone at every
query, while Poisson uses exactly the same temporal control weights.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from pde_agent_verifier_solver.sdk import AnswerView
from base import FIELDS
from moments import central_values
from macro_limiter import (LocalPrepared, pointwise_weights, bilinear_values,
                           bilinear_density_electric_modes, field_values)


class CubicPrepared(LocalPrepared):
    def __init__(self,baseline,times,raw_grid,diagnostics=None,check=lambda:None):
        super().__init__(baseline,times,raw_grid,diagnostics,check)
        check()
        slopes=CubicSpline(self.times,self.raw_grid,axis=0,bc_type='natural')(self.times,1)
        self.controls=[];self.control_fields=[]
        minimum=[];means=[]
        for i,step in enumerate(np.diff(self.times)):
            pair=[];fields=[]
            for endpoint,sign in ((i,1.),(i+1,-1.)):
                check()
                prior=self.raw_grid[endpoint]
                delta=sign*step/3*slopes[endpoint]
                weight=pointwise_weights(prior,delta)
                control=prior+weight[None]*delta
                control*=prior[0].mean()/control[0].mean()
                pair.append(control)
                field,frequencies=bilinear_density_electric_modes(control,baseline.periods)
                fields.append(field)
                minimum.append(float(weight.min()));means.append(float(weight.mean()))
            self.controls.append(pair);self.control_fields.append(fields)
        self.controls=np.asarray(self.controls)
        self.control_fields=np.asarray(self.control_fields)
        self.diagnostics['time_projection']='positive cubic Bernstein controls with natural-spline slopes'
        self.diagnostics['time_control_weight_min']=min(minimum)
        self.diagnostics['time_control_weight_mean']=float(np.mean(means))
        check()

    def evaluate(self,query):
        if self.closed:raise RuntimeError('prepared solution closed')
        self.baseline.evaluate(query)
        p=np.asarray([(v['t'],v['x'],v['y']) for v in query.points],float)
        left=np.clip(np.searchsorted(self.times,p[:,0],side='right')-1,0,len(self.times)-2)
        alpha=(p[:,0]-self.times[left])/(self.times[left+1]-self.times[left])
        raw=np.empty((10,len(p)));field=np.empty((2,len(p)))
        for i in np.unique(left):
            mask=left==i;xy=p[mask,1:];a=alpha[mask];b=1-a
            weights=(b**3,3*a*b*b,3*a*a*b,a**3)
            grids=(self.raw_grid[i],self.controls[i,0],self.controls[i,1],self.raw_grid[i+1])
            modes=(self.field_modes[i],self.control_fields[i,0],self.control_fields[i,1],self.field_modes[i+1])
            raw[:,mask]=sum(w*bilinear_values(g[:10],xy,self.baseline.periods) for w,g in zip(weights,grids))
            field[:,mask]=sum(w*field_values(m,self.field_frequencies,xy,self.baseline.periods) for w,m in zip(weights,modes))
        macros=central_values(raw,self.baseline.velocity_center)
        values=np.concatenate((macros[:1],field,macros[1:]))
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite cubic prepared projection')
        return AnswerView(kind=query.kind,
                          fields={key:tuple(map(float,values[FIELDS.index(key)])) for key in query.fields},
                          points=query.points,shape=(len(p),))
