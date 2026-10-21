"""Geometry-conditioned cubic source and density interpolation.

Bernstein control-vector norms give a worst-case interpolation amplification
bound for additive sample noise in the infinity norm. This does not bound
approximation bias or coupled VPFP error; source-pilot validation is needed.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline, CubicSpline, CubicHermiteSpline


def _cubic_block_coefficients(times, values, block_size=512):
    """Incumbent stable cubic coefficient construction, unchanged arithmetic."""
    n=len(times);shape=values.shape[1:];data=values.reshape(n,-1)
    if n<4:
        knots=np.r_[np.repeat(times[0],4),np.repeat(times[-1],4)]
        result=np.empty((4,data.shape[1]),dtype=np.result_type(values.dtype,float))
        for first in range(0,data.shape[1],block_size):
            part=slice(first,first+block_size)
            cubic=CubicSpline(times,data[:,part],axis=0);h=times[-1]-times[0]
            result[0,part]=data[0,part]
            result[1,part]=data[0,part]+(h/3)*cubic(times[0],1)
            result[2,part]=data[-1,part]-(h/3)*cubic(times[-1],1)
            result[3,part]=data[-1,part]
    else:
        knots=np.r_[np.repeat(times[0],4),times[2:-2],np.repeat(times[-1],4)]
        result=np.empty((n,data.shape[1]),dtype=np.result_type(values.dtype,float))
        spans=np.minimum(np.arange(n),n-2)
        d=np.stack([knots[1:n+1],knots[2:n+2],knots[3:n+3]],axis=1)-times[spans,None]
        linear=d.sum(axis=1)/3
        quadratic=(d[:,0]*d[:,1]+d[:,0]*d[:,2]+d[:,1]*d[:,2])/3
        cubic_term=d.prod(axis=1)
        for first in range(0,data.shape[1],block_size):
            part=slice(first,first+block_size)
            pp=CubicSpline(times,data[:,part],axis=0).c[:,spans]
            result[:,part]=(pp[3]+linear[:,None]*pp[2]+quadratic[:,None]*pp[1]+cubic_term[:,None]*pp[0])
    return knots,result.reshape((len(result),)+shape)


def _derivative_geometry(times, gain_limit=8.):
    times=np.asarray(times,float)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not np.isfinite(gain_limit) or gain_limit<5/3):
        raise ValueError('ordered finite source times and gain limit >= 5/3 required')
    n=len(times);identity=np.eye(n)
    ordinary=CubicSpline(times,identity,axis=0)(times,1)
    safe=np.zeros_like(ordinary)
    safe[0,:2]=np.array([-1.,1.])/(times[1]-times[0])
    safe[-1,-2:]=np.array([-1.,1.])/(times[-1]-times[-2])
    for i in range(1,n-1):
        safe[i,i-1]=-1/(times[i+1]-times[i-1])
        safe[i,i+1]=1/(times[i+1]-times[i-1])
    factors=np.ones(n);ordinary_max=1.;safe_max=1.
    for sign,width,indices in ((1.,np.diff(times),np.arange(n-1)),
                               (-1.,np.diff(times),np.arange(1,n))):
        old=identity[indices]+sign*width[:,None]*ordinary[indices]/3
        new=identity[indices]+sign*width[:,None]*safe[indices]/3
        old_norm=np.sum(abs(old),axis=1);new_norm=np.sum(abs(new),axis=1)
        ordinary_max=max(ordinary_max,float(old_norm.max()))
        safe_max=max(safe_max,float(new_norm.max()))
        scale=np.ones(len(indices))
        bad=old_norm>gain_limit
        scale[bad]=(gain_limit-new_norm[bad])/(old_norm[bad]-new_norm[bad])
        factors[indices]=np.minimum(factors[indices],np.clip(scale,0.,1.))
    operator=factors[:,None]*ordinary+(1-factors[:,None])*safe
    conditioned_max=1.
    for sign,width,indices in ((1.,np.diff(times),np.arange(n-1)),
                               (-1.,np.diff(times),np.arange(1,n))):
        controls=identity[indices]+sign*width[:,None]*operator[indices]/3
        conditioned_max=max(conditioned_max,float(np.sum(abs(controls),axis=1).max()))
    return operator,dict(gain_limit=float(gain_limit),ordinary_control_bound=ordinary_max,
        safe_control_bound=safe_max,conditioned_control_bound=conditioned_max,
        modified_derivatives=int(np.sum(factors<1.)),minimum_ordinary_fraction=float(factors.min())),factors


def derivative_operator(times,gain_limit=8.):
    operator,report,_=_derivative_geometry(times,gain_limit)
    return operator,report


@lru_cache(maxsize=256)
def _cached_conditioning(time_bytes,gain_limit):
    """Cache only small geometry factors and scalar diagnostics, never D."""
    times=np.frombuffer(time_bytes,dtype=np.float64)
    _,report,factors=_derivative_geometry(times,gain_limit)
    factors.setflags(write=False)
    return report,factors


def _conditioning_geometry(times,gain_limit):
    values=np.asarray(times,dtype=np.float64)
    report,factors=_cached_conditioning(values.tobytes(),float(gain_limit))
    # Diagnostics may be attached to a public handle; keep cached state private.
    return dict(report),factors


def piecewise_bounded_spline(times,values,corners=(),gain_limit=8.,block_size=512,
                             cubic_block_builder=None):
    times=np.asarray(times,float);values=np.asarray(values)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or values.ndim<1 or len(values)!=len(times) or not np.isfinite(values).all()
            or not np.isfinite(gain_limit) or gain_limit<5/3):
        raise ValueError('ordered finite times, finite matching values and gain limit >= 5/3 required')
    values=values.astype(np.result_type(values.dtype,np.float64),copy=False)
    if not isinstance(block_size,(int,np.integer)) or block_size<1:
        raise ValueError('positive integer channel block size required')
    if cubic_block_builder is None:cubic_block_builder=_cubic_block_coefficients
    corners=np.asarray(corners,float)
    if corners.ndim!=1 or not np.isfinite(corners).all() or not np.all(np.isin(corners,times)):
        raise ValueError('corners must be acquired finite source times')
    breaks=np.unique(np.r_[0,np.flatnonzero(np.isin(times,corners)),len(times)-1])
    coefficients=[];knots=[];diagnostics=[]
    for number,(a,b) in enumerate(zip(breaks[:-1],breaks[1:])):
        t=times[a:b+1];f=values[a:b+1]
        report,factors=_conditioning_geometry(t,gain_limit)
        if not report['modified_derivatives'] and cubic_block_builder is not None:
            kt,c=cubic_block_builder(t,f,block_size=block_size)
        else:
            # Apply the blend in O(time*channels), in bounded value blocks.
            # The cubic Hermite spline is C1: double interior B-spline knots.
            data=f.reshape(len(t),-1)
            result=np.empty((2*len(t),data.shape[1]),dtype=f.dtype)
            alpha=factors[:,None];h=np.diff(t)[:,None]
            for first in range(0,data.shape[1],int(block_size)):
                part=slice(first,first+int(block_size));v=data[:,part]
                ordinary=CubicSpline(t,v,axis=0)(t,1)
                safe=np.empty_like(ordinary)
                safe[0]=(v[1]-v[0])/(t[1]-t[0])
                safe[-1]=(v[-1]-v[-2])/(t[-1]-t[-2])
                safe[1:-1]=(v[2:]-v[:-2])/(t[2:]-t[:-2])[:,None]
                derivatives=alpha*ordinary+(1-alpha)*safe
                result[0,part]=v[0];result[-1,part]=v[-1]
                result[1:-1:2,part]=v[:-1]+h*derivatives[:-1]/3
                result[2:-1:2,part]=v[1:]-h*derivatives[1:]/3
            c=result.reshape((2*len(t),)+f.shape[1:])
            kt=np.r_[np.repeat(t[0],4),np.repeat(t[1:-1],2),np.repeat(t[-1],4)]
        if number:knots[-1]=knots[-1][:-4]
        knots.append(kt);coefficients.append(c)
        diagnostics.append(dict(start=float(t[0]),end=float(t[-1]),nodes=len(t),
            ordinary_coefficient_count=max(4,len(t)),coefficient_count=len(c),**report))
    spline=BSpline.construct_fast(np.concatenate(knots),np.concatenate(coefficients),3,extrapolate=False)
    return spline,{'segments':diagnostics,
        'ordinary_control_bound':max(r['ordinary_control_bound'] for r in diagnostics),
        'conditioned_control_bound':max(r['conditioned_control_bound'] for r in diagnostics),
        'modified_derivatives':sum(r['modified_derivatives'] for r in diagnostics),
        'ordinary_coefficient_count':sum(r['ordinary_coefficient_count'] for r in diagnostics),
        'coefficient_count':sum(r['coefficient_count'] for r in diagnostics)}


class BoundedPiecewiseCubic:
    """Drop-in pilot_density.PiecewiseCubic with a geometry-only cap.

    Set ``gain_limit`` on a subclass to choose the preparation-wide cap.
    Ordinary pieces use the identical CubicSpline constructor and query path.
    """
    gain_limit=32.
    def __init__(self,times,values,corners=()):
        self.times=np.asarray(times,float);self.values=np.asarray(values)
        if (self.times.ndim!=1 or len(self.times)<2 or np.any(np.diff(self.times)<=0)
                or not np.isfinite(self.times).all() or self.values.ndim<1
                or len(self.values)!=len(self.times) or not np.isfinite(self.values).all()):
            raise ValueError('ordered finite source times and matching finite values required')
        joins=np.flatnonzero(np.isin(self.times,corners))
        indices=np.unique(np.r_[0,joins,len(self.times)-1])
        self.ends=self.times[indices[1:]];self.parts=[];self.diagnostics=[]
        for a,b in zip(indices[:-1],indices[1:]):
            t=self.times[a:b+1];f=self.values[a:b+1]
            report,alpha=_conditioning_geometry(t,self.gain_limit)
            ordinary=CubicSpline(t,f,axis=0)
            if report['modified_derivatives']:
                derivatives=ordinary(t,1);safe=np.empty_like(derivatives)
                safe[0]=(f[1]-f[0])/(t[1]-t[0]);safe[-1]=(f[-1]-f[-2])/(t[-1]-t[-2])
                shape=(-1,)+(1,)*(f.ndim-1)
                safe[1:-1]=(f[2:]-f[:-2])/(t[2:]-t[:-2]).reshape(shape)
                alpha=alpha.reshape(shape)
                ordinary=CubicHermiteSpline(t,f,alpha*derivatives+(1-alpha)*safe,axis=0)
            self.parts.append(ordinary);self.diagnostics.append(report)
    def __call__(self,times):
        t=np.atleast_1d(np.asarray(times,float))
        slots=np.minimum(np.searchsorted(self.ends,t,side='right'),len(self.parts)-1)
        out=np.empty((len(t),)+self.values.shape[1:],dtype=np.result_type(self.values,float))
        for i,part in enumerate(self.parts):
            mask=slots==i
            if np.any(mask):out[mask]=part(t[mask])
        return out


def make_piecewise_cubic_spline(times,values,break_indices,*,join_multiplicity=4,gain_limit=32.):
    """Production-compatible bounded B-spline builder (ordinary joins use4)."""
    if join_multiplicity!=4:
        raise ValueError('bounded source builder requires independent multiplicity-four joins')
    times=np.asarray(times,float);breaks=np.asarray(break_indices,int)
    if breaks.ndim!=1 or np.any(breaks<0) or np.any(breaks>=len(times)):
        raise ValueError('break indices must lie within acquired source times')
    spline,diagnostics=piecewise_bounded_spline(times,values,times[breaks],gain_limit=gain_limit)
    spline.bounded_diagnostics=diagnostics
    return spline


def requires_conditioning(times,break_indices,gain_limit=32.):
    """Whether any acquired piece needs the geometry-only cubic guard.

    A quintic's ordinary-cubic fallback is not bounded by this guard. The
    preparation path retains degree three whenever this predicate is true.
    It leaves the original degree chooser unchanged on inactive geometry.
    """
    times=np.asarray(times,float);breaks=np.asarray(break_indices,int)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or breaks.ndim!=1
            or np.any(breaks<0) or np.any(breaks>=len(times))):
        raise ValueError('ordered finite times and valid source break indices required')
    bounds=np.unique(np.r_[0,breaks,len(times)-1])
    for a,b in zip(bounds[:-1],bounds[1:]):
        report,_=_conditioning_geometry(times[a:b+1],gain_limit)
        if report['modified_derivatives']:
            return True
    return False


def primitive_noise_bound(times,corners=(),gain_limit=None):
    """Bernstein bound on all sample-noise-to-primitive weights, all t.

    Returns both worst bounded-noise (L1) and independent-unit-noise standard
    deviation (L2) bounds. A Bernstein control-hull bound can be conservative.
    """
    times=np.asarray(times,float);n=len(times);identity=np.eye(n)
    breaks=np.unique(np.r_[0,np.flatnonzero(np.isin(times,corners)),n-1])
    primitive=np.zeros(n);max_l1=max_l2=0.
    for a,b in zip(breaks[:-1],breaks[1:]):
        t=times[a:b+1];local=identity[a:b+1]
        if gain_limit is None:
            pp=CubicSpline(t,local,axis=0)
        else:
            operator,_=derivative_operator(t,gain_limit)
            pp=CubicHermiteSpline(t,local,operator@local,axis=0)
        for i,h in enumerate(np.diff(t)):
            power=np.array([primitive,pp.c[3,i]*h,pp.c[2,i]*h*h/2,
                pp.c[1,i]*h**3/3,pp.c[0,i]*h**4/4])
            control=np.array([sum(power[k]*math.comb(j,k)/math.comb(4,k) for k in range(j+1)) for j in range(5)])
            max_l1=max(max_l1,float(np.sum(abs(control),axis=1).max()))
            max_l2=max(max_l2,float(np.linalg.norm(control,axis=1).max()))
            primitive=control[-1]
    return {'primitive_l1_bound':max_l1,'primitive_l2_bound':max_l2,
        'final_primitive_l1':float(np.sum(abs(primitive))),
        'final_primitive_l2':float(np.linalg.norm(primitive))}
