"""Small CPU planner for exact definite integrals of source B-splines.

The weights act on B-spline coefficient rows, not source samples.
Only complete time panels and intervals inside their closed domain are valid.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline


def integral_plan(knots, degree, stage_times):
    """Return starts, half/full weights for left/mid/right stage times.

    stage_times has shape (2*nstep+1,) and need not be uniformly spaced.
    Each output has one row per step. All nonzero weights occupy a common
    bounded local support band. Padding is exactly zero. The algebra removes
    cancellations outside the true B-spline support, without estimating an
    error from small interpolated source moments.
    """
    knots=np.asarray(knots,dtype=float)
    stage_times=np.asarray(stage_times,dtype=float)
    if len(stage_times)<3 or len(stage_times)%2!=1 or np.any(np.diff(stage_times)<=0):
        raise ValueError('Expected strictly increasing left/mid/right times')
    ncoef=len(knots)-degree-1
    if stage_times[0]<knots[degree] or stage_times[-1]>knots[-degree-1]:
        raise ValueError('Stage times exceed the source spline domain')
    anti=BSpline(knots,np.eye(ncoef),degree).antiderivative()
    basis=anti(stage_times)
    half=basis[1::2]-basis[:-2:2]
    full=basis[2::2]-basis[:-2:2]
    bounds=[]
    for a,b in zip(stage_times[:-2:2],stage_times[2::2]):
        active=np.flatnonzero((knots[:ncoef]<b)&(knots[degree+1:degree+1+ncoef]>a))
        bounds.append((int(active[0]),int(active[-1])+1))
    width=max(b-a for a,b in bounds)
    starts=np.empty(len(bounds),dtype=np.int64)
    half_local=np.zeros((len(bounds),width),dtype=float)
    full_local=np.zeros_like(half_local)
    for j,(a,b) in enumerate(bounds):
        # Shift the last padded support to remain inside the coefficient array.
        start=min(a,ncoef-width); starts[j]=start
        half_local[j,a-start:b-start]=half[j,a:b]
        full_local[j,a-start:b-start]=full[j,a:b]
    return starts,half_local,full_local


if __name__=='__main__':
    from scipy.interpolate import make_interp_spline
    rng=np.random.default_rng(10403)
    nodes=.75*(1-np.cos(np.pi*np.arange(129)/128))/2
    spline=make_interp_spline(nodes,rng.normal(size=(129,17)),k=7)
    times=np.linspace(0,.75,2049)
    starts,half,full=integral_plan(spline.t,7,times)
    ref=spline.antiderivative()(times)
    assert half.shape[1]<=12
    err=0.
    for j,start in enumerate(starts):
        rows=spline.c[start:start+half.shape[1]]
        err=max(err,float(np.max(abs(half[j]@rows-(ref[2*j+1]-ref[2*j])))),
                float(np.max(abs(full[j]@rows-(ref[2*j+2]-ref[2*j])))))
    assert err<1e-14,err
    assert np.max(abs(half.sum(axis=1)-np.diff(times)[::2]))<1e-15
    assert np.max(abs(full.sum(axis=1)-(times[2::2]-times[:-2:2])))<1e-15
    print({'max_error':err,'maximum_local_support':half.shape[1],'steps':len(starts)})
