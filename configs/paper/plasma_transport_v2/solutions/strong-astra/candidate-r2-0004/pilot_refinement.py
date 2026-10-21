"""Optional source-pilot refinement, independent of Oracle implementation.

The callback ``sample_moments(times)`` supplies normalized theta-Hermite
source moments with shape (time, spatial_probes, 10). All times are selected
before any scoring query. This helper changes only the scout representation;
the caller still acquires its own full spatial/velocity source coefficients.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar


def _fit_corner(times, values, lower, upper):
    center=.5*(lower+upper);width=upper-lower
    z=(times-center)/width
    y=values.reshape(len(times),-1)
    smooth=np.column_stack([np.ones(len(z)),z,z*z,z*z*z])
    scale=max(float(np.linalg.norm(y-y.mean(axis=0))),1e-30)

    def fit(c, details=False):
        pos=np.maximum(z-c,0)
        design=np.column_stack([smooth,pos,pos*pos,pos*pos*pos])
        coeff=np.linalg.lstsq(design,y,rcond=1e-12)[0]
        error=design@coeff-y
        loss=float(np.linalg.norm(error)/scale)**2
        return (loss,coeff,error) if details else loss

    grid=np.linspace((lower-center)/width,(upper-center)/width,65)
    losses=np.array([fit(c) for c in grid]);best=int(np.argmin(losses))
    left=grid[max(best-1,0)];right=grid[min(best+1,len(grid)-1)]
    optimum=minimize_scalar(fit,bounds=(left,right),method='bounded',options={'xatol':1e-11})
    loss,coeff,error=fit(optimum.x,True)
    smooth_error=smooth@np.linalg.lstsq(smooth,y,rcond=1e-12)[0]-y
    return {'location':float(center+width*optimum.x),
            'relative_residual':float(np.sqrt(loss)),
            'smooth_relative_residual':float(np.linalg.norm(smooth_error)/scale),
            'max_absolute_residual':float(np.max(abs(error))),
            'rms_derivative_jump':float(np.sqrt(np.mean((coeff[4]/width)**2)))}


def refine_temporal_pilot(times, values, sample_moments, *, corner_indices=(),
                          early_count=64, refine_corners=True, max_corners=32):
    """Enrich the initial layer and validate/localize existing corner brackets.

    On resolved smooth transitions or ambiguous fits, supplied boundaries
    are preserved. A corner is moved only when two independent local fits
    locate a common point and both clearly outperform a smooth cubic. A
    transition below the probe resolution may remain indistinguishable from
    a corner; this helper is an interpolation heuristic, not a regularity
    certificate. The
    callback may raise the caller's deadline exception; callers can retain
    the original pilot transactionally if that happens.

    Returns ``times, values, corner_indices, diagnostics``. The original
    pilot values are preserved exactly, and the new representation covers
    the same public time interval.
    """
    times=np.asarray(times,dtype=float);values=np.asarray(values,dtype=float)
    if (times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0)
            or values.ndim!=3 or values.shape[0]!=len(times) or values.shape[2]!=10
            or not np.isfinite(times).all() or not np.isfinite(values).all()):
        raise ValueError('Finite ordered times and (time, probes, 10) values required')
    if int(early_count)!=early_count or early_count<0 or early_count==1:
        raise ValueError('early_count must be zero or an integer of at least two')
    corners=np.unique(np.asarray(corner_indices,dtype=int))
    if np.any(corners<0) or np.any(corners>=len(times)):
        raise ValueError('corner indices leave the supplied pilot')
    original_times=times.copy();original_values=values.copy()
    entries={float(t):v.copy() for t,v in zip(times,values)}
    breaks=set(map(float,times[corners]));reports=[]
    span=times[-1]-times[0]

    def acquire(new_times):
        new_times=np.unique(np.asarray(new_times,dtype=float))
        new_times=new_times[(new_times>=times[0])&(new_times<=times[-1])]
        new_times=np.array([t for t in new_times if float(t) not in entries])
        if len(new_times):
            new_values=np.asarray(sample_moments(new_times),dtype=float)
            if new_values.shape!=(len(new_times),values.shape[1],10) or not np.isfinite(new_values).all():
                raise ValueError('Source callback returned invalid moment values')
            for t,v in zip(new_times,new_values): entries[float(t)]=v.copy()

    if early_count:
        acquire(times[0]+span*np.geomspace(1e-10,1/256,int(early_count)))
    if refine_corners and len(corners):
        spacing=float(np.median(np.diff(original_times)))
        groups=np.split(corners,np.flatnonzero(np.diff(original_times[corners])>1.01*spacing)+1)
        for group in groups[:max_corners]:
            lower=float(original_times[group[0]]-spacing)
            upper=float(original_times[group[-1]]+spacing)
            if lower<=times[0]+2*spacing or upper>=times[-1]-2*spacing:
                continue
            coarse=np.linspace(lower-2*spacing,upper+2*spacing,33)
            acquire(coarse)
            cv=np.array([entries[float(t)] for t in coarse])
            first=_fit_corner(coarse,cv,lower,upper)
            width=upper-lower;center=first['location']
            fine=center+width*np.linspace(-.1,.1,25)
            acquire(fine)
            fv=np.array([entries[float(t)] for t in fine])
            second=_fit_corner(fine,fv,center-.04*width,center+.04*width)
            accepted=(first['relative_residual']<.03*first['smooth_relative_residual']
                      and second['relative_residual']<.03*second['smooth_relative_residual']
                      and abs(second['location']-first['location'])<.002*width
                      and lower+.02*width<second['location']<upper-.02*width)
            reports.append({'bracket':[lower,upper],'first':first,'second':second,'accepted':bool(accepted)})
            if accepted:
                location=second['location']
                nearest=int(np.argmin(abs(original_times-location)))
                # Avoid a near-zero spline interval when an already sampled
                # node is numerically indistinguishable from the fitted kink.
                if abs(original_times[nearest]-location)<1.e-6*width:
                    location=float(original_times[nearest])
                acquire([location])
                reports[-1]['selected_break']=location
                breaks.difference_update(map(float,original_times[group]));breaks.add(location)
    result_times=np.array(sorted(entries))
    result_values=np.array([entries[t] for t in result_times])
    result_breaks=np.flatnonzero(np.isin(result_times,list(breaks)))
    # Every input sample survives exactly; this matters if the caller falls
    # back between full-pilot and partially acquired source representations.
    index=np.searchsorted(result_times,original_times)
    if not np.array_equal(result_values[index],original_values):
        raise AssertionError('Pilot refinement changed existing values')
    return result_times,result_values,result_breaks,{'added_times':len(result_times)-len(times),'corners':reports}
