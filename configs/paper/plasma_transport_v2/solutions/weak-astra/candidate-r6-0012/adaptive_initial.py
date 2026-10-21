"""Optional initial velocity-rule selection from a small public g0 pilot.

Only numerical equation data are used. The caller supplies its existing
sample(sites, relative_velocity, 'g0') function, which owns Oracle access,
request limits, physical velocity centering, and the shared solve clock.
"""
from __future__ import annotations
import numpy as np
from moments import central_values

PAIRS=tuple((degree-b,b) for degree in range(6) for b in range(degree+1))
GATES={'momentum':.000375,'pressure':.000375,'heat':.00075,
       'degree4':.0015,'degree5':.0015,'mass_max_defect':.00075}
SCALES={'momentum':.4788345668393978,'pressure':2.930763214320111,
        'heat':.9900774471118267}
FALLBACK=(64,18.)


def pilot_positions(periods,count=16):
    """Deterministic off-grid sites, independent of later scoring queries."""
    periods=np.asarray(periods,float)
    if (periods.shape!=(2,) or not np.isfinite(periods).all() or np.any(periods<=0)
            or isinstance(count,bool) or int(count)!=count or count<1):
        raise ValueError('positive periods and integer pilot count required')
    row=np.arange(int(count),dtype=float)+.5
    return np.column_stack((np.mod(row*np.sqrt(2.),1.)*periods[0],
                            np.mod(row*np.sqrt(3.),1.)*periods[1]))


def _velocity(n,half):
    axis=(np.arange(n)-n//2)*(2*half/n)
    return np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)


def _raw(values,n,half,rho):
    values=np.asarray(values,float);rho=np.asarray(rho,float)
    if values.shape!=(len(rho),n*n):
        raise ValueError('initial pilot sample must have (sites,n*n) shape')
    if not np.isfinite(values).all() or np.any(values<0):
        raise FloatingPointError('finite nonnegative initial pilot density required')
    v=_velocity(n,half)
    weights=np.stack([v[:,0]**a*v[:,1]**b*(2*half/n)**2 for a,b in PAIRS])
    conditional=weights@values.T
    mass=conditional[0].copy()
    if np.any(mass<=0) or not np.isfinite(conditional).all():
        raise FloatingPointError('finite positive pilot quadrature mass required')
    raw=conditional*(rho/mass)[None]
    raw[0]=rho
    return raw,mass


def _record(values,n,half,rho,center,theta,reference,reference_central):
    raw,mass=_raw(values,n,half,rho)
    central=central_values(raw,center)
    record={'n':int(n),'half':float(half),'nodes':int(n*n)}
    for name,ids,weights in (('momentum',(1,2),(1,1)),
                             ('pressure',(3,4,5),(1,2,1)),('heat',(6,7),(1,1))):
        ids=list(ids);weight=np.asarray(weights)[:,None]
        difference=central[ids]-reference_central[ids]
        error=np.sqrt(np.mean(np.sum(weight*difference*difference,axis=0)))
        scale=np.sqrt(np.mean(np.sum(weight*reference_central[ids]**2,axis=0))+SCALES[name]**2)
        record[name]=float(error/scale)
    for degree in (4,5):
        ids=[i for i,(a,b) in enumerate(PAIRS) if a+b==degree]
        error=np.sqrt(np.mean((raw[ids]-reference[ids])**2))
        scale=max(float(np.sqrt(np.mean(reference[ids]**2))),1.,theta**(degree/2))
        record['degree'+str(degree)]=float(error/scale)
    record['mass_max_defect']=float(np.max(abs(mass-1.)))
    record['gate_utilization']=float(max(record[k]/value for k,value in GATES.items()))
    record['accepted']=bool(all(np.isfinite(record[k]) and record[k]<=value for k,value in GATES.items()))
    return record


def select_initial_rule(xy,pilot_rho,center,theta,sample,*,check=None):
    """Return (velocity_count, half_width, diagnostics), with original fallback.

Reference: uniform NV128/half24. It supplies exact nested NV48/half18 and
NV80/half15 samples. Three additional calls test NV48/half14, NV48/half16,
and NV64/half16; NV56/half14 is a nested subwindow of the last. At16 sites the
total acquisition is401408 g0 points. Calls never exceed262144 points when
the pilot has16 sites; the supplied sampler remains responsible for batching.

Positive finite reference moments and a mass defect <=.002 are required.
This is a numerical validation gate, not a certificate of full velocity-space
accuracy. Failure returns the incumbent rule (64,18) without modifying state.
"""
    check=check or (lambda:None)
    diagnostics={'status':'fallback','fallback_rule':list(FALLBACK),'gates':dict(GATES),
                 'reference_rule':[128,24.],'reference_mass_tolerance':.002,
                 'reference_valid':False,'sampled_points':0,'trials':[],
                 'limitation':'sparse initial pilots do not certify unsampled spatial locations or unresolved tails'}
    try:
        check();xy=np.asarray(xy,float);rho=np.asarray(pilot_rho,float);center=np.asarray(center,float);theta=float(theta)
        if (xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<1 or rho.shape!=(len(xy),)
                or center.shape!=(2,) or theta<=0 or not np.isfinite(theta)
                or not np.isfinite(xy).all() or not np.isfinite(rho).all()
                or np.any(rho<=0) or not np.isfinite(center).all()):
            raise ValueError('finite pilot sites, positive density/temperature and center required')
        diagnostics['pilot_count']=len(xy)
        def acquire(n,half):
            check();v=_velocity(n,half)
            values=np.asarray(sample(xy,v,'g0'),float)
            diagnostics['sampled_points']+=len(xy)*len(v)
            check()
            # Validate before reshaping so errors preserve the old production rule.
            _raw(values,n,half,rho)
            return values.reshape(len(xy),n,n)
        ref_values=acquire(128,24.)
        reference,mass=_raw(ref_values.reshape(len(xy),-1),128,24.,rho)
        diagnostics['reference_mass_max_defect']=float(np.max(abs(mass-1.)))
        reference_central=central_values(reference,center)
        valid=(np.isfinite(reference_central).all()
               and diagnostics['reference_mass_max_defect']<=diagnostics['reference_mass_tolerance'])
        diagnostics['reference_valid']=bool(valid)
        if not valid:raise ValueError('initial reference failed finite/mass validation')
        trials={(48,18.):ref_values[:,16:112:2,16:112:2],
                (80,15.):ref_values[:,24:104,24:104]}
        for n,half in ((48,14.),(48,16.),(64,16.)):
            trials[n,half]=acquire(n,half)
        trials[56,14.]=trials[64,16.][:,4:60,4:60]
        for (n,half),values in trials.items():
            check()
            diagnostics['trials'].append(_record(values.reshape(len(xy),-1),n,half,rho,center,theta,
                                                 reference,reference_central))
        accepted=[r for r in diagnostics['trials'] if r['accepted']]
        if accepted:
            selected=min(accepted,key=lambda r:(r['nodes'],r['gate_utilization']))
            diagnostics.update(status='selected',selected=dict(selected))
            return selected['n'],selected['half'],diagnostics
        diagnostics['reason']='no tested rule passed every initial-moment gate'
    except (ValueError,FloatingPointError,TimeoutError,MemoryError) as error:
        diagnostics['reason']=type(error).__name__+': '+str(error)
    return FALLBACK[0],FALLBACK[1],diagnostics
