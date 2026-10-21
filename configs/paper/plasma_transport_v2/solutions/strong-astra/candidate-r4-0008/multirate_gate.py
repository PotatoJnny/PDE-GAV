"""High-Hermite temporal source gate using only solver-owned public pilots.

The response diagnostic retains diagonal OU damping only. It omits spatial,
electric-field and broad-basis collision coupling, so the thresholds are
numerical heuristics, not VPFP macroscopic-error bounds.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import math
import numpy as np


def _stable_builder():
    spec=importlib.util.spec_from_file_location('_multirate_stable_source',
        Path(__file__).with_name('source_spline_stable.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_piecewise_cubic_spline


def linear_residual_ou_response(times, residual, rates):
    """Exact OU response to the continuous piecewise-linear residual.

    ``residual`` is (time, probe, coefficient), ``rates`` is (coefficient,).
    Starts from zero response. This public helper supports independent
    polynomial, constant-forcing and variable-rate correctness tests.
    """
    times=np.asarray(times,dtype=float)
    residual=np.asarray(residual,dtype=float)
    rates=np.asarray(rates,dtype=float)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or residual.ndim!=3
            or residual.shape[0]!=len(times) or residual.shape[1]<1
            or rates.shape!=(residual.shape[2],)
            or not np.isfinite(residual).all() or not np.isfinite(rates).all()
            or np.any(rates<0)):
        raise ValueError('ordered times, finite (time,probe,coefficient) residual and nonnegative rates required')
    widths=np.diff(times)
    x=widths[:,None]*rates[None,:]
    small=x<1e-3
    phi1=np.empty_like(x);phi2=np.empty_like(x)
    xs=x[small]
    phi1[small]=1-xs/2+xs**2/6-xs**3/24+xs**4/120-xs**5/720
    phi2[small]=.5-xs/6+xs**2/24-xs**3/120+xs**4/720-xs**5/5040
    xl=x[~small]
    phi1[~small]=-np.expm1(-xl)/xl
    phi2[~small]=(xl+np.expm1(-xl))/(xl*xl)
    decay=np.exp(-x)
    response=np.zeros_like(residual)
    for i,h in enumerate(widths):
        response[i+1]=(decay[i]*response[i]
                       +h*(phi1[i]*residual[i]
                           +phi2[i]*(residual[i+1]-residual[i])))
    return response


def assess_high_moment_temporal_error(times, values, selected_indices,
        corner_indices, nu, *, degree=16, bands=((7,10),(11,16)),
        thresholds=(2e-6,1e-5), cubic_builder=None):
    """Assess coarse high-moment source interpolation at dense pilot times.

    Values have shape (time,probe,(degree+1)*(degree+2)//2), ordered by total
    Hermite degree as in the Candidate. They must all use the same velocity
    basis. The selected coarse snapshot set must cover both time endpoints.
    Missing endpoints/corners produce a failed gate rather than extrapolation.

    A band passes when its largest per-coefficient RMS response across probes,
    maximized over pilot time, is below the corresponding threshold. Aggregate
    band RMS and the largest absolute response are also returned so isolated
    probe/degree excursions remain visible. No solve or Oracle access occurs.
    """
    times=np.asarray(times,dtype=float);values=np.asarray(values,dtype=float)
    selected=np.unique(np.asarray(selected_indices,dtype=int))
    corners=np.unique(np.asarray(corner_indices,dtype=int))
    degree=int(degree)
    count=(degree+1)*(degree+2)//2
    bands=tuple((int(a),int(b)) for a,b in bands)
    thresholds=tuple(float(value) for value in thresholds)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or values.ndim!=3
            or values.shape[0]!=len(times) or values.shape[1]<1
            or values.shape[2]!=count or not np.isfinite(values).all()
            or not math.isfinite(nu) or nu<0 or degree<0
            or len(bands)!=len(thresholds) or not len(bands)
            or any(a<0 or b<0 or (b<a and a<=degree) for a,b in bands)
            or any(not math.isfinite(t) or t<=0 for t in thresholds)
            or np.any(selected<0) or np.any(selected>=len(times))
            or np.any(corners<0) or np.any(corners>=len(times))):
        raise ValueError('invalid source-pilot gate inputs')
    summary={'passed':False,'selected_count':int(len(selected)),
        'pilot_count':int(len(times)),'probe_count':int(values.shape[1]),
        'response_model':'diagonal_ou_piecewise_linear_residual',
        'score_bound':False,'bands':[]}
    if len(selected)<2 or selected[0]!=0 or selected[-1]!=len(times)-1:
        summary['reason']='selected snapshots do not cover both pilot endpoints'
        return summary
    required=corners[(corners>0)&(corners<len(times)-1)]
    if not np.all(np.isin(required,selected)):
        summary['reason']='selected snapshots omit a certified source corner'
        return summary
    all_degrees=np.repeat(np.arange(degree+1),np.arange(1,degree+2))
    used=np.zeros(count,dtype=bool)
    for lower,upper in bands:
        used|=(all_degrees>=lower)&(all_degrees<=upper)
    def empty_band(lower,upper,threshold):
        return {'degrees':[lower,upper],'coefficient_count':0,'threshold':threshold,
            'passed':True,'skipped':True,'max_coefficient_rms':0.,'max_band_rms':0.,
            'max_absolute_response':0.,'time_rms_response':0.,
            'final_max_coefficient_rms':0.,'max_absolute_source_defect':0.}
    if not np.any(used):
        summary['passed']=True
        summary['reason']='no retained coefficients in requested high-degree bands'
        summary['bands']=[empty_band(lower,upper,threshold)
            for (lower,upper),threshold in zip(bands,thresholds)]
        return summary
    active_degrees=all_degrees[used]
    active_values=values[:,:,used]
    local_breaks=np.flatnonzero(np.isin(selected,required))
    if cubic_builder is None:
        cubic_builder=_stable_builder()
    interpolant=cubic_builder(times[selected],active_values[selected],local_breaks)
    residual=interpolant(times)-active_values
    # Mathematically the interpolant is exact at selected samples. Removing
    # their roundoff residual avoids giving them spurious nonzero measure.
    residual[selected]=0.
    response=linear_residual_ou_response(times,residual,nu*active_degrees)
    widths=np.diff(times);duration=times[-1]-times[0]
    for (lower,upper),threshold in zip(bands,thresholds):
        mask=(active_degrees>=lower)&(active_degrees<=upper)
        if not np.any(mask):
            summary['bands'].append(empty_band(lower,upper,threshold))
            continue
        part=response[:,:,mask];defect=residual[:,:,mask]
        coefficient_rms=np.sqrt(np.mean(part*part,axis=1))
        time_band_rms=np.sqrt(np.mean(part*part,axis=(1,2)))
        max_coefficient_rms=float(coefficient_rms.max())
        integrated_energy=np.sum(widths*(time_band_rms[:-1]**2+time_band_rms[1:]**2)/2)/duration
        summary['bands'].append({'degrees':[lower,upper],
            'coefficient_count':int(mask.sum()),'threshold':threshold,
            'passed':bool(max_coefficient_rms<=threshold),
            'max_coefficient_rms':max_coefficient_rms,
            'max_band_rms':float(time_band_rms.max()),
            'max_absolute_response':float(np.max(abs(part))),
            'time_rms_response':float(np.sqrt(integrated_energy)),
            'final_max_coefficient_rms':float(coefficient_rms[-1].max()),
            'max_absolute_source_defect':float(np.max(abs(defect)))})
    summary['passed']=all(row['passed'] for row in summary['bands'])
    summary['reason']='within heuristic thresholds' if summary['passed'] else 'high-moment temporal response exceeds threshold'
    return summary
