"""Public held-out calibration for the spatial source-defect filter.

No Oracle access. The caller supplies source samples and their Fourier
representation at four existing time nodes. Comparisons concern central
forcing about f0 and do not certify later nonlinear or streaming errors.
"""
from __future__ import annotations
import numpy as np
from moments import central_tangent

FROZEN_SCALES = np.asarray((1.0041253663782175, .4788345668393978,
                            2.930763214320111, .9900774471118267))
GROUPS = ((0,), (1,2), (3,4,5), (6,7))
GROUP_WEIGHTS = ((1,), (1,1), (1,2,1), (1,1))
OBJECTIVE_WEIGHTS = np.asarray((.125,.25,.25,.25))


def calibration_plan(interval, periods, source_time_count, count=64):
    """Four exact source times and common off-grid, solver-owned spatial sites."""
    count=int(count)
    if source_time_count < 4 or count < 16 or count % 4:
        raise ValueError('at least four times and a multiple-of-four site count required')
    frames=np.rint(np.asarray((0.,.25,.5,1.))*(source_time_count-1)).astype(int)
    row=np.arange(count//4,dtype=float)+.5
    fractions=np.mod(row[:,None]*np.sqrt(np.asarray((2.,3.)))[None]
                     + np.asarray((.173,.419))[None],1.)
    xy=fractions*np.asarray(periods)[None]
    all_times=np.linspace(*interval,source_time_count)
    sites=np.column_stack((np.repeat(all_times[frames],len(xy)),np.tile(xy,(4,1))))
    return sites,np.repeat(frames,len(xy))


def select_taper(defect_modes, calibration_values, calibration_frames, calibration_xy,
                 initial_raw, moment_weights, periods, center, duration, source_n,
                 *, default=.60, candidates=(.60,.70,.80,.90), check=lambda:None):
    """Select a stronger passband only after a substantial held-out improvement.

`defect_modes` has (time, spatial_mode, velocity), and calibration_values
contains the sampled S plus the same Fourier initial operator used in the
prepared solve. Physical moment_weights have shape (at least10, velocity).
The filter commutes with free-OU response but not with all field feedback.
"""
    check()
    source=np.asarray(defect_modes)
    values=np.asarray(calibration_values,float)
    frames=np.asarray(calibration_frames,int)
    xy=np.asarray(calibration_xy,float)
    raw=np.asarray(initial_raw,float)
    weights=np.asarray(moment_weights,float)[:10]
    n=int(round(np.sqrt(source.shape[1])))
    if (source.ndim!=3 or n*n!=source.shape[1] or values.shape!=(len(frames),source.shape[2])
            or xy.shape!=(len(frames),2) or raw.shape[1:]!=(len(frames),)
            or weights.shape!=(10,source.shape[2]) or duration<=0
            or not all(np.isfinite(a).all() for a in (values,xy,raw,weights))):
        raise ValueError('finite compatible held-out taper calibration required')
    frequencies=np.fft.fftfreq(n)*n
    radius=np.maximum(abs(frequencies[:,None]),abs(frequencies[None,:])).ravel()
    unique=np.unique(frames)
    # Contract only four snapshots. No extra full (time,mode,velocity) copy.
    moments={int(frame):(source[frame]@weights.T).T for frame in unique}
    phase_x=np.exp(2j*np.pi*xy[:,0,None]*frequencies[None]/periods[0])
    phase_y=np.exp(2j*np.pi*xy[:,1,None]*frequencies[None]/periods[1])
    phase=(phase_x[:,:,None]*phase_y[:,None,:]).reshape(len(frames),n*n)
    reference=weights@values.T
    tangent=central_tangent(raw,reference,center)
    # Derivative units are normalized by the same physical score floors / T.
    floor=FROZEN_SCALES/max(float(duration),.25)
    denominators=[]
    for group,gw,scale in zip(GROUPS,GROUP_WEIGHTS,floor):
        target=tangent[list(group)]
        energy=np.sum(np.asarray(gw)[:,None]*target**2,axis=0)
        denominators.append(float(np.sqrt(np.mean(energy)+scale*scale)))
    denominators=np.asarray(denominators)
    trials=[]
    for fraction in sorted(set(map(float,candidates))|{float(default)}):
        check()
        taper=np.exp(-(radius/(fraction*source_n/2))**8)
        approximate=np.empty_like(reference)
        for frame in unique:
            mask=frames==frame
            approximate[:,mask]=((moments[int(frame)]*taper[None])@phase[mask].T).real
        difference=central_tangent(raw,approximate-reference,center)
        energies=[]
        for group,gw,denominator in zip(GROUPS,GROUP_WEIGHTS,denominators):
            energies.append(np.sum(np.asarray(gw)[:,None]*difference[list(group)]**2,axis=0)/denominator**2)
        energies=np.asarray(energies)
        errors=np.sqrt(np.mean(energies,axis=1))
        by_time=np.asarray([np.sqrt(np.mean(energies[:,frames==frame],axis=1))@OBJECTIVE_WEIGHTS for frame in unique])
        by_half=np.asarray([np.sqrt(np.mean(energies[:,np.arange(len(frames))%2==i],axis=1))@OBJECTIVE_WEIGHTS for i in range(2)])
        trials.append({'fraction':fraction,'group_errors':errors.tolist(),
            'score_proxy':float(errors@OBJECTIVE_WEIGHTS),
            'score_proxy_by_time':by_time.tolist(),'score_proxy_by_half':by_half.tolist()})
    baseline=next(t for t in trials if t['fraction']==float(default))
    best=min(trials,key=lambda t:t['score_proxy'])
    # Independent spatial halves must both improve; no target may regress
    # substantially. Time blocks permit one nearly stationary/noisy endpoint.
    eps=2e-3
    improvements=np.asarray(best['score_proxy_by_time']) <= .98*np.asarray(baseline['score_proxy_by_time'])+eps
    accepted=(baseline['score_proxy']>=.002
        and best['fraction']>default
        and best['score_proxy'] <= .85*baseline['score_proxy']
        and np.all(np.asarray(best['group_errors']) <= 1.15*np.asarray(baseline['group_errors'])+eps)
        and np.all(np.asarray(best['score_proxy_by_half']) <= .90*np.asarray(baseline['score_proxy_by_half'])+eps)
        and np.count_nonzero(improvements)>=max(1,len(unique)-1)
        and np.all(np.asarray(best['score_proxy_by_time']) <= 1.10*np.asarray(baseline['score_proxy_by_time'])+eps))
    selected=best['fraction'] if accepted else float(default)
    check()
    return selected,{'selected_fraction':selected,'best_fraction':best['fraction'],
        'accepted_stronger_passband':bool(accepted),'default_fraction':float(default),
        'normalization':'central forcing RMS plus frozen transport scale / duration',
        'scope':'held-out public source defect, not a future macro accuracy certificate',
        'calibration_count':int(len(frames)), 'frames':unique.tolist(),'trials':trials}
