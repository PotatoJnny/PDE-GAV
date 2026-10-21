"""Resolve public-pilot source corners on successively smaller local meshes.

``sample`` accepts a finite 1-D time vector and returns its (time,probe,moment)
values. No scoring points or source parameterization is used.
"""
from __future__ import annotations
import numpy as np


def corner_windows(times, corner_indices):
    """Merge touching kink stencils and keep one pilot cell of padding."""
    times=np.asarray(times,float)
    corners=np.unique(np.asarray(corner_indices,int))
    corners=corners[(corners>0)&(corners<len(times)-1)]
    groups=[]
    for i in corners:
        if groups and i<=groups[-1][-1]+1:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [(times[g[0]-1],times[g[-1]+1]) for g in groups]


def local_corner_bracket(times, values, ratio_threshold=3.):
    """Return kink bracketing nodes, or None when curvature is not isolated.

    A C0 corner contributes an aligned rank-one impulse to adjacent second
    differences. Reconstruct its subcell location using impulse barycenters;
    use a full-cell bracket on each side for robust subsequent refinement.
    """
    times=np.asarray(times,float)
    values=np.asarray(values,float)
    h=float(times[1]-times[0])
    second=values[2:]-2*values[1:-1]+values[:-2]
    flat=second.reshape(len(second),-1)
    magnitude=np.linalg.norm(flat,axis=1)
    noise=max(1.e-12,1.e-10*np.linalg.norm(values.reshape(len(values),-1),axis=1).max())
    candidates=[]
    for j in range(1,len(second)-1):
        outside=max(magnitude[j-1],magnitude[j+1],noise)
        if magnitude[j]>ratio_threshold*outside:
            candidates.append((magnitude[j],times[j+1]))
    for j in range(1,len(second)-2):
        mass=magnitude[j]+magnitude[j+1]
        outside=magnitude[j-1]+magnitude[j+2]+noise
        if (mass>ratio_threshold*outside and
                np.dot(flat[j],flat[j+1])>.8*magnitude[j]*magnitude[j+1]):
            center=(magnitude[j]*times[j+1]+magnitude[j+1]*times[j+2])/mass
            candidates.append((mass,center))
    if not candidates:
        return None
    _,center=max(candidates)
    return (max(times[0],center-h),min(times[-1],center+h))


def refine_corner_pilot(times, values, corner_indices, sample,
                        rounds=3, points_per_window=17, trusted_corner_indices=None):
    """Augment a pilot and return narrow corner-bracket break indices.

    At 17 points/window, successful stages typically shrink the corner
    bracket by 4--8x after endpoints are snapped outward to sampled nodes.
    All windows at each stage are sampled together. If a feature ceases to
    be isolated, preserve its last certified bracket without extrapolating.
    By default the input corners are trusted. If extra exploratory proposals
    are supplied, ``trusted_corner_indices`` identifies the original joins
    that must survive; unconfirmed additional proposals are discarded.
    """
    times=np.asarray(times,float)
    values=np.asarray(values,float)
    if (times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0)
            or values.shape[0]!=len(times)):
        raise ValueError('ordered pilot times and matching values required')
    if points_per_window<9 or points_per_window%2!=1:
        raise ValueError('an odd number of at least nine local points required')
    windows=corner_windows(times,corner_indices)
    if not windows:
        return times,values,np.empty(0,dtype=int)
    bank={float(t):value for t,value in zip(times,values)}
    trusted=corner_indices if trusted_corner_indices is None else trusted_corner_indices
    original=times[np.asarray(trusted,dtype=int)]
    accepted=[original[(original>left)&(original<right)] for left,right in windows]
    active=list(range(len(windows)))
    for _ in range(int(rounds)):
        if not active:break
        meshes={i:np.linspace(*windows[i],points_per_window) for i in active}
        new=np.unique(np.concatenate(list(meshes.values())))
        new=np.array([t for t in new if float(t) not in bank])
        if len(new):
            observed=np.asarray(sample(new),float)
            if observed.shape!=(len(new),)+values.shape[1:] or not np.isfinite(observed).all():
                raise ValueError('invalid public pilot sampler result')
            bank.update((float(t),value) for t,value in zip(new,observed))
        continuing=[]
        for i in active:
            mesh=meshes[i]
            local=np.array([bank[float(t)] for t in mesh])
            bracket=local_corner_bracket(mesh,local)
            if bracket is not None:
                # Keep sample-aligned endpoints so acquisition needs no
                # source estimation at the piecewise interpolation joins.
                lo=max(0,int(np.searchsorted(mesh,bracket[0],side='right')-1))
                hi=min(len(mesh)-1,int(np.searchsorted(mesh,bracket[1],side='left')))
                windows[i]=(float(mesh[lo]),float(mesh[hi]))
                # Preserve an existing exact-node corner whenever it lies
                # in the refined bracket; otherwise replace coarse joins.
                retained=original[(original>=mesh[lo])&(original<=mesh[hi])]
                accepted[i]=np.unique(np.r_[mesh[lo],retained,mesh[hi]])
                continuing.append(i)
        active=continuing
    merged_times=np.array(sorted(bank))
    merged_values=np.array([bank[float(t)] for t in merged_times])
    corner_times=np.unique(np.concatenate(accepted))
    indices=np.searchsorted(merged_times,corner_times)
    return merged_times,merged_values,indices
