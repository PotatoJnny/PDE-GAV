"""Exploratory high-pass corner proposals, to be recertified by resampling."""
import numpy as np
from scipy.ndimage import median_filter

def propose_weak_corner_indices(times,values,ratio_threshold=6.):
    times=np.asarray(times,float);values=np.asarray(values,float)
    h=float(np.median(np.diff(times)))
    intervals=int(round((times[-1]-times[0])/h))
    if intervals<32 or intervals>2*len(times):return np.empty(0,int)
    targets=np.linspace(times[0],times[-1],intervals+1)
    right=np.minimum(np.searchsorted(times,targets),len(times)-1)
    left=np.maximum(right-1,0)
    regular=np.where(abs(times[left]-targets)<=abs(times[right]-targets),left,right)
    if np.max(abs(times[regular]-targets))>1.e-6*h:return np.empty(0,int)
    v=values[regular]
    fourth=v[4:]-4*v[3:-1]+6*v[2:-2]-4*v[1:-3]+v[:-4]
    magnitude=np.sqrt(np.mean(fourth**2,axis=tuple(range(1,fourth.ndim))))
    background=median_filter(magnitude,size=17,mode='nearest')
    floor=max(1.e-11,1.e-10*float(np.sqrt(np.mean(v*v))))
    selected=(magnitude>ratio_threshold*np.maximum(background,floor))
    selected[:4]=False;selected[-2:]=False
    return regular[np.flatnonzero(selected)+2]
