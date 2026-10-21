"""Data-driven temporal source corners and progressive piecewise sampling.

Uses only supplied public source pilots. No Oracle access or assumed task
time partition is used. Values are normalized theta-Hermite moments 00..03.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import CubicSpline


def detect_corner_indices(times, values, ratio_threshold=3.):
    """Find isolated slope jumps on the solver-owned regular pilot submesh.

    A corner between pilot nodes is bracketed by two returned nodes; its
    subinterval remains linear unless further source samples are acquired.
    Thus corner localization is limited by the public pilot spacing.
    """
    times=np.asarray(times,dtype=float)
    values=np.asarray(values,dtype=float)
    if len(times)<7 or values.shape[0]!=len(times):
        return np.empty(0,dtype=np.int64)
    spacing=float(np.median(np.diff(times)))
    if not spacing>0:
        raise ValueError("strictly increasing pilot times required")
    intervals=int(round((times[-1]-times[0])/spacing))
    if intervals<6 or intervals>2*len(times):
        return np.empty(0,dtype=np.int64)
    targets=np.linspace(times[0],times[-1],intervals+1)
    right=np.minimum(np.searchsorted(times,targets),len(times)-1)
    left=np.maximum(right-1,0)
    regular=np.where(abs(times[left]-targets)<=abs(times[right]-targets),left,right)
    # An irregular pilot cloud cannot support this isolated-jump detector.
    if np.max(abs(times[regular]-targets))>1.e-6*spacing:
        return np.empty(0,dtype=np.int64)
    data=values[regular]
    second=data[2:]-2*data[1:-1]+data[:-2]
    size=np.sqrt(np.mean(second*second,axis=tuple(range(1,second.ndim))))
    neighbor=np.maximum(np.r_[size[0],size[:-1]],np.r_[size[1:],size[-1]])
    noise_floor=max(1.e-10,1.e-7*float(np.sqrt(np.mean(data*data))))
    detected=(size>ratio_threshold*neighbor)&(size>noise_floor)
    # A corner away from a pilot node shares its second difference between
    # two adjacent nodes. Test their combined mass against outside neighbors,
    # requiring aligned vector jumps to reject smooth oscillatory extrema.
    isolated=detected.copy()
    for i in range(1,len(size)-2):
        mass=size[i]+size[i+1]
        if (isolated[i] or isolated[i+1] or mass<2*noise_floor
                or min(size[i],size[i+1])<.15*mass
                or mass<=ratio_threshold*(size[i-1]+size[i+2])):
            continue
        product=float(np.mean(second[i]*second[i+1]))
        if product>.8*size[i]*size[i+1]:
            detected[i:i+2]=True
    return regular[np.flatnonzero(detected)+1].astype(np.int64)


def detect_corner_times(times,values,theta=None,nu=None):
    """Convenience wrapper returning physical times; theta/nu are unused."""
    times=np.asarray(times,dtype=float)
    return times[detect_corner_indices(times,values)]


def _channel_map(theta):
    matrix=np.zeros((10,8))
    scale=math.sqrt(theta)
    matrix[0,0]=4/1.0041253663782175
    matrix[1,1]=matrix[2,2]=scale/.4788345668393978
    matrix[0,3]=matrix[0,5]=theta/2.930763214320111
    matrix[3,3]=matrix[4,4]=matrix[5,5]=theta*math.sqrt(2)/2.930763214320111
    heat=.5*scale**3/.9900774471118267
    matrix[6,6],matrix[8,6],matrix[1,6]=heat*math.sqrt(6),heat*math.sqrt(2),4*heat
    matrix[9,7],matrix[7,7],matrix[2,7]=heat*math.sqrt(6),heat*math.sqrt(2),4*heat
    return matrix


def select_corner_knot_order(times,values,theta,nu,target_count,corners=None):
    """Endpoints/corners/early layer first, then OU-weighted segment defects.

    Returns acquisition order. Sort an acquired prefix to construct a spline.
    If the budget admits fewer nodes than the seed, endpoints and broad
    coverage are retained before remaining corners and initial-layer nodes.
    """
    times=np.asarray(times,dtype=float)
    values=np.asarray(values,dtype=float)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or values.ndim!=3 or values.shape[0]!=len(times) or values.shape[-1]!=10
            or not np.isfinite(values).all() or theta<=0 or nu<0):
        raise ValueError("ordered finite times and finite theta-basis (time,probe,10) values required")
    count=min(int(target_count),len(times))
    if count<=0:
        return np.empty(0,dtype=np.int64)
    if corners is None:
        corners=detect_corner_indices(times,values)
    corners=np.unique(np.asarray(corners,dtype=np.int64))
    corners=corners[(corners>0)&(corners<len(times)-1)]
    order=[]
    def append(index):
        index=int(index)
        if index not in order and len(order)<count:
            order.append(index)
    for fraction in [0.,1.]+[odd/(2**level) for level in range(1,5) for odd in range(1,2**level,2)]:
        append(np.argmin(abs(times-(times[0]+fraction*(times[-1]-times[0])))))
    # Farthest first protects an interrupted acquisition from a late-time gap.
    remaining=list(map(int,corners))
    while remaining and len(order)<count:
        distance=np.min(abs(times[np.array(remaining),None]-times[np.array(order)][None,:]),axis=1)
        append(remaining.pop(int(np.argmax(distance))))
    fraction=(times-times[0])/(times[-1]-times[0])
    early=list(map(int,np.flatnonzero(fraction<1/256)))
    positive=fraction[fraction>0]
    logs=np.log(np.maximum(fraction,positive[0]*.5))
    while early and len(order)<count:
        distance=np.min(abs(logs[np.array(early),None]-logs[np.array(order)][None,:]),axis=1)
        append(early.pop(int(np.argmax(distance))))
    if len(order)==count:
        return np.asarray(order,dtype=np.int64)
    chosen=np.zeros(len(times),dtype=bool);chosen[order]=True
    boundaries=np.r_[0,corners,len(times)-1]
    if not np.all(chosen[boundaries]):
        raise ArithmeticError("source corner seeds not acquired")
    rates=nu*np.array([0.,1.,1.,2.,2.,2.,3.,3.,3.,3.])
    mapping=_channel_map(theta)
    indicator=np.zeros(len(times))
    def refresh(segment):
        left,right=boundaries[segment:segment+2]
        query=np.arange(left,right+1)
        selected=query[chosen[query]]
        prediction=CubicSpline(times[selected],values[selected],axis=0)(times[query])
        slots=np.clip(np.searchsorted(times[selected],times[query],side="right")-1,0,len(selected)-2)
        widths=times[selected[slots+1]]-times[selected[slots]]
        transfer=np.broadcast_to(widths[:,None],(len(query),10)).copy()
        np.divide(-np.expm1(-widths[:,None]*rates),rates,out=transfer,where=rates>0)
        error=(prediction-values[query])*transfer[:,None,:]
        channels=error@mapping
        indicator[query]=np.max(np.mean(channels*channels,axis=1),axis=-1)
    for segment in range(len(boundaries)-1):
        refresh(segment)
    while len(order)<count:
        indicator[chosen]=-1
        if np.max(indicator)<=1.e-28:
            selected=np.flatnonzero(chosen)
            distance=np.min(abs(times[:,None]-times[selected][None,:]),axis=1)
            distance[chosen]=-1
            index=int(np.argmax(distance))
        else:
            index=int(np.argmax(indicator))
        order.append(index);chosen[index]=True
        segment=int(np.clip(np.searchsorted(boundaries,index,side="right")-1,0,len(boundaries)-2))
        refresh(segment)
    return np.asarray(order,dtype=np.int64)
