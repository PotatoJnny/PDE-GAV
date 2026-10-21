"""Stable low-order VPFP perturbation hierarchy with a frozen top boundary.

All velocity moments use w=v-center, while x remains the laboratory coordinate.
Source shape is (time,power,x,y); initial shape is (power,x,y).
This module owns no Oracle access and performs no query-dependent solves.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm
from scipy.interpolate import CubicSpline
from numpy.polynomial.hermite_e import herme2poly


def powers_through(degree):
    return [(a,d-a) for d in range(degree+1) for a in range(d,-1,-1)]


def raw_to_hermite_matrix(raw_powers, degree, sigma):
    target=powers_through(degree)
    lookup={tuple(p):i for i,p in enumerate(raw_powers)}
    out=np.zeros((len(target),len(raw_powers)))
    polys=[]
    for a in range(degree+1):
        c=np.zeros(a+1);c[a]=1
        polys.append(herme2poly(c)/(sigma**np.arange(a+1)*np.sqrt(math.factorial(a))))
    for i,(a,b) in enumerate(target):
        for j,cx in enumerate(polys[a]):
            for k,cy in enumerate(polys[b]):
                if cx*cy:
                    if (j,k) not in lookup: raise ValueError(f'missing raw moment {(j,k)}')
                    out[i,lookup[j,k]]+=cx*cy
    return out


def hermite_to_raw_matrix(degree,sigma):
    powers=powers_through(degree);out=np.zeros((len(powers),len(powers)))
    for i,(a,b) in enumerate(powers):
        for j,(c,d) in enumerate(powers):
            if c<=a and d<=b and (a-c)%2==0 and (b-d)%2==0:
                k=(a-c)//2;l=(b-d)//2
                out[i,j]=sigma**(a+b)*math.factorial(a)*math.factorial(b)/(2**(k+l)*math.factorial(k)*math.factorial(l)*np.sqrt(math.factorial(c)*math.factorial(d)))
    return out


def project_nyquist(modes):
    """In-place projection for a real continuous trigonometric polynomial."""
    nx,ny=modes.shape[-2:]
    if nx%2==0:modes[...,nx//2,:]=0
    if ny%2==0:modes[...,:,ny//2]=0
    return modes


def resize_fourier(modes, shape, normalized=False):
    """Resize a Fourier polynomial, preserving retained physical coefficients.

    Nyquist modes are excluded at both resolutions. ``normalized=False`` is
    for NumPy fft2 arrays; True is for Fourier-series coefficients.
    """
    modes=np.asarray(modes)
    nx,ny=modes.shape[-2:];mx,my=map(int,shape)
    out=np.zeros(modes.shape[:-2]+(mx,my),dtype=modes.dtype)
    kx=np.rint(np.fft.fftfreq(nx)*nx).astype(int)
    ky=np.rint(np.fft.fftfreq(ny)*ny).astype(int)
    tx={int(k):i for i,k in enumerate(np.rint(np.fft.fftfreq(mx)*mx).astype(int))}
    ty={int(k):i for i,k in enumerate(np.rint(np.fft.fftfreq(my)*my).astype(int))}
    scale=1. if normalized else (mx*my)/(nx*ny)
    for ix,k in enumerate(kx):
        if k not in tx or (nx%2==0 and ix==nx//2) or (mx%2==0 and tx[k]==mx//2):continue
        for iy,l in enumerate(ky):
            if l not in ty or (ny%2==0 and iy==ny//2) or (my%2==0 and ty[l]==my//2):continue
            out[...,tx[k],ty[l]]=modes[...,ix,iy]*scale
    return out


def evolve(initial, source, times, raw_powers, periods, center, nu, theta,
           degree=3, sigma=None, field='selfconsistent', substeps=8,
           output_times=None, deadline_check=None, dealias=True):
    """Evolve and return raw snapshots through degree N.

    ETD2 handles the entire constant streaming/collision matrix exactly.
    Only E*Hermite-lowering and source interpolation remain explicit. The
    degree N+1 initial coefficients provide the frozen closure boundary.
    Source may contain fewer powers than initial provided all degrees<=N exist.
    ``source`` can also be a (source_array, source_powers) pair.
    """
    initial=np.asarray(initial,float);times=np.asarray(times,float)
    raw_powers=[tuple(p) for p in raw_powers]
    if isinstance(source,tuple): source,source_powers=source
    else: source_powers=raw_powers
    source=np.asarray(source,float)
    nx,ny=initial.shape[-2:]
    center=np.asarray(center,float)
    if sigma is None:
        ix={p:i for i,p in enumerate(raw_powers)}
        mass=np.mean(initial[ix[0,0]])
        thermal=(np.mean(initial[ix[2,0]]+initial[ix[0,2]])/max(mass,1e-12))/2
        sigma=float(np.sqrt(max(theta,thermal,1e-4)))
    powers=powers_through(degree);full_powers=powers_through(degree+1)
    n=len(powers);lookup={p:i for i,p in enumerate(powers)}
    full_lookup={p:i for i,p in enumerate(full_powers)}
    convert=raw_to_hermite_matrix(raw_powers,degree+1,sigma)
    f0=np.einsum('ab,bxy->axy',convert,initial)
    src_convert=raw_to_hermite_matrix(source_powers,degree,sigma)
    sf=np.einsum('ab,tbxy->taxy',src_convert,source)
    f0h=np.fft.fft2(f0);sh=np.fft.fft2(sf)
    # The Fourier polynomial excludes ambiguous Nyquist planes consistently.
    project_nyquist(f0h);project_nyquist(sh)
    source_spline=CubicSpline(times,sh,axis=0)
    kx=2*np.pi*np.fft.fftfreq(nx,d=periods[0]/nx)
    ky=2*np.pi*np.fft.fftfreq(ny,d=periods[1]/ny)
    k2=kx[:,None]**2+ky[None,:]**2
    gradx=kx.copy();grady=ky.copy()
    if nx%2==0: gradx[nx//2]=0
    if ny%2==0: grady[ny//2]=0
    invk2=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    def electric_modes(y):
        return np.stack((-1j*gradx[:,None]*invk2*y[0],
                         -1j*grady[None,:]*invk2*y[0]))
    e0h=electric_modes(f0h)
    padded_shape=(int(np.ceil(1.5*nx)),int(np.ceil(1.5*ny)))
    lowerx=np.zeros((n,n));lowery=np.zeros((n,n))
    linear=np.zeros((nx,ny,n,n),complex);boundary=np.zeros((n,nx,ny),complex)
    for i,(a,b) in enumerate(powers):
        linear[:,:,i,i]=-1j*(kx[:,None]*center[0]+ky[None,:]*center[1])-nu*(a+b)
        for axis,order,k in ((0,a,kx[:,None]),(1,b,ky[None,:])):
            upper=(a+1,b) if axis==0 else (a,b+1)
            if upper in lookup: linear[:,:,i,lookup[upper]]+=-1j*k*sigma*np.sqrt(order+1)
            else: boundary[i]+=-1j*k*sigma*np.sqrt(order+1)*f0h[full_lookup[upper]]
            if order:
                lower=(a-1,b) if axis==0 else (a,b-1)
                j=lookup[lower]
                linear[:,:,i,j]+=-1j*k*sigma*np.sqrt(order)-nu*center[axis]/sigma*np.sqrt(order)
                (lowerx if axis==0 else lowery)[i,j]=np.sqrt(order)/sigma
            if order>=2:
                low2=(a-2,b) if axis==0 else (a,b-2)
                linear[:,:,i,lookup[low2]]+=nu*(theta/sigma**2-1)*np.sqrt(order*(order-1))
    if output_times is None:
        output_times=np.linspace(times[0],times[-1],max(129,len(times)))
    output_times=np.asarray(output_times,float)
    if output_times[0]!=times[0] or output_times[-1]!=times[-1] or np.any(np.diff(output_times)<=0):
        raise ValueError('output_times must strictly span source times')
    if field not in ('selfconsistent','frozen','none'): raise ValueError('unknown field treatment')
    def forcing(y,t):
        out=source_spline(t)+boundary
        if field!='none':
            eh=e0h if field=='frozen' else electric_modes(y)
            if dealias:
                fp=np.fft.ifft2(resize_fourier(y,padded_shape)).real
                ee=np.fft.ifft2(resize_fourier(eh,padded_shape)).real
            else:
                fp=np.fft.ifft2(y).real
                ee=np.fft.ifft2(eh).real
            force=np.einsum('ab,bxy->axy',lowerx,fp)*ee[0]+np.einsum('ab,bxy->axy',lowery,fp)*ee[1]
            forceh=np.fft.fft2(force)
            if dealias:forceh=resize_fourier(forceh,(nx,ny))
            out=out+forceh
        return project_nyquist(out)
    # Output intervals are split additionally relative to source spacing.
    hmax=float(np.min(np.diff(times)))/substeps
    matrices={}
    def operators(h):
        key=round(float(h),14)
        if key not in matrices:
            ee=np.empty_like(linear);p1=np.empty_like(linear);p2=np.empty_like(linear)
            for ix in range(nx):
                for iy in range(ny):
                    block=np.zeros((3*n,3*n),complex)
                    block[:n,:n]=h*linear[ix,iy]
                    block[:n,n:2*n]=h*np.eye(n)
                    block[n:2*n,2*n:]=h*np.eye(n)
                    eb=expm(block)
                    ee[ix,iy]=eb[:n,:n];p1[ix,iy]=eb[:n,n:2*n];p2[ix,iy]=eb[:n,2*n:]/h
            matrices[key]=(ee,p1,p2)
        return matrices[key]
    def mm(a,y):return np.einsum('xyab,bxy->axy',a,y,optimize=True)
    y=f0h[:n].copy();snapshots=[y.copy()];current=times[0]
    for target in output_times[1:]:
        count=max(1,int(np.ceil((target-current)/hmax-1e-12)))
        h=(target-current)/count;ee,p1,p2=operators(h)
        for step in range(count):
            if deadline_check is not None:deadline_check()
            f=forcing(y,current)
            pred=mm(ee,y)+mm(p1,f)
            fnew=forcing(pred,current+h)
            y=project_nyquist(pred+mm(p2,fnew-f))
            current+=h
        current=float(target)
        if not np.isfinite(y).all():raise FloatingPointError('nonfinite Hermite trajectory')
        snapshots.append(y.copy())
    back=hermite_to_raw_matrix(degree,sigma)
    snapshots=np.asarray(snapshots)
    result=np.einsum('ab,tbxy->taxy',back,np.fft.ifft2(snapshots).real)
    correction_modes=np.einsum('ab,tbxy->taxy',back,snapshots-f0h[None,:n])/(nx*ny)
    rho=result[:,0]
    return {'times':output_times,'moments':result,'powers':powers,'sigma':sigma,
            'correction_modes':correction_modes,
            'minimum_density':float(np.min(rho)),
            'maximum_hermite_amplitude':float(np.max(np.abs(snapshots)))}

