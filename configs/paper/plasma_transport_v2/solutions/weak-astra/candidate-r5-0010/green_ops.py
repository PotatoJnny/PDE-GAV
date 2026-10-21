"""Continuous free-OU macro response to natural-cubic source histories.

No velocity distribution is evolved. Velocity quadrature is used only to
integrate analytic Ornstein--Uhlenbeck Green functions. All source Fourier
coefficients use physical velocity coordinates; output moments use v-bulk.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, CancelledError
import os
import threading
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.linalg import solve_banded

PAIRS=tuple((degree-b,b) for degree in range(4) for b in range(degree+1))


def natural_curvature_coefficients(source):
    """Return h**2 S''/6 for the natural spline on any uniform time mesh.

    Keeping h out of the system avoids dividing small temporal differences by
    h**2 only to multiply by h**2 during the convolution. Endpoints are zero.
    """
    source=np.asarray(source)
    if source.ndim<1 or source.shape[0]<2:
        raise ValueError('at least two source times required')
    result=np.zeros(source.shape,dtype=np.result_type(source.dtype,np.float64))
    interior=source.shape[0]-2
    if interior:
        bands=np.ones((3,interior));bands[1]=4.
        rhs=source[2:]-2*source[1:-1]+source[:-2]
        result[1:-1]=solve_banded((1,1),bands,rhs.reshape(interior,-1),
                                overwrite_ab=True,overwrite_b=True,
                                check_finite=False).reshape(rhs.shape)
    return result


def ou_parameters(tau,nu,theta):
    """Return decay, integrated decay, velocity variance, VX covariance, X variance."""
    tau=np.asarray(tau,dtype=float);nu=float(nu);theta=float(theta)
    if nu<0 or theta<=0 or np.any(tau<0) or not np.isfinite(tau).all():
        raise ValueError('nonnegative OU time/nu and positive theta required')
    if nu==0:
        zero=np.zeros_like(tau)
        return np.ones_like(tau),tau.copy(),zero,zero.copy(),zero.copy()
    z=nu*tau;decay=np.exp(-z);a=-np.expm1(-z)/nu
    variance=theta*(-np.expm1(-2*z));cross=theta*nu*a*a
    # The direct displacement variance loses all significant digits at zero.
    # Taylor coefficients start at z^3; use theta*nu*tau^3 to avoid nu^-2.
    p=(2/3+z*(-1/2+z*(7/30+z*(-1/12+z*(31/1260+z*(-1/160+z*(127/90720-z*17/60480)))))))
    displacement=np.where(z<.1,theta*nu*tau**3*p,
                          theta/nu**2*(2*z+4*np.expm1(-z)-np.expm1(-2*z)))
    displacement=np.maximum(displacement,0.)
    return decay,a,variance,cross,displacement


def _kernel_terms(tau,wavevectors,velocity,bulk_center,nu,theta,density_only):
    tau=np.atleast_1d(np.asarray(tau,dtype=float))
    k=np.asarray(wavevectors,dtype=float);v=np.asarray(velocity,dtype=float)
    decay,a,variance,cross,displacement=ou_parameters(tau,nu,theta)
    dot=k@v.T;k2=np.sum(k*k,axis=1)
    phase=np.exp(-1j*a[:,None,None]*dot[None]-.5*displacement[:,None,None]*k2[None,:,None])
    yield phase
    if density_only:return
    bulk=np.asarray(bulk_center,dtype=float)
    mx=decay[:,None,None]*v[None,None,:,0]-bulk[0]-1j*cross[:,None,None]*k[None,:,None,0]
    my=decay[:,None,None]*v[None,None,:,1]-bulk[1]-1j*cross[:,None,None]*k[None,:,None,1]
    var=variance[:,None,None]
    yield phase*mx
    yield phase*my
    xx=mx*mx+var;xy=mx*my;yy=my*my+var
    yield phase*xx
    yield phase*xy
    yield phase*yy
    yield phase*(mx*mx*mx+3*var*mx)
    yield phase*xx*my
    yield phase*mx*yy
    yield phase*(my*my*my+3*var*my)


def ou_kernel(tau,wavevectors,velocity,bulk_center,nu,theta,*,density_only=False):
    """Return analytic kernels shaped (moments, times, modes, velocities)."""
    return np.stack(list(_kernel_terms(tau,wavevectors,velocity,bulk_center,nu,theta,density_only)))


def _serial_green_response(source_modes,source_times,velocity,integration,wavevectors,
                   bulk_center,nu,theta,*,density_only=False,mode_block=64,
                   quadrature_order=4,check=None,kernel_model=None):
    """Integrate a natural-cubic history against continuous free-OU kernels.

    source_modes: complex (nt, n_modes, n_vel), unweighted Fourier coefficients.
    source_times: uniformly spaced physical times; only their differences matter.
    velocity: physical (n_vel,2); integration: Lebesgue quadrature weights.
    wavevectors: physical Fourier wave numbers (n_modes,2).
    Return: complex (nt, 1 or 10, n_modes), raw moments about bulk_center.

    Gauss integration uses at least quadrature_order nodes on each half-hat.
    The order increases with the largest physical streaming phase per step
    in each mode block (4/8/12 nodes through phases 4/12/20, then growing).
    The causal
    convolution uses full interior hats and curvature hats. The extra half-hat
    multiplying source_modes[0] is removed. Natural curvature vanishes at both
    endpoints, so there is no corresponding initial curvature correction.
    """
    check=check or (lambda:None);check()
    source=np.asarray(source_modes);times=np.asarray(source_times,float)
    v=np.asarray(velocity,float);weights=np.asarray(integration,float)
    k=np.asarray(wavevectors,float);bulk=np.asarray(bulk_center,float)
    if source.ndim!=3 or len(times)<2 or times.ndim!=1:
        raise ValueError('source must have (time,mode,velocity) shape and at least two times')
    nt,nm,nv=source.shape
    if times.shape!=(nt,) or v.shape!=(nv,2) or weights.shape!=(nv,) or k.shape!=(nm,2) or bulk.shape!=(2,):
        raise ValueError('incompatible Green response axes')
    step=float(times[1]-times[0])
    if step<=0 or not np.allclose(np.diff(times),step,rtol=1e-10,atol=1e-13*step):
        raise ValueError('Green response requires uniform increasing source times')
    if not all(np.isfinite(a).all() for a in (source,times,v,weights,k,bulk)):
        raise FloatingPointError('finite Green response inputs required')
    if np.any(weights<0) or mode_block<1 or quadrature_order<1:
        raise ValueError('positive quadrature controls and nonnegative velocity weights required')
    if kernel_model is not None:
        # The alternative cardinal kernel includes its own physical dv rule.
        weights=np.ones_like(weights)
    moments=1 if density_only else len(PAIRS)
    result=np.zeros((nt,moments,nm),dtype=np.complex128)
    nfft=next_fast_len(2*nt-1);lags=np.arange(nt,dtype=float)
    for start in range(0,nm,int(mode_block)):
        check();stop=min(nm,start+int(mode_block));kb=k[start:stop];count=stop-start
        beta=float(np.max(np.abs(kb@v.T)))*step
        adaptive_order=4 if beta<=4 else 8 if beta<=12 else 12 if beta<=20 else int(np.ceil(beta/2))+2
        used_order=max(int(quadrature_order),adaptive_order)
        roots,gauss_weights=np.polynomial.legendre.leggauss(used_order)
        offsets=(roots+1)/2;gauss_weights=gauss_weights/2
        weighted_source=np.asarray(source[:,start:stop],dtype=np.complex128)*weights[None,None,:]
        initial_source=weighted_source[0].copy()
        curvature=natural_curvature_coefficients(weighted_source)
        curvature_fft=fft(curvature,n=nfft,axis=0,workers=1,overwrite_x=True)
        del curvature
        source_fft=fft(weighted_source,n=nfft,axis=0,workers=1,overwrite_x=True)
        del weighted_source
        hats=np.zeros((moments,nt,count,nv),dtype=np.complex128)
        curvature_hats=np.zeros_like(hats)
        initial_extra=np.zeros((moments,nt,count),dtype=np.complex128)
        for right in (True,False):
            for offset,gauss_weight in zip(offsets,gauss_weights):
                check()
                tau=(lags+offset if right else np.maximum(lags-1+offset,0))*step
                hat_weight=np.full(nt,step*gauss_weight*((1-offset) if right else offset))
                coordinate=(1-offset) if right else offset
                curvature_weight=np.full(nt,step*gauss_weight*(coordinate**3-coordinate))
                if not right:hat_weight[0]=0.
                if not right:curvature_weight[0]=0.
                terms=(_kernel_terms(tau,kb,v,bulk,nu,theta,density_only) if kernel_model is None
                       else kernel_model.iter_kernel_terms(tau,kb,density_only=density_only))
                for moment,kernel in enumerate(terms):
                    weighted_kernel=kernel*hat_weight[:,None,None]
                    hats[moment]+=weighted_kernel
                    curvature_hats[moment]+=kernel*curvature_weight[:,None,None]
                    if right:
                        initial_extra[moment]+=np.einsum('tbv,bv->tb',weighted_kernel,initial_source,optimize=True)
        for moment in range(moments):
            check()
            kernel_fft=fft(hats[moment],n=nfft,axis=0,workers=1,overwrite_x=True)
            response_fft=np.einsum('tbv,tbv->tb',kernel_fft,source_fft,optimize=True)
            curvature_kernel_fft=fft(curvature_hats[moment],n=nfft,axis=0,workers=1,overwrite_x=True)
            response_fft+=np.einsum('tbv,tbv->tb',curvature_kernel_fft,curvature_fft,optimize=True)
            response=ifft(response_fft,axis=0,workers=1)[:nt]
            result[:,moment,start:stop]=response-initial_extra[moment]
        del hats,curvature_hats,initial_extra,source_fft,curvature_fft
    result[0]=0.
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite Green response')
    check();return result


def available_workers():
    """Respect the execution service's CPU affinity and cap concurrency at four."""
    try:count=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):count=os.cpu_count() or 1
    return max(1,min(4,int(count)))


def green_response(source_modes,source_times,velocity,integration,wavevectors,
                   bulk_center,nu,theta,*,density_only=False,mode_block=64,
                   quadrature_order=4,check=None,kernel_model=None,max_workers=None):
    """Evaluate the unchanged serial algorithm on independent Fourier blocks.

    Only the caller thread invokes the SDK-backed check. Worker checks consult
    a cancellation event; on failure, pending work is cancelled and all running
    blocks are joined before the exception is returned to preparation.
    """
    check=check or (lambda:None);check()
    source=np.asarray(source_modes);wavevectors=np.asarray(wavevectors,float)
    if source.ndim!=3 or mode_block<1:
        raise ValueError('source must have time/mode/velocity axes and positive mode block')
    block_size=int(mode_block);nm=source.shape[1]
    starts=list(range(0,nm,block_size))
    permitted=available_workers()
    if max_workers is not None:
        if isinstance(max_workers,bool) or int(max_workers)!=max_workers or max_workers<1:
            raise ValueError('positive integer max_workers required')
        permitted=min(permitted,int(max_workers))
    workers=min(permitted,max(1,len(starts)))
    if workers==1:
        return _serial_green_response(source,source_times,velocity,integration,wavevectors,
                bulk_center,nu,theta,density_only=density_only,mode_block=block_size,
                quadrature_order=quadrature_order,check=check,kernel_model=kernel_model)
    # Preserve the serial block boundaries exactly, including the last partial block.
    result=np.empty((source.shape[0],1 if density_only else len(PAIRS),nm),complex)
    cancelled=threading.Event()
    def worker_check():
        if cancelled.is_set():raise CancelledError('parallel Green response cancelled')
    def work(start):
        stop=min(nm,start+block_size)
        answer=_serial_green_response(source[:,start:stop],source_times,velocity,integration,
                wavevectors[start:stop],bulk_center,nu,theta,density_only=density_only,
                mode_block=block_size,quadrature_order=quadrature_order,
                check=worker_check,kernel_model=kernel_model)
        return start,stop,answer
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='green-mode')
    active=set();next_block=0
    try:
        while next_block<len(starts) and len(active)<workers:
            check();active.add(pool.submit(work,starts[next_block]));next_block+=1
        while active:
            check()
            completed,active=wait(active,timeout=.05,return_when=FIRST_COMPLETED)
            for future in completed:
                start,stop,answer=future.result()
                result[:,:,start:stop]=answer
            check()
            while next_block<len(starts) and len(active)<workers:
                check();active.add(pool.submit(work,starts[next_block]));next_block+=1
        check()
    except BaseException:
        cancelled.set()
        for future in active:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    return result
