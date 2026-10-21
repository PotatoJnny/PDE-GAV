"""CUDA evaluation of the same continuous OU/cardinal-spline response.

Kernels are built before projection and cached only within one preparation.
Complex128 arithmetic preserves the CPU quadrature convention.
"""
from __future__ import annotations
from itertools import islice
import numpy as np
import torch
from scipy.fft import next_fast_len
from green_ops import ou_parameters, PAIRS, natural_cubic_coefficients, green_response as cpu_response


CUDA_CACHE_BYTES=30*1024**3


def _tensor(value):
    return torch.as_tensor(value, device='cuda')


def _integrals(model,frequency,*,density_only=False):
    phase=float(torch.max(torch.abs(frequency)).item())*float(np.max(np.diff(model.z)))
    order=max(model.quadrature_order,int(np.ceil(phase/2))+4)
    locations,tensor=model._quadrature(order)
    powers=1 if density_only else 4
    tensor=tensor[:powers]
    locations=_tensor(locations);tensor=_tensor(tensor).to(torch.complex128)
    shape=frequency.shape
    oscillation=torch.exp(-1j*frequency.reshape(-1,1,1)*locations[None])
    local=torch.einsum('fcq,plcq->fplc',oscillation,tensor)
    direct=torch.zeros((oscillation.shape[0],powers,len(model.z)),device='cuda',dtype=torch.complex128)
    curvature=torch.zeros_like(direct)
    direct[:,:,:-1]+=local[:,:,0];direct[:,:,1:]+=local[:,:,1]
    curvature[:,:,:-1]+=local[:,:,2];curvature[:,:,1:]+=local[:,:,3]
    values=(direct+curvature@_tensor(model.second_derivatives).to(torch.complex128)).reshape(tuple(shape)+(powers,len(model.z)))
    if getattr(model,'axis_blend',False):
        weight=-torch.expm1(-(torch.abs(frequency)*float(np.min(np.diff(model.z)))/(np.pi*getattr(model,'blend_scale',1.)))**getattr(model,'blend_power',8.))
        z=_tensor(model.z)
        gh=torch.exp(-1j*frequency[...,None]*z)*_tensor(model.gh_axis_integration)
        gh=gh[...,None,:]*torch.stack([z**p for p in range(powers)])
        values=weight[...,None,None]*values+(1-weight[...,None,None])*gh
    return values


def _multiply(left,right):
    out={}
    for (a,b),x in left.items():
        for (c,d),y in right.items():
            pair=(a+c,b+d);out[pair]=out.get(pair,0.)+x*y
    return out


def _polynomials(linear,variance):
    square=_multiply(linear,linear)
    second=dict(square);second[0,0]=second.get((0,0),0.)+variance
    third=_multiply(square,linear)
    for power,value in linear.items():third[power]=third.get(power,0.)+3*variance*value
    return [{(0,0):torch.ones_like(variance)},linear,second,third]


def _terms(model,tau,k,*,density_only=False):
    decay,a,variance,cross,displacement=map(_tensor,ou_parameters(tau,model.nu,model.theta))
    transform=_tensor(model.transform);center=_tensor(model.center);bulk=_tensor(model.bulk)
    rotated=k@transform
    ix=_integrals(model,a[:,None]*rotated[None,:,0],density_only=density_only)
    iy=_integrals(model,a[:,None]*rotated[None,:,1],density_only=density_only)
    phase=model.jacobian*torch.exp(-1j*a[:,None]*(k@center)[None]-.5*displacement[:,None]*torch.sum(k*k,dim=1)[None])
    global_blend=model.gh_blend and not getattr(model,'axis_blend',False)
    if global_blend:
        dimensionless=torch.amax(torch.abs(a[:,None,None]*rotated[None]),dim=-1)*float(np.min(np.diff(model.z)))
        spline_weight=-torch.expm1(-(dimensionless/(np.pi*getattr(model,'blend_scale',1.)))**getattr(model,'blend_power',8.))
        velocity=_tensor(model.physical_velocity)
        gh_phase=torch.exp(-1j*a[:,None,None]*(k@velocity.T)[None]-.5*displacement[:,None,None]*torch.sum(k*k,dim=1)[None,:,None])
        gh_weight=_tensor(model.gh_integration)
        if not density_only:
            mx=decay[:,None,None]*velocity[None,None,:,0]-bulk[0]-1j*cross[:,None,None]*k[None,:,None,0]
            my=decay[:,None,None]*velocity[None,None,:,1]-bulk[1]-1j*cross[:,None,None]*k[None,:,None,1]
            var=variance[:,None,None]
            px=[torch.ones_like(mx),mx,mx*mx+var,mx*mx*mx+3*var*mx]
            py=[torch.ones_like(my),my,my*my+var,my*my*my+3*var*my]
    if density_only:
        result=(phase[:,:,None,None]*ix[:,:,0,:,None]*iy[:,:,0,None,:]).reshape(len(tau),len(k),-1)
        if global_blend:
            result=spline_weight[:,:,None]*result+(1-spline_weight[:,:,None])*gh_phase*gh_weight[None,None]
        yield result
        return
    variance=torch.broadcast_to(variance[:,None],phase.shape)
    polynomials=[]
    for axis in range(2):
        linear={(0,0):decay[:,None]*center[axis]-bulk[axis]-1j*cross[:,None]*k[None,:,axis],
                (1,0):torch.broadcast_to(decay[:,None]*transform[axis,0],phase.shape),
                (0,1):torch.broadcast_to(decay[:,None]*transform[axis,1],phase.shape)}
        polynomials.append(_polynomials(linear,variance))
    for degree_x,degree_y in PAIRS:
        polynomial=_multiply(polynomials[0][degree_x],polynomials[1][degree_y])
        result=torch.zeros(tuple(phase.shape)+(len(model.z),len(model.z)),device='cuda',dtype=torch.complex128)
        for q in range(degree_x+degree_y+1):
            left=torch.zeros(tuple(phase.shape)+(len(model.z),),device='cuda',dtype=torch.complex128)
            for (p,power_y),coefficient in polynomial.items():
                if power_y==q:left+=coefficient[:,:,None]*ix[:,:,p,:]
            result+=left[:,:,:,None]*iy[:,:,q,None,:]
        result=(phase[:,:,None,None]*result).reshape(len(tau),len(k),-1)
        if global_blend:
            gh=gh_phase*px[degree_x]*py[degree_y]*gh_weight[None,None]
            result=spline_weight[:,:,None]*result+(1-spline_weight[:,:,None])*gh
        yield result


def green_response(source_modes,source_times,velocity,integration,wavevectors,
                   bulk_center,nu,theta,*,density_only=False,mode_block=64,
                   quadrature_order=4,check=None,kernel_model=None,max_workers=None,
                   time_interpolation='cubic'):
    """Cache OU convolutions on assigned cuda:0; return ordinary host arrays."""
    if not torch.cuda.is_available() or kernel_model is None:
        return cpu_response(source_modes,source_times,velocity,integration,wavevectors,bulk_center,nu,theta,
            density_only=density_only,mode_block=mode_block,quadrature_order=quadrature_order,check=check,
            kernel_model=kernel_model,max_workers=max_workers,time_interpolation=time_interpolation)
    check=check or (lambda:None);check()
    source=np.asarray(source_modes);times=np.asarray(source_times,float);v=np.asarray(velocity,float);k=np.asarray(wavevectors,float)
    nt,nm,nv=source.shape;step=float(times[1]-times[0]);nfft=next_fast_len(2*nt-1)
    if len(times)!=nt or nt<2 or step<=0 or not np.allclose(np.diff(times),step):raise ValueError('uniform time required')
    if not all(np.isfinite(x).all() for x in (source,times,v,k)):raise FloatingPointError('nonfinite Green inputs')
    if time_interpolation not in ('cubic','linear'):
        raise ValueError('time_interpolation must be cubic or linear')
    cubic=time_interpolation=='cubic'
    # A complete cache can exceed device capacity on large spatial meshes.
    # Keep all moments only when their aggregate fits; otherwise retain the
    # density kernels used by Picard passes and build final full-moment blocks
    # transiently. Exceptionally large density histories use no persistent
    # cache. All three policies evaluate the same kernels and contractions.
    bytes_per_moment=(nfft+nt+(nfft if cubic else 0))*nm*nv*16
    cache_moments=10 if 10*bytes_per_moment<=CUDA_CACHE_BYTES else 1 if bytes_per_moment<=CUDA_CACHE_BYTES else 0
    key=(nt,nm,nv,step,int(mode_block),int(quadrature_order),k.tobytes(),time_interpolation,cache_moments,CUDA_CACHE_BYTES)
    if getattr(kernel_model,'_cuda_cache_key',None)!=key:
        kernel_model._cuda_cache_key=key;kernel_model._cuda_cache={}
    cache=kernel_model._cuda_cache
    kernel_model._cuda_cache_policy=('full' if cache_moments==10 else 'density' if cache_moments==1 else 'none')
    kernel_model._cuda_cache_estimated_bytes=cache_moments*bytes_per_moment
    kernel_model._cuda_cache_limit_bytes=CUDA_CACHE_BYTES
    moments=1 if density_only else 10
    out=np.zeros((nt,moments,nm),complex)
    lags=np.arange(nt,dtype=float)
    for start in range(0,nm,int(mode_block)):
        check();stop=min(nm,start+int(mode_block));count=stop-start
        entry=cache.get(start)
        if entry is None or entry[0].shape[0]<moments:
            build_moments=10 if cache_moments==10 else moments
            kb=_tensor(k[start:stop]);beta=float(np.max(np.abs(k[start:stop]@v.T)))*step
            used_order=max(int(quadrature_order),4 if beta<=4 else 8 if beta<=12 else 12 if beta<=20 else int(np.ceil(beta/2))+2)
            roots,gauss=np.polynomial.legendre.leggauss(used_order);offsets=(roots+1)/2;gauss=gauss/2
            hats=torch.zeros((build_moments,nt,count,nv),device='cuda',dtype=torch.complex128)
            right_hats=torch.zeros_like(hats)
            if cubic:cubic_hats=torch.zeros_like(hats)
            # Left-half lag j uses precisely the right-half kernel at j-1.
            # Reuse that value rather than repeating the continuous OU integral.
            for offset,weight in zip(offsets,gauss):
                check();tau=(lags+offset)*step
                right_weight=step*weight*(1-offset);left_weight=step*weight*offset
                for moment,kernel in enumerate(islice(_terms(kernel_model,tau,kb,
                                      density_only=(build_moments==1)),build_moments)):
                    weighted=kernel*right_weight
                    hats[moment]+=weighted
                    hats[moment,1:]+=kernel[:-1]*left_weight
                    right_hats[moment]+=weighted
                    if cubic:
                        cubic_hats[moment]+=kernel*(step*weight*((1-offset)**3-(1-offset)))
                        cubic_hats[moment,1:]+=kernel[:-1]*(step*weight*(offset**3-offset))
            fft_hats=torch.fft.fft(hats,n=nfft,dim=1)
            fft_cubic=torch.fft.fft(cubic_hats,n=nfft,dim=1) if cubic else None
            if cache_moments>=build_moments:
                cache[start]=(fft_hats,right_hats,fft_cubic)
            elif cache_moments==1 and entry is None:
                # A full-response-first caller can still leave a density-only
                # cache. clone avoids retaining the full tensors' storage.
                cache[start]=(fft_hats[:1].clone(),right_hats[:1].clone(),
                              fft_cubic[:1].clone() if cubic else None)
            del hats
            if cubic:del cubic_hats
            del kernel,weighted
            torch.cuda.synchronize();check()
        else:fft_hats,right_hats,fft_cubic=entry
        src=_tensor(np.ascontiguousarray(source[:,start:stop]));source_fft=torch.fft.fft(src,n=nfft,dim=0)
        response_fft=torch.sum(fft_hats[:moments]*source_fft[None],dim=-1)
        if cubic:
            coefficients=_tensor(natural_cubic_coefficients(np.ascontiguousarray(source[:,start:stop])))
            cubic_source_fft=torch.fft.fft(coefficients,n=nfft,dim=0)
            response_fft+=torch.sum(fft_cubic[:moments]*cubic_source_fft[None],dim=-1)
        response=torch.fft.ifft(response_fft,dim=1)[:,:nt]
        response-=torch.sum(right_hats[:moments]*src[0][None,None],dim=-1)
        out[:,:,start:stop]=response.permute(1,0,2).cpu().numpy()
        del entry,fft_hats,right_hats,fft_cubic,src,source_fft,response_fft,response
        if cubic:del coefficients,cubic_source_fft
    out[0]=0.
    if not np.isfinite(out).all():raise FloatingPointError('nonfinite CUDA response')
    check();return out


def clear_cache(kernel_model):
    if hasattr(kernel_model,'_cuda_cache'):
        kernel_model._cuda_cache.clear()
    torch.cuda.empty_cache()
