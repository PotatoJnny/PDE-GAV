"""D4 Hermite response with independently supplied degree5 raw prior closure."""
from __future__ import annotations
import math
import numpy as np
from hermite_response import (powers,hermite_transform,raw_to_hermite,
    _normalized_fft,_grid,_project,_matrix_functions_batched,brownian_prior)

P4=powers(4)
P5=powers(5)


def moment_rhs(prior,source,periods,center,nu,theta):
    prior=np.asarray(prior,float);source=np.asarray(source,float)
    if prior.ndim!=4 or prior.shape[1]!=21 or source.shape!=prior[:,:15].shape:
        raise ValueError('need prior (nt,21,nx,ny) and source (nt,15,nx,ny)')
    nx,ny=prior.shape[-2:]
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    ph=_normalized_fft(prior);rh=ph[:,0];k2=kx*kx+ky*ky
    inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    ex=_grid(-1j*kx*inv*rh);ey=_grid(-1j*ky*inv*rh)
    output=source.copy();ids={p:i for i,p in enumerate(P5)}
    for ii,(a,b) in enumerate(P4):
        output[:,ii]+=_grid(-1j*kx*(ph[:,ids[a+1,b]]+center[0]*ph[:,ii])
                            -1j*ky*(ph[:,ids[a,b+1]]+center[1]*ph[:,ii]))
        output[:,ii]-=nu*(a+b)*prior[:,ii]
        if a:output[:,ii]+=a*(ex-nu*center[0])*prior[:,ids[a-1,b]]
        if b:output[:,ii]+=b*(ey-nu*center[1])*prior[:,ids[a,b-1]]
        if a>1:output[:,ii]+=nu*theta*a*(a-1)*prior[:,ids[a-2,b]]
        if b>1:output[:,ii]+=nu*theta*b*(b-1)*prior[:,ids[a,b-2]]
    return output


def _central(raw,degree):
    pairs=powers(degree);ids={p:i for i,p in enumerate(pairs)}
    u,v=raw[1]/raw[0],raw[2]/raw[0];result=np.zeros_like(raw[:len(pairs)])
    for row,(a,b) in enumerate(pairs):
        for i in range(a+1):
            for j in range(b+1):
                result[row]+=math.comb(a,i)*math.comb(b,j)*(-u)**(a-i)*(-v)**(b-j)*raw[ids[i,j]]
    return result


def _fifth_pair_products(covariance,third):
    """Sum the ten covariance×central-third partitions of a fifth moment."""
    ids={p:i for i,p in enumerate(powers(3)[6:])};result=[]
    xx,xy,yy=covariance
    for a,b in P5[15:]:
        value=np.zeros_like(xx)
        if a>=2:value+=math.comb(a,2)*xx*third[ids[a-2,b]]
        if a and b:value+=a*b*xy*third[ids[a-1,b-1]]
        if b>=2:value+=math.comb(b,2)*yy*third[ids[a,b-2]]
        result.append(value)
    return np.asarray(result)


def brownian_prior5(initial,times,periods,nu,theta,thermal_covariance=None):
    """Extend the existing degree4 Brownian prior using decaying fifth cumulants."""
    initial=np.asarray(initial,float);times=np.asarray(times,float)
    if initial.ndim!=3 or initial.shape[0]!=21:raise ValueError('initial must contain21 degree5 raw moments')
    initial=_grid(_normalized_fft(initial))
    if np.min(initial[0])<=0:raise ValueError('positive initial density required')
    lower=brownian_prior(initial[:15],times,periods,nu,theta,thermal_covariance)
    central=_central(initial,5);cov=central[3:6]/central[0]
    fifth_cumulant=central[15:]-_fifth_pair_products(cov,central[6:10])
    nx,ny=initial.shape[-2:];kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                                             2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    elapsed=times-times[0];z=float(nu)*elapsed
    if nu==0:noise=np.zeros_like(elapsed)
    else:
        polynomial=z**3*(2/3+z*(-.5+z*(7/30+z*(-1/12+z*(31/1260)))))
        noise=theta/nu**2*np.where(z<1e-3,polynomial,2*z+4*np.expm1(-z)-np.expm1(-2*z))
    filt=np.exp(-.5*noise[:,None,None]*(kx*kx+ky*ky)-5*z[:,None,None])
    cumulant=_grid(_normalized_fft(fifth_cumulant)[None]*filt[:,None])
    result=np.empty((len(times),21,nx,ny));result[:,:15]=lower
    ids={p:i for i,p in enumerate(P5)}
    for it in range(len(times)):
        current=_central(lower[it],4);cov=current[3:6]/current[0]
        c5=cumulant[it]+_fifth_pair_products(cov,current[6:10])
        ct=np.concatenate((current,c5));u,v=lower[it,1]/lower[it,0],lower[it,2]/lower[it,0]
        for row,(a,b) in enumerate(P5[15:],start=15):
            result[it,row]=0
            for i in range(a+1):
                for j in range(b+1):
                    result[it,row]+=math.comb(a,i)*math.comb(b,j)*u**(a-i)*v**(b-j)*ct[ids[i,j]]
    result[0]=initial
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite degree5 prior')
    return result


def evolve(times,source,prior,periods,center,nu,theta,*,relaxation=1.,
           closure_relaxation=1.,taper_fraction=.6,taper_power=8,
           field_feedback=True,conserve_mass=True,basis_temperature=None,deadline_check=None):
    """D4 stable response; source<=4 and prior<=5, degree-major ordering."""
    check=deadline_check or (lambda:None);check()
    times=np.asarray(times,float);source=np.asarray(source,float);prior=np.asarray(prior,float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('ordered times required')
    if prior.ndim!=4 or prior.shape[:2]!=(len(times),21) or source.shape!=prior[:,:15].shape:
        raise ValueError('invalid D4 source/prior shapes')
    if not np.isfinite(source).all() or not np.isfinite(prior).all():raise ValueError('finite moments required')
    nu=float(nu);theta=float(theta);center=np.asarray(center,float)
    if theta<=0 or nu<0 or relaxation<0 or closure_relaxation<0:raise ValueError('invalid physical/regularization parameters')
    basis=theta if basis_temperature is None else float(basis_temperature)
    if not np.isfinite(basis) or basis<=0:raise ValueError('positive finite basis temperature required')
    nx,ny=prior.shape[-2:];scale=math.sqrt(basis);n=15;ids={p:i for i,p in enumerate(P4)}
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    k2=kx*kx+ky*ky
    prior_h=_normalized_fft(raw_to_hermite(prior,scale,5))[:,:15]
    exact_h=_normalized_fft(raw_to_hermite(moment_rhs(prior,source,periods,center,nu,theta),scale,4))
    mean_h=prior_h[0,:,0,0].real;matrix=np.zeros((nx,ny,n,n),complex)
    for row,(a,b) in enumerate(P4):
        matrix[...,row,row]=-1j*(kx*center[0]+ky*center[1])-nu*(a+b)
        for axis,(power,k) in enumerate(((a,kx),(b,ky))):
            lower=(a-1,b) if axis==0 else (a,b-1);upper=(a+1,b) if axis==0 else (a,b+1)
            if power:
                col=ids[lower]
                matrix[...,row,col]+=-1j*k*scale*math.sqrt(power)-nu*center[axis]/scale*math.sqrt(power)
                if field_feedback:
                    factor=np.divide(-1j*k,k2,out=np.zeros_like(k,dtype=complex),where=k2>0)
                    matrix[...,row,0]+=math.sqrt(power)/scale*mean_h[col]*factor
            if upper in ids:matrix[...,row,ids[upper]]+=-1j*k*scale*math.sqrt(power+1)
            if power>1:
                lower2=(a-2,b) if axis==0 else (a,b-2)
                matrix[...,row,ids[lower2]]+=nu*(theta/basis-1)*math.sqrt(power*(power-1))
    rates=np.full((nx,ny,n),float(relaxation))
    rates[...,10:]+=float(closure_relaxation)*scale*np.sqrt(k2)[...,None]
    for i in range(n):matrix[...,i,i]-=rates[...,i]
    hprior=np.moveaxis(prior_h,1,-1)
    forcing=np.moveaxis(exact_h,1,-1)-np.einsum('xyab,txyb->txya',matrix,hprior)
    if conserve_mass:forcing[:,0,0,0]=rates[0,0,0]*hprior[0,0,0,0]
    steps={};result=np.empty_like(hprior);result[0]=hprior[0]
    for it,h in enumerate(np.diff(times)):
        check();key=round(float(h),13)
        if key not in steps:steps[key]=_matrix_functions_batched(matrix,h)
        e,k0,k1=steps[key]
        result[it+1]=(np.einsum('xyab,xyb->xya',e,result[it])
                      +np.einsum('xyab,xyb->xya',k0,forcing[it])
                      +np.einsum('xyab,xyb->xya',k1,forcing[it+1]-forcing[it]))
    result=np.moveaxis(result,-1,1);increments=result-prior_h
    if taper_fraction is not None:
        normalized=np.maximum(np.abs(kx)/(np.pi*nx/float(periods[0])),np.abs(ky)/(np.pi*ny/float(periods[1])))
        increments*=np.exp(-(normalized/float(taper_fraction))**int(taper_power))
    result=_project(prior_h+increments)
    raw=np.einsum('ab,tbxy->taxy',np.linalg.inv(hermite_transform(4,scale)),result)
    if not np.isfinite(raw).all():raise FloatingPointError('nonfinite H4 response')
    check();return {'times':times.copy(),'raw_modes':raw,'correction_modes':raw-_normalized_fft(prior[:,:15]),
                    'max_increment':float(np.max(np.abs(increments))),
                    'parameters':{'relaxation':relaxation,'closure_relaxation':closure_relaxation,
                                  'taper_fraction':taper_fraction,'field_feedback':field_feedback,'basis_temperature':basis}}
