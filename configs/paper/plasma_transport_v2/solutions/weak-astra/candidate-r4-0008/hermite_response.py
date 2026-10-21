"""Causal, regularized Hermite response to public source moment snapshots.

No Oracle access, candidate SDK calls, or hidden/query coordinates. Arrays use
normalized Fourier coefficients internally; raw input axes are (time,moment,x,y).
The stable streaming closure sets Hermite deviations of total degree four to
zero, around an independently supplied degree-four prior. Relaxation penalizes
deviation from that prior rather than deleting the exact initial condition.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm


def powers(degree):
    return tuple((d-b,b) for d in range(degree+1) for b in range(d+1))


P3=powers(3)
P4=powers(4)


def hermite_transform(degree,scale):
    """H_ab = integral He_a(w_x/s)He_b(w_y/s)/sqrt(a!b!) f dw."""
    pairs=powers(degree);indices={p:i for i,p in enumerate(pairs)}
    transform=np.zeros((len(pairs),len(pairs)))
    for row,(a,b) in enumerate(pairs):
        for i in range(a//2+1):
            for j in range(b//2+1):
                x,y=a-2*i,b-2*j
                coef=(math.factorial(a)*math.factorial(b)*(-1.)**(i+j)
                      /(2.**(i+j)*math.factorial(i)*math.factorial(j)
                        *math.factorial(x)*math.factorial(y)
                        *math.sqrt(math.factorial(a)*math.factorial(b))))
                transform[row,indices[x,y]]=coef/scale**(x+y)
    return transform


def raw_to_hermite(raw,scale,degree):
    return np.einsum('ab,tbxy->taxy',hermite_transform(degree,scale),raw)


def hermite_to_raw(hermite,scale,degree):
    return np.einsum('ab,tbxy->taxy',np.linalg.inv(hermite_transform(degree,scale)),hermite)


def _project(a):
    if a.shape[-2]%2==0:a[...,a.shape[-2]//2,:]=0
    if a.shape[-1]%2==0:a[...,:,a.shape[-1]//2]=0
    return a


def _normalized_fft(a):
    return _project(np.fft.fft2(a,axes=(-2,-1))/np.prod(a.shape[-2:]))


def _grid(a):
    return np.fft.ifft2(a,axes=(-2,-1)).real*np.prod(a.shape[-2:])


def brownian_prior(initial,times,periods,nu,theta,thermal_covariance=None):
    """Extend the incumbent Brownian macro prior to all degree-four moments.

    Degree<=2 raw moments receive the incumbent OU displacement spatial filter.
    Degree3 *central* moments receive Brownian-only displacement and 3nu decay.
    Degree4 cumulants receive Brownian-only displacement and 4nu decay, while
    the Gaussian fourth moment is rebuilt from current central pressure.
    This is an explicit closure hypothesis, not exact VPFP evolution.
    """
    initial=np.asarray(initial,float);times=np.asarray(times,float)
    if initial.ndim!=3 or initial.shape[0]!=15:raise ValueError('initial must be (15,nx,ny)')
    if np.min(initial[0])<=0:raise ValueError('positive initial density required')
    nu=float(nu);theta=float(theta);nx,ny=initial.shape[-2:]
    initial=_grid(_normalized_fft(initial));rho=initial[0]
    if np.min(rho)<=0:raise ValueError('positive projected initial density required')
    ux,uy=initial[1]/rho,initial[2]/rho
    ids={p:i for i,p in enumerate(P4)}
    central=np.zeros_like(initial)
    for row,(a,b) in enumerate(P4):
        for i in range(a+1):
            for j in range(b+1):
                central[row]+=(math.comb(a,i)*math.comb(b,j)*(-ux)**(a-i)*(-uy)**(b-j)
                               *initial[ids[i,j]])
    def gaussian4(c2,density):
        xx,xy,yy=c2
        return np.asarray([3*xx*xx,3*xx*xy,xx*yy+2*xy*xy,3*xy*yy,3*yy*yy])/density
    fourth_cumulant=central[10:]-gaussian4(central[3:6],rho)
    if thermal_covariance is None:
        c=central[3:6].mean(axis=(-2,-1))/rho.mean()
        thermal_covariance=np.array([[c[0],c[1]],[c[1],c[2]]])
    thermal_covariance=np.asarray(thermal_covariance,float)
    if thermal_covariance.shape!=(2,2) or np.min(np.linalg.eigvalsh(thermal_covariance))<0:
        raise ValueError('positive semidefinite thermal covariance required')
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    elapsed=times-times[0]
    if np.min(elapsed)<0:raise ValueError('ordered time range required')
    z=nu*elapsed
    if nu==0:
        drift=elapsed;noise=np.zeros_like(elapsed)
    else:
        drift=-np.expm1(-z)/nu
        polynomial=z**3*(2/3+z*(-.5+z*(7/30+z*(-1/12+z*(31/1260)))))
        noise=theta/nu**2*np.where(z<1e-3,polynomial,2*z+4*np.expm1(-z)-np.expm1(-2*z))
    thermal=(thermal_covariance[0,0]*kx*kx+2*thermal_covariance[0,1]*kx*ky
             +thermal_covariance[1,1]*ky*ky)
    brownian=np.exp(-.5*noise[:,None,None]*(kx*kx+ky*ky))
    low_filter=brownian*np.exp(-.5*drift[:,None,None]**2*thermal)
    raw=np.empty((len(times),15,nx,ny))
    raw[:,:6]=_grid(_normalized_fft(initial[:6])[None]*low_filter[:,None])
    c3=_grid(_normalized_fft(central[6:10])[None]*brownian[:,None])*np.exp(-3*z)[:,None,None,None]
    cumulant4=_grid(_normalized_fft(fourth_cumulant)[None]*brownian[:,None])*np.exp(-4*z)[:,None,None,None]
    for it in range(len(times)):
        r=raw[it,0];u,v=raw[it,1]/r,raw[it,2]/r
        c2=np.asarray([raw[it,3]-r*u*u,raw[it,4]-r*u*v,raw[it,5]-r*v*v])
        ct=np.concatenate((np.asarray([r,np.zeros_like(r),np.zeros_like(r)]),c2,c3[it],
                           gaussian4(c2,r)+cumulant4[it]))
        for row,(a,b) in enumerate(P4[6:],start=6):
            raw[it,row]=0
            for i in range(a+1):
                for j in range(b+1):
                    raw[it,row]+=math.comb(a,i)*math.comb(b,j)*u**(a-i)*v**(b-j)*ct[ids[i,j]]
    raw[0]=initial  # preserve initial higher moments, including product alias.
    if not np.isfinite(raw).all():raise FloatingPointError('nonfinite Brownian prior')
    return raw


def moment_rhs(prior,source,periods,center,nu,theta):
    """Exact degree<=3 hierarchy on a supplied degree4 raw-moment grid."""
    prior=np.asarray(prior,float);source=np.asarray(source,float)
    if prior.ndim!=4 or prior.shape[1]!=15 or source.shape!=prior[:,:10].shape:
        raise ValueError('need prior (nt,15,nx,ny) and source (nt,10,nx,ny)')
    nx,ny=prior.shape[-2:]
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    ph=_normalized_fft(prior);rh=ph[:,0];k2=kx*kx+ky*ky
    inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    ex=_grid(-1j*kx*inv*rh);ey=_grid(-1j*ky*inv*rh)
    output=source.copy();ids={p:i for i,p in enumerate(P4)}
    for ii,(a,b) in enumerate(P3):
        output[:,ii]+=_grid(-1j*kx*(ph[:,ids[a+1,b]]+center[0]*ph[:,ii])
                            -1j*ky*(ph[:,ids[a,b+1]]+center[1]*ph[:,ii]))
        output[:,ii]-=nu*(a+b)*prior[:,ii]
        if a:output[:,ii]+=a*(ex-nu*center[0])*prior[:,ids[a-1,b]]
        if b:output[:,ii]+=b*(ey-nu*center[1])*prior[:,ids[a,b-1]]
        if a>1:output[:,ii]+=nu*theta*a*(a-1)*prior[:,ids[a-2,b]]
        if b>1:output[:,ii]+=nu*theta*b*(b-1)*prior[:,ids[a,b-2]]
    return output


def _matrix_functions(matrix,h):
    """Return exp(h A), h phi1(h A), h phi2(h A) without inversion."""
    n=matrix.shape[-1]
    # One block exponential remains accurate when A has zero eigenvalues.
    extended=np.zeros((3*n,3*n),complex)
    extended[:n,:n]=matrix*h
    extended[:n,n:2*n]=np.eye(n)
    extended[n:2*n,2*n:]=np.eye(n)
    result=expm(extended)
    return result[:n,:n],h*result[:n,n:2*n],h*result[:n,2*n:]


def _matrix_functions_batched(matrix,h):
    """Batched entire functions by Taylor scaling and analytic doubling."""
    n=matrix.shape[-1];z=np.asarray(matrix,complex)*float(h)
    largest=float(np.max(np.sum(np.abs(z),axis=-1)))
    squarings=max(0,int(math.ceil(math.log2(max(largest/.5,1.)))))
    z=z/2.**squarings
    eye=np.broadcast_to(np.eye(n),z.shape).astype(complex)
    e=eye.copy();p1=eye.copy();p2=eye*.5;power=eye.copy()
    for order in range(1,25):
        power=np.matmul(power,z)/order
        e+=power;p1+=power/(order+1);p2+=power/((order+1)*(order+2))
    for _ in range(squarings):
        p2=.25*(np.matmul(eye+e,p2)+p1)
        p1=.5*np.matmul(eye+e,p1)
        e=np.matmul(e,e)
    return e,float(h)*p1,float(h)*p2


def evolve(times,source,prior,periods,center,nu,theta,*,relaxation=1.,
           closure_relaxation=1.,taper_fraction=.6,taper_power=8,
           field_feedback=True,conserve_mass=True,basis_temperature=None,deadline_check=None):
    """Evolve source-driven corrections around a degree-four raw prior.

    `source` contains raw moments in fixed w=v-center through degree3 (P3);
    `prior` contains the same moments through degree4 (P4). Both are physical
    grids at strictly ordered `times`, and prior[0] fixes the initial condition.
    The source is piecewise linear between its prepared time nodes. The returned
    raw modes have shape (nt,10,nx,ny). Relaxation units are inverse time;
    closure_relaxation multiplies |k| sqrt(theta) on only degree3 components.
    `field_feedback` linearizes Poisson force around the homogeneous prior.
    """
    check=deadline_check or (lambda:None);check()
    times=np.asarray(times,float);source=np.asarray(source,float);prior=np.asarray(prior,float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('ordered times required')
    if prior.ndim!=4 or prior.shape[:2]!=(len(times),15) or source.shape!=prior[:,:10].shape:
        raise ValueError('invalid time/source/prior shapes')
    if not np.isfinite(source).all() or not np.isfinite(prior).all():raise ValueError('finite moments required')
    nu=float(nu);theta=float(theta);center=np.asarray(center,float)
    if theta<=0 or nu<0 or relaxation<0 or closure_relaxation<0:raise ValueError('invalid physical/regularization parameters')
    basis=theta if basis_temperature is None else float(basis_temperature)
    if not np.isfinite(basis) or basis<=0:raise ValueError('positive finite basis temperature required')
    nx,ny=prior.shape[-2:];scale=math.sqrt(basis);n=10;ids={p:i for i,p in enumerate(P3)}
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    k2=kx*kx+ky*ky
    prior_h=_normalized_fft(raw_to_hermite(prior,scale,4))[:,:10]
    exact_h=_normalized_fft(raw_to_hermite(moment_rhs(prior,source,periods,center,nu,theta),scale,3))
    # Initial homogeneous coefficients define the self-consistent linear force.
    mean_h=prior_h[0,:,0,0].real
    matrices=np.zeros((nx,ny,n,n),complex)
    base_matrices=np.zeros_like(matrices)
    for row,(a,b) in enumerate(P3):
        base_matrices[...,row,row]=-1j*(kx*center[0]+ky*center[1])-nu*(a+b)
        for axis,(power,k) in enumerate(((a,kx),(b,ky))):
            lower=(a-1,b) if axis==0 else (a,b-1)
            upper=(a+1,b) if axis==0 else (a,b+1)
            if power:
                col=ids[lower]
                base_matrices[...,row,col]+=-1j*k*scale*math.sqrt(power)-nu*center[axis]/scale*math.sqrt(power)
                if field_feedback:
                    factor=np.divide(-1j*k,k2,out=np.zeros_like(k,dtype=complex),where=k2>0)
                    base_matrices[...,row,0]+=math.sqrt(power)/scale*mean_h[col]*factor
            if upper in ids:base_matrices[...,row,ids[upper]]+=-1j*k*scale*math.sqrt(power+1)
            if power>1:
                lower2=(a-2,b) if axis==0 else (a,b-2)
                base_matrices[...,row,ids[lower2]]+=nu*(theta/basis-1)*math.sqrt(power*(power-1))
    matrices[:]=base_matrices
    rates=np.full((nx,ny,n),float(relaxation))
    # At highest retained degree only, unresolved Hermite flux is regularized.
    rates[...,6:]+=float(closure_relaxation)*scale*np.sqrt(k2)[...,None]
    for i in range(n):matrices[...,i,i]-=rates[...,i]
    # The exact prior RHS includes all fourth moments and nonlinear prior force.
    # Subtraction leaves only those terms missing from the constant linear core.
    hprior=np.moveaxis(prior_h,1,-1)
    forcing=np.moveaxis(exact_h,1,-1)-np.einsum('xyab,txyb->txya',matrices,hprior)
    if conserve_mass:
        # Spatially periodic public VPFP has constant total particles. This
        # removes quadrature noise in the source's sole invariant component.
        forcing[:,0,0,0]=rates[0,0,0]*hprior[0,0,0,0]
    steps={};result=np.empty_like(hprior);result[0]=hprior[0]
    for it,h in enumerate(np.diff(times)):
        check();key=round(float(h),13)
        if key not in steps:
            steps[key]=_matrix_functions_batched(matrices,h)
        e,k0,k1=steps[key]
        result[it+1]=(np.einsum('xyab,xyb->xya',e,result[it])
                      +np.einsum('xyab,xyb->xya',k0,forcing[it])
                      +np.einsum('xyab,xyb->xya',k1,forcing[it+1]-forcing[it]))
    result=np.moveaxis(result,-1,1)
    increments=result-prior_h
    if taper_fraction is not None:
        # Tensor-product cutoffs preserve low resolved modes across aspect ratios.
        normalized=np.maximum(np.abs(kx)/(np.pi*nx/float(periods[0])),
                              np.abs(ky)/(np.pi*ny/float(periods[1])))
        increments*=np.exp(-(normalized/float(taper_fraction))**int(taper_power))
    result=_project(prior_h+increments)
    raw=np.einsum('ab,tbxy->taxy',np.linalg.inv(hermite_transform(3,scale)),result)
    if not np.isfinite(raw).all():raise FloatingPointError('nonfinite Hermite response')
    check()
    return {'times':times.copy(),'raw_modes':raw,'correction_modes':raw-_normalized_fft(prior[:,:10]),
            'max_increment':float(np.max(np.abs(increments))),
            'parameters':{'relaxation':relaxation,'closure_relaxation':closure_relaxation,
                          'taper_fraction':taper_fraction,'field_feedback':field_feedback,'basis_temperature':basis}}
