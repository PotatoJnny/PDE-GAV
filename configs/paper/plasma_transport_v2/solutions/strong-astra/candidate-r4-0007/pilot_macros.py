"""Guarded low-moment source corrections from independent thermal pilots.

No Oracle, completed solution, GPU, or candidate entrypoint is used. Inputs
are unnormalized source Fourier coefficients and same-quadrature source
pilots. Validation excludes full frames from both POD and the
coarse temporal interpolant. This is an empirical gate, not an error bound.
"""
# The optimized not-a-knot construction follows scipy.interpolate.CubicSpline
# (SciPy 1.15.2); its numerical equations and operation order are retained.
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.linalg import solve, solve_banded


@lru_cache(maxsize=512)
def _cubic_geometry(time_bytes):
    """Reuse the same not-a-knot system across channels, ranks and folds."""
    t=np.frombuffer(time_bytes,dtype=np.float64);dx=np.diff(t);n=len(t)
    A=np.zeros((3,n))
    A[1,1:-1]=2*(dx[:-1]+dx[1:])
    A[0,2:]=dx[:-1];A[-1,:-2]=dx[1:]
    A[1,0]=dx[1];A[0,1]=t[2]-t[0]
    A[1,-1]=dx[-2];A[-1,-2]=t[-1]-t[-3]
    return dx,A,t[2]-t[0],t[-1]-t[-3]


def _fast_cubic(times, values):
    """Same CubicSpline derivative system with cached geometry and less setup.

    The linear, quadratic and ordinary not-a-knot conventions match SciPy's
    public CubicSpline implementation. Floating-point operation order and
    LAPACK solves are retained; only invariant geometry/setup is shared.
    """
    t=np.array(times,dtype=np.float64,copy=True,order='C');v=np.asarray(values)
    v=v.astype(np.complex128 if np.iscomplexobj(v) else np.float64,copy=False)
    n=len(t)
    if n<2 or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t)<=0):
        return CubicSpline(t,v,axis=0)
    if n>3:dx,A,start_width,end_width=_cubic_geometry(t.tobytes())
    else:dx=np.diff(t)
    dxr=dx.reshape((n-1,)+(1,)*(v.ndim-1))
    slope=np.diff(v,axis=0)/dxr
    c=np.empty((4,n-1)+v.shape[1:],dtype=v.dtype)
    if n==2:
        c[:2]=0.;c[2]=slope;c[3]=v[:-1]
    elif n==3:
        A=np.zeros((3,3));b=np.empty_like(v)
        A[0,0]=1.;A[0,1]=1.
        A[1,0]=dx[1];A[1,1]=2*(dx[0]+dx[1]);A[1,2]=dx[0]
        A[2,1]=1.;A[2,2]=1.
        b[0]=2*slope[0]
        b[1]=3*(dxr[0]*slope[1]+dxr[1]*slope[0])
        b[2]=2*slope[1]
        s=solve(A,b,overwrite_a=True,overwrite_b=True,check_finite=False)
    else:
        b=np.empty_like(v)
        b[1:-1]=3*(dxr[1:]*slope[:-1]+dxr[:-1]*slope[1:])
        b[0]=((dxr[0]+2*start_width)*dxr[1]*slope[0]+dxr[0]**2*slope[1])/start_width
        b[-1]=((dxr[-1]**2*slope[-2]+(2*end_width+dxr[-1])*dxr[-2]*slope[-1])/end_width)
        s=solve_banded((1,1),A.copy(),b,overwrite_ab=True,overwrite_b=True,check_finite=False)
    if n>2:
        term=(s[:-1]+s[1:]-2*slope)/dxr
        c[0]=term/dxr
        c[1]=(slope-s[:-1])/dxr-term
        c[2]=s[:-1];c[3]=v[:-1]
    return PPoly.construct_fast(c,t,extrapolate=True,axis=0)


class PiecewiseCubic:
    def __init__(self,times,values,corners=()):
        self.times=np.asarray(times,float);self.values=np.asarray(values)
        joins=np.flatnonzero(np.isin(self.times,corners))
        indices=np.unique(np.r_[0,joins,len(self.times)-1])
        self.ends=self.times[indices[1:]]
        self.parts=[_fast_cubic(self.times[a:b+1],self.values[a:b+1])
                    for a,b in zip(indices[:-1],indices[1:])]
    def __call__(self,times):
        t=np.atleast_1d(np.asarray(times,float))
        slots=np.minimum(np.searchsorted(self.ends,t,side='right'),len(self.parts)-1)
        out=np.empty((len(t),)+self.values.shape[1:],dtype=np.result_type(self.values,float))
        for i,part in enumerate(self.parts):
            mask=slots==i
            if np.any(mask):out[mask]=part(t[mask])
        return out


def _phase(xy,n,periods):
    k=np.fft.fftfreq(n)*n
    return np.exp(2j*np.pi*(xy[:,0,None,None]/periods[0]*k[:,None]
                            +xy[:,1,None,None]/periods[1]*k[None,:]))/n**2


def _project(modes,phase):
    size=modes.shape[-2]*modes.shape[-1]
    return (modes.reshape(modes.shape[:-2]+(size,))@phase.reshape(len(phase),size).T).real


def _pod(modes,zero_mean=True,times=None,corners=(),periods=(2*np.pi,2*np.pi),*,density_weight=0.):
    """POD of independent training-frame interpolation defects.

    The outer validation frames are absent from times/modes. Protect exact
    corners and endpoints during two disjoint inner parity holdouts. Keeping
    the defect's temporal mean includes coherent interpolation bias modes.
    """
    if times is None:raise ValueError('residual POD requires training times')
    times=np.asarray(times,float)
    eligible=np.flatnonzero(~np.isin(times,np.r_[times[0],times[-1],corners]))
    defects=[]
    for offset in (0,1):
        held=eligible[offset::2]
        if not len(held):continue
        retained=np.setdiff1d(np.arange(len(times)),held)
        coarse=PiecewiseCubic(times[retained],modes[retained],corners)
        defects.append(modes[held]-coarse(times[held]))
    if not defects:
        return np.empty((0,modes.shape[-1],modes.shape[-1]),complex)
    modes=np.concatenate(defects,axis=0)
    n=modes.shape[-1]
    whitening=None
    if zero_mean and density_weight:
        # Density alone uses the frozen density/field floor metric. Applying
        # sqrt(1 + floor_n^2/(floor_E^2 |k|^2)) before POD prioritizes residual
        # directions that contribute to electric-field error as well as mass.
        k=np.fft.fftfreq(n)*n
        kx,ky=np.meshgrid(2*np.pi*k/periods[0],2*np.pi*k/periods[1],indexing='ij')
        k2=kx*kx+ky*ky
        inverse_k2=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
        whitening=np.sqrt(1.+density_weight*(1.0041253663782175/.0698185715388132)**2*inverse_k2)
    physical=np.fft.ifft2(modes if whitening is None else modes*whitening,
                         axes=(-2,-1)).real.reshape(len(modes),-1)
    if zero_mean:physical-=physical.mean(axis=1,keepdims=True)
    _,singular,spatial=np.linalg.svd(physical,full_matrices=False)
    active=singular>max(1.e-13,1.e-10*singular[0]) if len(singular) else []
    spatial=spatial[active][:10]
    if whitening is not None and len(spatial):
        spatial=np.fft.ifft2(np.fft.fft2(spatial.reshape(-1,n,n),axes=(-2,-1))
                             /whitening,axes=(-2,-1)).real.reshape(len(spatial),-1)
        # QR preserves every retained prefix span while restoring the existing
        # physical L2 basis normalization and all probe noise-gain gate units.
        spatial=np.linalg.qr(spatial.T,mode='reduced')[0].T
    spatial=spatial*n
    basis=np.fft.fft2(spatial.reshape(-1,n,n),axes=(-2,-1))
    if zero_mean:basis[:,0,0]=0.
    return basis


def _fit_operator(basis_at_probes,nuisance,sites,condition_limit=50.,gain_limit=2.):
    """Eliminate nuisance before checking identifiability of retained modes."""
    B=basis_at_probes[sites];N=nuisance[sites]
    un,sn,_=np.linalg.svd(N,full_matrices=False)
    q=un[:,sn>max(1.e-13,1.e-10*sn[0])] if len(sn) else un
    projector=np.eye(len(sites))-q@q.T
    centered=projector@B
    singular=np.linalg.svd(centered,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf
    gain=float(1/singular[-1]) if singular[-1]>0 else np.inf
    if condition>=condition_limit or gain>gain_limit:
        raise ValueError(f'ill-conditioned centered probe map: condition={condition}, gain={gain}')
    inverse=np.linalg.pinv(centered,rcond=1.e-12)@projector
    nuisance_inverse=np.linalg.pinv(N,rcond=1.e-12)
    def fit(residual):
        data=residual[:,sites]
        coeff=data@inverse.T
        nuisance_coeff=(data-coeff@B.T)@nuisance_inverse.T
        return coeff,nuisance_coeff
    return fit,{'condition':condition,'noise_gain':gain}


class DensityCorrection:
    def __init__(self,times,modes,corners,*,basis=None,dense=None,anchor=None,diagnostics=None,coarse=None,zero_mean=True):
        self.times=np.asarray(times);self.modes=np.asarray(modes)
        self.coarse=PiecewiseCubic(times,modes,corners) if coarse is None else coarse
        self.basis=basis;self.dense=dense;self.anchor=anchor
        self.enabled=basis is not None;self.diagnostics=diagnostics or {}
        self.zero_mean=zero_mean
    def evaluate(self,times):
        scalar=np.ndim(times)==0;t=np.atleast_1d(np.asarray(times,float))
        if np.any(t<self.times[0]) or np.any(t>self.times[-1]):
            raise ValueError('time outside prepared source interval')
        answer=self.coarse(t)
        if self.enabled:
            correction=self.dense(t)-self.anchor(t)
            answer+=np.einsum('tr,rij->tij',correction,self.basis)
        # Preserve every acquired full frame exactly, including duplicates.
        index=np.searchsorted(self.times,t);hit=index<len(self.times)
        hit[hit]&=self.times[index[hit]]==t[hit]
        answer[hit]=self.modes[index[hit]]
        if self.zero_mean:answer[:,0,0]=0.
        return answer[0] if scalar else answer


def _residual_cache(times,modes,pilot_times,phase,pilot_values,corners,coarse=None):
    coarse=PiecewiseCubic(times,modes,corners) if coarse is None else coarse
    projected=PiecewiseCubic(times,_project(modes,phase),corners)
    residual=pilot_values-projected(pilot_times)
    raw=PiecewiseCubic(pilot_times,residual,corners)
    raw_anchor=PiecewiseCubic(times,raw(times),corners)
    residual-=raw_anchor(pilot_times)
    return coarse,residual


def _construct(times,modes,pilot_times,phase,pilot_values,corners,rank,nuisance,
               basis_all=None,coarse=None,residual=None,zero_mean=True,*,
               periods=(2*np.pi,2*np.pi),density_weight=0.):
    basis_all=(_pod(modes,zero_mean,times,corners,periods,density_weight=density_weight)
               if basis_all is None else basis_all)
    if len(basis_all)<rank:raise ValueError('insufficient independent POD modes')
    basis=basis_all[:rank]
    # Up to two omitted POD directions absorb unresolved pilot contamination;
    # they are nuisance terms and never enter the spatial correction.
    extra=_project(basis_all[rank:rank+2],phase).T
    constant=np.ones((len(phase),1)) if zero_mean else np.empty((len(phase),0))
    nuisance_matrix=np.column_stack((constant,nuisance,extra))
    B=_project(basis,phase).T
    all_sites=np.arange(len(phase));fit,metrics=_fit_operator(B,nuisance_matrix,all_sites)
    if residual is None:
        coarse,residual=_residual_cache(times,modes,pilot_times,phase,pilot_values,corners,coarse)
    coefficients,nuisance_coeff=fit(residual)
    spatial_folds=[]
    for offset in (0,2):
        withheld_sites=all_sites[offset::4];fitting_sites=np.setdiff1d(all_sites,withheld_sites)
        reduced,leave_metrics=_fit_operator(B,nuisance_matrix,fitting_sites)
        reduced_coeff,reduced_nuisance=reduced(residual)
        predicted=reduced_coeff@B.T+reduced_nuisance@nuisance_matrix.T
        base=float(np.sqrt(np.mean(residual[:,withheld_sites]**2)))
        defect=float(np.sqrt(np.mean((predicted[:,withheld_sites]-residual[:,withheld_sites])**2)))
        amplitude=float(np.sqrt(np.mean(coefficients**2)))
        instability=float(np.sqrt(np.mean((coefficients-reduced_coeff)**2)))
        spatial_folds.append(dict(offset=offset,heldout_probe_ratio=defect/max(base,1.e-14),
            coefficient_instability=instability/max(amplitude,1.e-14),
            heldout_condition=leave_metrics['condition'],heldout_noise_gain=leave_metrics['noise_gain']))
        if defect>max(1.e-10,.5*base) or instability>max(1.e-10,.25*amplitude):
            raise ValueError('withheld-probe reconstruction or coefficient stability failed')
    metrics.update({key:max(row[key] for row in spatial_folds) for key in (
        'heldout_probe_ratio','coefficient_instability','heldout_condition','heldout_noise_gain')})
    metrics['spatial_folds']=spatial_folds
    dense=PiecewiseCubic(pilot_times,coefficients,corners)
    anchor=PiecewiseCubic(times,dense(times),corners)
    return DensityCorrection(times,modes,corners,basis=basis,dense=dense,anchor=anchor,coarse=coarse,zero_mean=zero_mean),metrics


def _norms(error,periods,zero_mean=True):
    n=error.shape[-1];k=np.fft.fftfreq(n)*n
    kx,ky=np.meshgrid(2*np.pi*k/periods[0],2*np.pi*k/periods[1],indexing='ij')
    k2=kx*kx+ky*ky
    weight=(np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0) if zero_mean else 1./(1.+k2))
    energy=abs(error/n**2)**2
    return np.sqrt(np.sum(energy,axis=(-2,-1))),np.sqrt(np.sum(energy*weight,axis=(-2,-1)))


def prepare_scalar_correction(full_times,full_modes,pilot_times,pilot_xy,pilot_values,
        *,periods=(2*np.pi,2*np.pi),corner_times=(),ranks=(1,2,3,4,5,6,7,8),nuisance=None,
        should_continue=None,first_passing=False,zero_mean=True,density_weight=0.):
    """Return a gated source interpolant; rejection returns the coarse spline.

    Full-frame heldout folds use two disjoint quarter-index sets. Neither
    heldout full frames nor their POD contribution enters their predictor.
    Dense pilots remain available, as intended for eventual preparation use.
    """
    t=np.asarray(full_times,float);f=np.asarray(full_modes,complex)
    pt=np.asarray(pilot_times,float);xy=np.asarray(pilot_xy,float);pv=np.asarray(pilot_values,float)
    periods=np.asarray(periods,float);corners=np.asarray(corner_times,float)
    if not np.isfinite(density_weight) or density_weight<0:
        raise ValueError('density POD weight must be finite and nonnegative')
    if (t.ndim!=1 or len(t)<12 or np.any(np.diff(t)<=0) or f.ndim!=3
            or f.shape[0]!=len(t) or f.shape[1]!=f.shape[2] or f.shape[1]%2!=1
            or pt.ndim!=1 or len(pt)<2 or np.any(np.diff(pt)<=0) or pt[0]!=t[0] or pt[-1]!=t[-1]
            or xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<12 or pv.shape!=(len(pt),len(xy))
            or periods.shape!=(2,) or np.any(periods<=0)
            or not all(np.isfinite(a).all() for a in (t,f,pt,xy,pv,periods,corners))):
        raise ValueError('invalid source correction data')
    if zero_mean and np.max(abs(f[:,0,0]))>1.e-12:
        raise ValueError('full source snapshots must already have zero density mean')
    physical=np.fft.ifft2(f,axes=(-2,-1))
    if np.max(abs(physical.imag))>1.e-10*(1.+np.max(abs(physical.real))):
        raise ValueError('density snapshots must represent real spatial fields')
    if not np.all(np.isin(corners,t)) or not np.all(np.isin(corners,pt)):
        raise ValueError('source corners must be sampled in both time arrays')
    if not np.all(np.isin(t,pt)):
        raise ValueError('all full frames must be sampled in the dense pilot')
    n=f.shape[-1];phase=_phase(xy,n,periods)
    nuisance=np.empty((len(xy),0)) if nuisance is None else np.asarray(nuisance,float)
    if nuisance.ndim!=2 or nuisance.shape[0]!=len(xy) or not np.isfinite(nuisance).all():
        raise ValueError('nuisance must be a finite probe-by-direction matrix')
    diagnostics={'enabled':False,'empirical_gate':True,'ranks':[]}
    fallback=DensityCorrection(t,f,corners,diagnostics=diagnostics,zero_mean=zero_mean)
    passing=[]
    def best_available():
        if not passing:return fallback
        _,rank,result=min(passing,key=lambda row:(row[0],row[1]))
        diagnostics.update(enabled=True,selected_rank=rank)
        result.diagnostics=diagnostics
        return result
    def continuing():
        if should_continue is None or should_continue():return True
        diagnostics['reason']='cooperative density-correction deadline'
        return False
    if not continuing():return best_available()
    eligible=np.flatnonzero(~np.isin(t,np.r_[t[0],t[-1],corners]))
    folds=[eligible[offset::4] for offset in (0,2)]
    if any(len(held)<3 for held in folds):
        diagnostics['reason']='fewer than three independent heldout full frames per fold'
        return fallback
    train_folds=[np.setdiff1d(np.arange(len(t)),held) for held in folds]
    caches=[]
    for train in train_folds+[np.arange(len(t))]:
        if not continuing():return best_available()
        basis=_pod(f[train],zero_mean,t[train],corners,periods,density_weight=density_weight)
        if not continuing():return best_available()
        coarse=(fallback.coarse if len(train)==len(t) else None)
        coarse,residual=_residual_cache(t[train],f[train],pt,phase,pv,corners,coarse)
        caches.append((basis,coarse,residual))
        if not continuing():return best_available()
    for rank in ranks:
        rank=int(rank)
        if rank<1 or rank>8:raise ValueError('retained POD rank must be between one and eight')
        entry={'rank':rank,'passed':False,'folds':[]}
        try:
            for held,train,(basis,coarse,residual) in zip(folds,train_folds,caches):
                if not continuing():return best_available()
                model,metrics=_construct(t[train],f[train],pt,phase,pv,corners,rank,nuisance,basis,coarse,residual,zero_mean)
                coarse_error=model.coarse(t[held])-f[held]
                corrected_error=model.evaluate(t[held])-f[held]
                base_n,base_e=_norms(coarse_error,periods,zero_mean)
                new_n,new_e=_norms(corrected_error,periods,zero_mean)
                rms=lambda x:float(np.sqrt(np.mean(x*x)))
                ratios=[rms(new_n)/max(rms(base_n),1.e-15),rms(new_e)/max(rms(base_e),1.e-15)]
                raw=np.zeros_like(f);fixed=np.zeros_like(f)
                raw[held]=coarse_error;fixed[held]=corrected_error
                widths=np.diff(t)[:,None,None]
                integral_raw=np.cumsum(widths*(raw[1:]+raw[:-1])/2,axis=0)
                integral_fixed=np.cumsum(widths*(fixed[1:]+fixed[:-1])/2,axis=0)
                ib_n,ib_e=_norms(integral_raw,periods,zero_mean);in_n,in_e=_norms(integral_fixed,periods,zero_mean)
                integral_ratios=[float(in_n.max()/max(ib_n.max(),1.e-15)),float(in_e.max()/max(ib_e.max(),1.e-15))]
                passed=(rms(base_n)>1.e-10 and rms(base_e)>1.e-11
                    and max(ratios)<=.5 and max(integral_ratios)<=.6
                    and new_n.max()<=1.25*base_n.max() and new_e.max()<=1.25*base_e.max())
                entry['folds'].append(dict(metrics,density_ratio=ratios[0],poisson_ratio=ratios[1],
                    cumulative_density_ratio=integral_ratios[0],cumulative_poisson_ratio=integral_ratios[1],passed=bool(passed)))
                if not passed:raise ValueError('withheld full-frame accuracy gate failed')
            if not continuing():return best_available()
            final,final_metrics=_construct(t,f,pt,phase,pv,corners,rank,nuisance,*caches[-1],zero_mean=zero_mean)
            entry.update(passed=True,final_probe_metrics=final_metrics)
            score=max(max(row['density_ratio'],row['poisson_ratio']) for row in entry['folds'])
            passing.append((score,rank,final))
            if first_passing:
                diagnostics['ranks'].append(entry)
                diagnostics.update(enabled=True,selected_rank=rank)
                final.diagnostics=diagnostics
                return final
        except (ValueError,np.linalg.LinAlgError) as error:
            entry['reason']=str(error)
        diagnostics['ranks'].append(entry)
    return best_available()


def hermite_change_matrix(source_variance,target_variance,degree=3):
    """Map normalized probabilists' Hermite moments between variances.

    The coefficient definition is integral f He_a(v/s)/sqrt(a!) dv. The
    triangular polynomial identity is exact for arbitrary integrable f.
    """
    if source_variance<=0 or target_variance<=0 or degree<0:
        raise ValueError('positive variances and nonnegative degree required')
    r=math.sqrt(source_variance/target_variance);d=r*r-1
    one=np.zeros((degree+1,degree+1))
    for a in range(degree+1):
        for j in range(a//2+1):
            c=a-2*j
            one[a,c]=(math.sqrt(math.factorial(a)/math.factorial(c))*r**c*d**j
                      /(2**j*math.factorial(j)))
    powers=[(a,total-a) for total in range(degree+1) for a in range(total,-1,-1)]
    return np.array([[one[a,c]*one[b,e] for c,e in powers] for a,b in powers])



def _change_basis(matrix,values,block_size=64):
    """Equivalent BLAS transform with bounded temporary time blocks."""
    result=np.empty_like(values)
    for first in range(0,len(values),block_size):
        last=min(first+block_size,len(values))
        result[first:last]=(matrix@values[first:last].reshape(
            last-first,values.shape[1],-1)).reshape(values[first:last].shape)
    return result


def _evaluate_models(models,matrix,times,block_size=64):
    """Avoid simultaneously retaining all dense thermal source channels."""
    shape=models[0].modes.shape[-2:]
    answer=np.empty((len(times),len(models))+shape,dtype=complex)
    for first in range(0,len(times),block_size):
        last=min(first+block_size,len(times));query=times[first:last]
        thermal=np.empty((last-first,len(models))+shape,dtype=complex)
        for channel,model in enumerate(models):
            thermal[:,channel]=model.evaluate(query)
        answer[first:last]=(matrix@thermal.reshape(
            last-first,len(models),-1)).reshape((last-first,len(models))+shape)
    return answer


class MacroCorrection:
    """Continuous state-basis retained low moments, exact at full frames."""
    def __init__(self,times,modes,models,thermal_to_state,diagnostics):
        self.times=np.asarray(times,float);self.modes=np.asarray(modes)
        self.models=models;self.thermal_to_state=thermal_to_state
        self.enabled=any(model.enabled for model in models)
        self.diagnostics=diagnostics

    def evaluate(self,times):
        scalar=np.ndim(times)==0;t=np.atleast_1d(np.asarray(times,float))
        answer=_evaluate_models(self.models,self.thermal_to_state,t)
        # Avoid loss of acquired-frame exactness in an inverse matrix round trip.
        indices=np.searchsorted(self.times,t);hit=indices<len(self.times)
        hit[hit]&=self.times[indices[hit]]==t[hit]
        answer[hit]=self.modes[indices[hit]]
        answer[:,0,0,0]=0.
        return answer[0] if scalar else answer


def _density_pareto(baseline,weighted):
    """Compare independently gated selected ranks on identical heldout folds."""
    if not weighted.enabled:return False,'weighted density rejected'
    keys=('density_ratio','poisson_ratio','cumulative_density_ratio','cumulative_poisson_ratio')
    def selected_folds(model):
        selected=model.diagnostics.get('selected_rank')
        entry=next((row for row in model.diagnostics.get('ranks',[])
                    if row.get('rank')==selected and row.get('passed')),None)
        if entry is None or len(entry.get('folds',[]))!=2:
            return None
        folds=entry['folds']
        if not all(row.get('passed') for row in folds):return None
        values=np.array([[row.get(key,np.nan) for key in keys] for row in folds])
        return values if np.isfinite(values).all() else None
    new=selected_folds(weighted)
    if new is None:return False,'incomplete selected-rank validation diagnostics'
    if not baseline.enabled:return True,'weighted density accepted after baseline rejection'
    old=selected_folds(baseline)
    if old is None:return False,'incomplete selected-rank validation diagnostics'
    if np.all(new<=old) and np.any(new<old):return True,'weighted density Pareto improves validation'
    return False,'weighted density does not Pareto improve validation'


def prepare_macro_correction(full_times,full_modes,pilot_times,pilot_xy,pilot_values,
        *,theta,variance,periods=(2*np.pi,2*np.pi),corner_times=(),
        ranks=(1,2,3,4,5,6,7,8),should_continue=None,first_passing=False,degree=3,
        optional_density_should_continue=None):
    """Independently validate low thermal channels, then map to state basis.

    Full modes contain complete broad-basis degrees zero through ``degree``;
    pilots contain the corresponding theta-basis moments. A rejected channel retains
    its full-frame cubic. Non-density means are retained and included in both
    the spatial fit and its low-frequency validation norm. Density alone is
    projected to zero mean; its probe mean remains an uncorrected nuisance.
    """
    modes=np.asarray(full_modes,complex)
    values=np.asarray(pilot_values,float)
    if isinstance(degree,bool) or int(degree)!=degree or not 0<=degree<=16:
        raise ValueError('integer correction degree between zero and sixteen required')
    degree=int(degree);count=(degree+1)*(degree+2)//2
    if modes.ndim!=4 or modes.shape[1]!=count or values.ndim!=3 or values.shape[2]!=count:
        raise ValueError('complete equal-degree full and pilot moment channels required')
    if np.max(abs(modes[:,0,0,0]))>1.e-12:
        raise ValueError('full density source must already have zero spatial mean')
    forward=hermite_change_matrix(theta,variance,degree)
    inverse=hermite_change_matrix(variance,theta,degree)
    thermal=_change_basis(inverse,modes)
    thermal[:,0,0,0]=0.
    models=[];diagnostics={'enabled':False,'basis':'thermal','degree':degree,'channels':[]}
    for channel in range(count):
        model=prepare_scalar_correction(full_times,thermal[:,channel],pilot_times,
            pilot_xy,values[:,:,channel],periods=periods,corner_times=corner_times,
            ranks=ranks,should_continue=should_continue,first_passing=first_passing,
            zero_mean=(channel==0),density_weight=0.)
        models.append(model)
        diagnostics['channels'].append(model.diagnostics)
    # Finish every ordinary channel first. The optional trial has a stricter
    # caller-owned deadline and can never replace a baseline with an ungated fit.
    trial_diagnostics={'attempted':False,'selected_weighted':False,
        'baseline_enabled':models[0].enabled,
        'baseline_rank':models[0].diagnostics.get('selected_rank'),
        'reason':'optional density reserve unavailable'}
    if optional_density_should_continue is not None and optional_density_should_continue():
        trial_diagnostics['attempted']=True
        try:
            weighted=prepare_scalar_correction(full_times,thermal[:,0],pilot_times,
                pilot_xy,values[:,:,0],periods=periods,corner_times=corner_times,
                ranks=ranks,should_continue=optional_density_should_continue,
                first_passing=first_passing,zero_mean=True,density_weight=1.)
            select,reason=_density_pareto(models[0],weighted)
            trial_diagnostics.update(selected_weighted=select,reason=reason,
                weighted_enabled=weighted.enabled,
                weighted_rank=weighted.diagnostics.get('selected_rank'),
                weighted_diagnostics=weighted.diagnostics)
            if select:
                trial_diagnostics['baseline_diagnostics']=models[0].diagnostics
                models[0]=weighted
                diagnostics['channels'][0]=weighted.diagnostics
        except (ValueError,ArithmeticError,np.linalg.LinAlgError,MemoryError) as error:
            trial_diagnostics['reason']='optional density fit failed: '+str(error)
    diagnostics['density_pod_trial']=trial_diagnostics
    diagnostics['enabled']=any(model.enabled for model in models)
    diagnostics['accepted_channels']=[i for i,model in enumerate(models) if model.enabled]
    return MacroCorrection(full_times,modes,models,forward,diagnostics)
