"""Research-only guarded density-source correction from independent pilots.

No Oracle, completed solution, GPU, or candidate entrypoint is used. Inputs
are unnormalized density-source Fourier coefficients and same-quadrature
scalar source pilots. Validation excludes full frames from both POD and the
coarse temporal interpolant. This is an empirical gate, not an error bound.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


class PiecewiseCubic:
    def __init__(self,times,values,corners=()):
        self.times=np.asarray(times,float);self.values=np.asarray(values)
        joins=np.flatnonzero(np.isin(self.times,corners))
        indices=np.unique(np.r_[0,joins,len(self.times)-1])
        self.ends=self.times[indices[1:]]
        self.parts=[CubicSpline(self.times[a:b+1],self.values[a:b+1],axis=0)
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
    return np.einsum('...ij,pij->...p',modes,phase).real


def _pod(modes,zero_mean=True):
    n=modes.shape[-1]
    physical=np.fft.ifft2(modes,axes=(-2,-1)).real.reshape(len(modes),-1)
    if zero_mean:physical-=physical.mean(axis=1,keepdims=True)
    physical-=physical.mean(axis=0,keepdims=True)
    _,singular,spatial=np.linalg.svd(physical,full_matrices=False)
    active=singular>max(1.e-13,1.e-10*singular[0]) if len(singular) else []
    spatial=spatial[active][:10]*n
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
    residual=pilot_values-_project(coarse(pilot_times),phase)
    raw=PiecewiseCubic(pilot_times,residual,corners)
    raw_anchor=PiecewiseCubic(times,raw(times),corners)
    residual-=raw_anchor(pilot_times)
    return coarse,residual


def _construct(times,modes,pilot_times,phase,pilot_values,corners,rank,nuisance,
               basis_all=None,coarse=None,residual=None,zero_mean=True):
    basis_all=_pod(modes,zero_mean) if basis_all is None else basis_all
    if len(basis_all)<rank:raise ValueError('insufficient independent POD modes')
    basis=basis_all[:rank]
    # Up to two omitted POD directions absorb unresolved pilot contamination;
    # they are nuisance terms and never enter the spatial correction.
    extra=_project(basis_all[rank:rank+2],phase).T
    nuisance_matrix=np.column_stack((np.ones((len(phase),int(zero_mean))),nuisance,extra))
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
    weight=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    if not zero_mean:weight[0,0]=1.
    energy=abs(error/n**2)**2
    return np.sqrt(np.sum(energy,axis=(-2,-1))),np.sqrt(np.sum(energy*weight,axis=(-2,-1)))


def prepare_density_correction(full_times,full_modes,pilot_times,pilot_xy,pilot_values,
        *,periods=(2*np.pi,2*np.pi),corner_times=(),ranks=(1,2,4,6,8),nuisance=None,
        should_continue=None,first_passing=False,zero_mean=True):
    """Return a gated source interpolant; rejection returns the coarse spline.

    Full-frame heldout folds use two disjoint quarter-index sets. Neither
    heldout full frames nor their POD contribution enters their predictor.
    Dense pilots remain available, as intended for eventual preparation use.
    """
    t=np.asarray(full_times,float);f=np.asarray(full_modes,complex)
    pt=np.asarray(pilot_times,float);xy=np.asarray(pilot_xy,float);pv=np.asarray(pilot_values,float)
    periods=np.asarray(periods,float);corners=np.asarray(corner_times,float)
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
    def continuing():
        if should_continue is None or should_continue():return True
        diagnostics['reason']='cooperative density-correction deadline'
        return False
    if not continuing():return fallback
    eligible=np.flatnonzero(~np.isin(t,np.r_[t[0],t[-1],corners]))
    folds=[eligible[offset::4] for offset in (0,2)]
    if any(len(held)<3 for held in folds):
        diagnostics['reason']='fewer than three independent heldout full frames per fold'
        return fallback
    train_folds=[np.setdiff1d(np.arange(len(t)),held) for held in folds]
    caches=[]
    for train in train_folds+[np.arange(len(t))]:
        if not continuing():return fallback
        basis=_pod(f[train],zero_mean)
        if not continuing():return fallback
        coarse=(fallback.coarse if len(train)==len(t) else None)
        coarse,residual=_residual_cache(t[train],f[train],pt,phase,pv,corners,coarse)
        caches.append((basis,coarse,residual))
        if not continuing():return fallback
    passing=[]
    for rank in ranks:
        rank=int(rank)
        if rank<1 or rank>8:raise ValueError('retained POD rank must be between one and eight')
        entry={'rank':rank,'passed':False,'folds':[]}
        try:
            for held,train,(basis,coarse,residual) in zip(folds,train_folds,caches):
                if not continuing():return fallback
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
            if not continuing():return fallback
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
    if not passing:return fallback
    _,rank,result=min(passing,key=lambda row:(row[0],row[1]))
    diagnostics.update(enabled=True,selected_rank=rank)
    result.diagnostics=diagnostics
    return result
