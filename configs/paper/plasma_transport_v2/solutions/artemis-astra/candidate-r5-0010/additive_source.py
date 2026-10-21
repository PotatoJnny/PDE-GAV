"""Guarded two-tone extraction with an ordinary spline residual.

Two endpoint snapshot pairs seed a weighted rank-two matrix pencil in each
whole G6 velocity cell. Even times fit amplitudes; odd times and independent
causal queries validate the model. Existing accepted carriers are untouched.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import BSpline,make_interp_spline


def fit_amplitudes(values,times,frequency):
    phase=torch.exp(1j*times[:,None,None]*frequency[None])
    cross=(phase[:,:,0].conj()*phase[:,:,1]).sum(0)
    rhs=torch.einsum('tcj,tcg->cjg',phase.conj(),values)
    n=float(len(times));det=(n*n-cross.abs()**2).clamp_min(1e-12*n*n)
    first=(n*rhs[:,0]-cross[:,None]*rhs[:,1])/det[:,None]
    second=(n*rhs[:,1]-cross.conj()[:,None]*rhs[:,0])/det[:,None]
    return torch.stack((first,second),dim=1)


class ToneForcing:
    def __init__(self,rows,cells,frequency,amplitude,times,spline,kx,velocity,phi,info):
        device=amplitude.device
        self.spatial_rows=rows
        self.cells=cells
        self.velocity_columns=6*cells[:,None]+torch.arange(6,device=device)[None]
        self.indices=(rows[:,None],self.velocity_columns)
        self.frequency=frequency
        self.amplitude=amplitude
        self.tmin=float(times[0])
        self.info=info
        self.phi=phi
        self.plans={}
        self.kx=np.asarray(kx)
        self.nmodels=len(rows)
        phase=torch.exp(1j*torch.as_tensor(np.asarray(times)-self.tmin,device=device)[:,None,None]*frequency[None])
        tone_values=torch.einsum('tcj,cjg->tcg',phase,amplitude)
        transform=torch.as_tensor(spline.c,device=device,dtype=torch.complex128)
        self.coefficients=transform@tone_values.reshape(len(times),-1)
        self.linear=np.asarray(kx)[rows.cpu().numpy(),None]*np.asarray(velocity)[self.velocity_columns.cpu().numpy()]

    def subtract(self,value,start,weights):
        correction=(weights@self.coefficients[start:start+8]).reshape(self.nmodels,6)
        value[self.indices]=value[self.indices]-correction

    def at(self,level,h,tick):
        if level not in self.plans:
            omega=self.frequency.cpu().numpy()
            linear=-1j*(self.linear[:,None,:]+omega[:,:,None])
            half=torch.as_tensor((h/2)*self.phi(linear*h/2,1),device=self.amplitude.device)*self.amplitude
            full=torch.as_tensor(h*self.phi(linear*h,1),device=self.amplitude.device)*self.amplitude
            self.plans[level]=(half,full)
        half,full=self.plans[level]
        phase_half=torch.exp(1j*((tick+.5)*h)*self.frequency)
        phase_full=torch.exp(1j*((tick+1.)*h)*self.frequency)
        return (half*phase_half[:,:,None]).sum(1),(full*phase_full[:,:,None]).sum(1)


def prepare(panel,times,x,velocity,quadrature,kx,coefficients,carrier,
            endpoint_source,endpoint_delta,spline,query,reserve_exception,phi,
            max_models=16384):
    nt,nx,nv=panel.shape;ng=6;nc=nv//ng
    info=dict(method='two_tone_pencil_with_ordinary_residual',accepted_models=0,
              fitted_models=0,reason='insufficient_data')
    if nt<33 or nv%6 or endpoint_source is None:
        return None,info
    device=panel.device
    time=np.asarray(times)-float(times[0]);coarse_time=time[::2]
    tt=torch.as_tensor(time,device=device)
    source=torch.fft.fft(panel,dim=1).reshape(nt,nx,nc,ng)
    source[:,nx//2]=0.
    near=torch.fft.fft(endpoint_source,dim=1).reshape(2,nx,nc,ng)
    weight=torch.as_tensor(np.asarray(quadrature)*(1+np.asarray(velocity)**6),device=device).reshape(nc,ng)
    coarse=source[::2,1:nx//2]
    exact=source[1::2,1:nx//2]
    matrix=torch.as_tensor(make_interp_spline(coarse_time,np.eye(len(coarse_time)),k=7)(time[1::2]),device=device,dtype=torch.complex128)
    ordinary=(matrix@coarse.reshape(len(coarse_time),-1)).reshape_as(exact)
    baseline_error=((ordinary-exact).abs()**2*weight).sum((0,3))
    norm=(exact.abs()**2*weight).sum((0,3))
    available=(norm>norm.max()*1e-20)&(baseline_error>norm*1e-20)
    if carrier is not None:
        if carrier.ndim==1:
            available&=(carrier[1:nx//2]==0)[:,None]
        else:
            available&=~(carrier[1:nx//2]!=0).reshape(nx//2-1,nc,ng).any(-1)
    priority=torch.where(available,baseline_error,torch.zeros_like(baseline_error)).ravel()
    count=min(max_models,int(available.sum().item()))
    if not count:
        info['reason']='no_rejected_cells'
        return None,info
    chosen=torch.topk(priority,count,sorted=False).indices
    rows=chosen//nc+1;cells=chosen%nc
    values=source[:,rows,cells]
    weights=weight[cells]
    xx=torch.stack((values[0],near[-1,rows,cells]),dim=-1)
    yy=torch.stack((near[0,rows,cells],values[-1]),dim=-1)
    gram=torch.einsum('cgj,cgk->cjk',xx.conj()*weights[:,:,None],xx)
    cross=torch.einsum('cgj,cgk->cjk',xx.conj()*weights[:,:,None],yy)
    a=gram[:,0,0].real;d=gram[:,1,1].real;b=gram[:,0,1]
    determinant=a*d-b.abs()**2
    rank_ok=determinant>1e-12*(a+d)**2
    # Solve the weighted snapshot pencil with reorthogonalized QR. The
    # original Gram determinant above still defines the eligibility mask.
    weighted_x=xx*weights.sqrt()[:,:,None]
    weighted_y=yy*weights.sqrt()[:,:,None]
    weighted_x=torch.where(rank_ok[:,None,None],weighted_x,0.)
    weighted_y=torch.where(rank_ok[:,None,None],weighted_y,0.)
    r00=torch.linalg.vector_norm(weighted_x[:,:,0],dim=-1)
    safe0=torch.where(r00>0.,r00,torch.ones_like(r00))
    q0=weighted_x[:,:,0]/safe0[:,None]
    r01=(q0.conj()*weighted_x[:,:,1]).sum(-1)
    residual=weighted_x[:,:,1]-q0*r01[:,None]
    correction=(q0.conj()*residual).sum(-1)
    residual=residual-q0*correction[:,None]
    r01=r01+correction
    r11=torch.linalg.vector_norm(residual,dim=-1)
    safe1=torch.where(r11>0.,r11,torch.ones_like(r11))
    q1=residual/safe1[:,None]
    c0=(q0.conj()[:,:,None]*weighted_y).sum(1)
    c1=(q1.conj()[:,:,None]*weighted_y).sum(1)
    b10=c1[:,0]/safe1
    b11=c1[:,1]/safe1
    b00=(c0[:,0]-r01*b10)/safe0
    b01=(c0[:,1]-r01*b11)/safe0
    trace=b00+b11
    root=torch.sqrt(trace**2-4*(b00*b11-b01*b10))
    eigenvalues=torch.stack(((trace+root)/2,(trace-root)/2),dim=-1)
    frequency=torch.angle(eigenvalues)/float(endpoint_delta)
    rank_ok&=torch.isfinite(frequency).all(-1)
    rank_ok&=((eigenvalues.abs()-1).abs()<.25).all(-1)
    rank_ok&=(frequency[:,0]-frequency[:,1]).abs()*(time[-1]-time[0])>1e-3
    amplitude=fit_amplitudes(values[::2],tt[::2],frequency)
    phase=torch.exp(1j*tt[:,None,None]*frequency[None])
    fitted_hold=torch.einsum('tcj,cjg->tcg',phase[1::2]-(matrix@phase[::2].reshape(len(coarse_time),-1)).reshape(-1,count,2),amplitude)
    base=ordinary[:,rows-1,cells]
    proposed=base+fitted_hold
    truth=exact[:,rows-1,cells]
    error=((proposed-truth).abs()**2*weights).sum((0,2))
    base_error=baseline_error[rows-1,cells];source_norm=norm[rows-1,cells]
    accepted=rank_ok&(error<.01*base_error)&(error<1e-4*source_norm)
    info.update(fitted_models=count,rank_two_models=int(rank_ok.sum().item()),
                holdout_models=int(accepted.sum().item()),
                holdout_baseline_error=float(base_error[accepted].sum().item()),
                holdout_model_error=float(error[accepted].sum().item()))
    if not bool(accepted.any().item()):
        info['reason']='holdout_rejected'
        return None,info
    rows=rows[accepted];cells=cells[accepted];frequency=frequency[accepted]
    amplitude=fit_amplitudes(values[:,accepted],tt,frequency)
    # Explicit conjugate pairing preserves real source and excludes k=0.
    rows=torch.cat((rows,nx-rows));cells=torch.cat((cells,cells))
    frequency=torch.cat((frequency,-frequency))
    amplitude=torch.cat((amplitude,amplitude.conj()))
    model=ToneForcing(rows,cells,frequency,amplitude,times,spline,kx,velocity,phi,info)
    del source,near,coarse,exact,ordinary,values
    return validate(model,panel,times,x,velocity,quadrature,coefficients,carrier,
                    spline,query,reserve_exception,info)


def validate(model,panel,times,x,velocity,quadrature,coefficients,carrier,
             spline,query,reserve_exception,info):
    nt,nx,nv=panel.shape;device=panel.device
    relative=(times[-1]-times[0])*np.array([np.pi-3,np.sqrt(5)-2,np.sqrt(2)-1,np.sqrt(3)-1])
    probes=times[0]+relative
    weighted=np.asarray(quadrature)*(1+np.asarray(velocity)**6)
    power=(panel.square().sum((0,1))*torch.as_tensor(weighted,device=device)).cpu().numpy()
    cells=np.unique(model.cells.cpu().numpy())
    node_order=np.argsort(power.reshape(-1,6),axis=1)[:,-2:]
    vi=(6*cells[:,None]+node_order[cells]).ravel()
    rng=np.random.default_rng(836911)
    nxprobe=min(32,nx);nvprobe=len(vi)
    xindices=[];points=[]
    for time in probes:
        edges=np.linspace(0,nx,nxprobe+1).astype(int)
        xi=np.array([rng.integers(edges[i],edges[i+1]) for i in range(nxprobe)])
        xindices.append(xi)
        points.append(np.column_stack((np.full(nxprobe*nvprobe,time),np.repeat(np.asarray(x)[xi],nvprobe),np.tile(np.asarray(velocity)[vi],nxprobe))))
    points=np.concatenate(points)
    try:
        truth=np.asarray(query(points,reserve=30.)).reshape(4,nxprobe,nvprobe)
    except reserve_exception:
        info['reason']='validation_budget'
        return None,info
    if not np.isfinite(truth).all():
        info['reason']='nonfinite_validation'
        return None,info
    basis=torch.as_tensor(BSpline.design_matrix(probes,spline.t,spline.k).toarray(),device=device,dtype=torch.complex128)
    modes=(basis@coefficients.reshape(nt,-1)).reshape(4,nx,nv)
    if carrier is not None:
        phase=torch.as_tensor(relative,device=device)
        if carrier.ndim==1:
            modes*=torch.exp(1j*phase[:,None]*carrier[None])[:,:,None]
        else:
            modes*=torch.exp(1j*phase[:,None,None]*carrier[None])
    original=torch.fft.ifft(modes,dim=1).real.cpu().numpy()
    correction=(basis@model.coefficients).reshape(4,model.nmodels,6)
    phase=torch.exp(1j*torch.as_tensor(relative,device=device)[:,None,None]*model.frequency[None])
    correction=torch.einsum('tcj,cjg->tcg',phase,model.amplitude)-correction
    modes[:,model.indices[0],model.indices[1]]+=correction
    proposed=torch.fft.ifft(modes,dim=1).real.cpu().numpy()
    original=np.stack([original[j][xi][:,vi] for j,xi in enumerate(xindices)])
    proposed=np.stack([proposed[j][xi][:,vi] for j,xi in enumerate(xindices)])
    scale=weighted[vi][None,None,:]
    old=(original-truth)**2*scale;new=(proposed-truth)**2*scale
    norm=truth**2*scale
    failed=[]
    for cell in cells:
        selected=vi//6==cell
        if (not np.isfinite(new[:,:,selected]).all() or
            new[:,:,selected].sum()>1.01*old[:,:,selected].sum()+1e-24*max(norm[:,:,selected].sum(),1e-300)):
            failed.append(int(cell))
    info.update(validation_points=len(points),validation_cells=len(cells),validation_failed_cells=failed,
                validation_baseline_error=float(old.sum()),validation_model_error=float(new.sum()))
    if failed:
        keep=~np.isin(model.cells.cpu().numpy(),failed)
        if not np.any(keep):
            info['reason']='independent_validation_rejected'
            return None,info
        keep=torch.as_tensor(keep,device=device)
        model=ToneForcing(model.spatial_rows[keep],model.cells[keep],model.frequency[keep],model.amplitude[keep],times,spline,
                          model.kx,velocity,model.phi,info)
    info.update(accepted_models=model.nmodels,reason='passed',
                max_frequency=float(model.frequency.abs().max().item()))
    return model,info
