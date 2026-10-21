"""Guarded Fourier carrier demodulation for a completed source panel.

The input is the ordinary real (time,x,v) source panel. Coarse even times
train a carrier model; odd times test its degree-seven envelope interpolant.
All final panel times remain interpolated exactly. No oracle is used here.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline, make_interp_spline


def _prepare_rows(panel, times, velocity, quadrature, kx, ordinary_coefficients,
            improvement=.01, max_relative_defect=1e-8, max_frequency=np.inf,
            endpoint_source=None, endpoint_delta=None):
    nt,nx,nv=panel.shape
    info=dict(method='fitted_carrier_spline',accepted_modes=0,nt=nt)
    if nt < (33 if endpoint_source is not None else 65):
        return ordinary_coefficients, None, info
    device=panel.device
    time=np.asarray(times)-float(times[0])
    source=torch.fft.fft(panel,dim=1)
    source[:,nx//2]=0
    coarse=source[::2]
    coarse_time=time[::2]
    # Source-weighted temporal phase increments at the two ends avoid
    # underresolved central intervals of the Chebyshev panel.
    weighted=torch.as_tensor(np.asarray(quadrature)*(1+np.asarray(velocity)**6),
                             device=device,dtype=torch.float64)
    near=(None if endpoint_source is None else torch.fft.fft(endpoint_source,dim=1))
    left=((coarse[1] if near is None else near[0])*coarse[0].conj()*weighted).sum(-1)
    right=(coarse[-1]*(coarse[-2] if near is None else near[1]).conj()*weighted).sum(-1)
    delta_left=coarse_time[1]-coarse_time[0] if near is None else endpoint_delta
    delta_right=coarse_time[-1]-coarse_time[-2] if near is None else endpoint_delta
    wl=left.abs(); wr=right.abs()
    omega=(wl*torch.angle(left)/delta_left
           +wr*torch.angle(right)/delta_right)/(wl+wr).clamp_min(1e-300)
    # Pair frequencies explicitly; this also makes the selection invariant
    # to roundoff in conjugate Fourier rows.
    negative=torch.as_tensor((-np.arange(nx))%nx,device=device)
    omega=(omega-omega[negative])/2
    omega[0]=0.; omega[nx//2]=0.
    phase=torch.exp(-1j*torch.as_tensor(time,device=device)[:,None]*omega[None,:])
    transformed=source*phase[:,:,None]
    interpolation=make_interp_spline(coarse_time,np.eye(len(coarse_time)),k=7)(time[1::2])
    matrix=torch.as_tensor(interpolation,device=device,dtype=torch.complex128)
    original_hold=(matrix@coarse.reshape(len(coarse_time),-1)).reshape(-1,nx,nv)
    model_hold=(matrix@transformed[::2].reshape(len(coarse_time),-1)).reshape(-1,nx,nv)/phase[1::2,:,None]
    exact_hold=source[1::2]
    baseline_error=((original_hold-exact_hold).abs()**2*weighted).sum((0,2))
    model_error=((model_hold-exact_hold).abs()**2*weighted).sum((0,2))
    source_norm=(exact_hold.abs()**2*weighted).sum((0,2))
    selected=(model_error<improvement*baseline_error)&(model_error<max_relative_defect*source_norm)
    selected &= source_norm>source_norm.max()*1e-20
    selected &= baseline_error>source_norm*1e-20
    phase_rejected=int((selected & (omega.abs()>max_frequency)).sum().item())
    selected &= omega.abs()<=max_frequency
    selected &= selected[negative]
    selected[0]=False;selected[nx//2]=False
    indices=selected.nonzero().ravel()
    count=len(indices)
    info.update(accepted_modes=count,phase_rejected_modes=phase_rejected,
                frequency_cap=float(max_frequency),
                baseline_defect=float(baseline_error.sum().sqrt().item()),
                model_defect=float(model_error.sum().sqrt().item()),
                selected_baseline_defect=float(baseline_error[selected].sum().sqrt().item()),
                selected_model_defect=float(model_error[selected].sum().sqrt().item()))
    if not count:
        return ordinary_coefficients,None,info
    spline=make_interp_spline(time,np.eye(nt),k=7,axis=0)
    transform=torch.as_tensor(spline.c,device=device,dtype=torch.complex128)
    modified=(transform@transformed[:,indices].reshape(nt,-1)).reshape(nt,count,nv)
    coefficients=ordinary_coefficients.clone()
    coefficients[:,indices]=modified
    frequency=torch.where(selected,omega,0.)
    info['max_frequency']=float(frequency.abs().max().item())
    return coefficients,frequency,info


def prepare(panel, times, velocity, quadrature, kx, ordinary_coefficients,
            improvement=.01, max_relative_defect=1e-8, max_frequency=np.inf,
            endpoint_source=None, endpoint_delta=None):
    """Fit extra carriers independently on complete six-node DG cells.

    The inherited shared-velocity row model is considered first. Cell
    carriers are added only to rejected rows, and conjugate cells must both
    pass the same training/held-out tests. No interpolation model is fitted
    to individual Gauss nodes.
    """
    coefficients, frequency, info = _prepare_rows(
        panel, times, velocity, quadrature, kx, ordinary_coefficients,
        improvement, max_relative_defect, max_frequency,
        endpoint_source, endpoint_delta)
    info['shared_carrier_modes']=int(info['accepted_modes'])
    nt, nx, nv = panel.shape
    info.update(accepted_cells=0,cell_carrier_method='six_node_dg_cell')
    ng = 6
    if nt < (33 if endpoint_source is not None else 65) or nv % ng:
        return coefficients, frequency, info
    nc = nv//ng
    device = panel.device
    time = np.asarray(times)-float(times[0])
    coarse_time = time[::2]
    source = torch.fft.fft(panel,dim=1).reshape(nt,nx,nc,ng)
    source[:,nx//2] = 0
    coarse = source[::2]
    weighted = torch.as_tensor(np.asarray(quadrature)*(1+np.asarray(velocity)**6),
                               device=device,dtype=torch.float64).reshape(nc,ng)
    near=(None if endpoint_source is None else torch.fft.fft(endpoint_source,dim=1).reshape(2,nx,nc,ng))
    left=((coarse[1] if near is None else near[0])*coarse[0].conj()*weighted).sum(-1)
    right=(coarse[-1]*(coarse[-2] if near is None else near[1]).conj()*weighted).sum(-1)
    delta_left=coarse_time[1]-coarse_time[0] if near is None else endpoint_delta
    delta_right=coarse_time[-1]-coarse_time[-2] if near is None else endpoint_delta
    wl,wr=left.abs(),right.abs()
    omega=(wl*torch.angle(left)/delta_left
           +wr*torch.angle(right)/delta_right)/(wl+wr).clamp_min(1e-300)
    negative=torch.as_tensor((-np.arange(nx))%nx,device=device)
    omega=(omega-omega[negative])/2
    omega[0]=0.;omega[nx//2]=0.
    phase=torch.exp(-1j*torch.as_tensor(time,device=device)[:,None,None]*omega[None])
    transformed=source*phase[:,:,:,None]
    interpolation=make_interp_spline(coarse_time,np.eye(len(coarse_time)),k=7)(time[1::2])
    matrix=torch.as_tensor(interpolation,device=device,dtype=torch.complex128)
    original=(matrix@coarse.reshape(len(coarse_time),-1)).reshape(-1,nx,nc,ng)
    modeled=(matrix@transformed[::2].reshape(len(coarse_time),-1)).reshape(-1,nx,nc,ng)
    modeled/=phase[1::2,:,:,None]
    exact=source[1::2]
    baseline_error=((original-exact).abs()**2*weighted).sum((0,3))
    model_error=((modeled-exact).abs()**2*weighted).sum((0,3))
    norm=(exact.abs()**2*weighted).sum((0,3))
    selected=(model_error<improvement*baseline_error)&(model_error<max_relative_defect*norm)
    selected&=norm>norm.max()*1e-20
    selected&=baseline_error>norm*1e-20
    selected&=omega.abs()<=max_frequency
    selected&=selected[negative]
    selected[0]=False;selected[nx//2]=False
    if frequency is not None:
        selected[frequency!=0]=False
    count=int(selected.sum().item())
    info.update(accepted_cells=count,
                endpoint_seeded=endpoint_source is not None,
                endpoint_delta=endpoint_delta,
                cell_baseline_defect=float(baseline_error[selected].sum().sqrt().item()),
                cell_model_defect=float(model_error[selected].sum().sqrt().item()))
    if not count:
        return coefficients,frequency,info
    spline=make_interp_spline(time,np.eye(nt),k=7)
    transform=torch.as_tensor(spline.c,device=device,dtype=torch.complex128)
    modified=(transform@transformed[:,selected].reshape(nt,-1)).reshape(nt,count,ng)
    coefficients=coefficients.clone()
    coefficients.reshape(nt,nx,nc,ng)[:,selected]=modified
    base=torch.zeros((nx,nv),device=device,dtype=torch.float64)
    if frequency is not None:
        base+=frequency[:,None]
    cell_frequency=base.reshape(nx,nc,ng)
    cell_frequency[selected]=omega[selected][:,None]
    info['accepted_modes']=int((base!=0).any(-1).sum().item())
    info['max_frequency']=float(base.abs().max().item())
    return coefficients,base,info


def validate(panel,times,x,velocity,quadrature,ordinary_coefficients,
             coefficients,frequency,info,query,reserve_exception):
    """Reject a model that fails an independent sparse exact-source check.

    Four irrational times and existing x/v nodes form at most4096points.
    Existing nodes avoid adding spatial or velocity interpolation errors.
    This is a finite heuristic; it does not certify unobserved source values.
    """
    if frequency is None:
        return coefficients,frequency,info
    nt,nx,nv=panel.shape
    device=panel.device
    fractions=np.array([np.pi-3,np.sqrt(5)-2,np.sqrt(2)-1,np.sqrt(3)-1])
    times=np.asarray(times)
    relative=fractions*(times[-1]-times[0])
    probe_times=times[0]+relative
    rng=np.random.default_rng(471193)
    nxprobe=min(32,nx);nvprobe=min(32,nv)
    # Sixteen velocity strata cover tails; sixteen energy quantiles cover
    # narrow active cores. Fill duplicate selections without replacement.
    weighted=np.asarray(quadrature)*(1+np.asarray(velocity)**6)
    power=(panel.square().sum((0,1))*torch.as_tensor(weighted,device=device)).cpu().numpy()
    if not np.isfinite(power).all() or power.sum()<=0:
        info.update(accepted_modes=0,validation_reason='invalid_source_power')
        return ordinary_coefficients,None,info
    cdf=np.cumsum(power);cdf/=cdf[-1]
    count=nvprobe//2
    vi=np.unique(np.r_[np.searchsorted(cdf,(np.arange(count)+.5)/count),
                       np.floor((np.arange(nvprobe-count)+.5)*nv/(nvprobe-count)).astype(int)])
    if len(vi)<nvprobe:
        available=np.setdiff1d(np.arange(nv),vi,assume_unique=True)
        vi=np.sort(np.r_[vi,rng.choice(available,nvprobe-len(vi),replace=False)])
    cell_carriers=frequency.ndim==2
    selected_cells=np.array([],dtype=int)
    if cell_carriers:
        # Every cell carrying an accepted model receives two exact probes.
        # Keep all six nodes together during fitting, then probe the two
        # nodes with largest weighted source energy in each selected cell.
        active=(frequency!=0).reshape(nx,nv//6,6).any((0,2)).cpu().numpy()
        selected_cells=np.flatnonzero(active)
        node_order=np.argsort(power.reshape(nv//6,6),axis=1)[:,-2:]
        extra=(6*selected_cells[:,None]+node_order[selected_cells]).ravel()
        vi=np.unique(np.r_[vi,extra])
        nvprobe=len(vi)
    xindices=[];points=[]
    for t in probe_times:
        edges=np.linspace(0,nx,nxprobe+1).astype(int)
        xi=np.array([rng.integers(edges[i],edges[i+1]) for i in range(nxprobe)])
        xindices.append(xi)
        points.append(np.column_stack((np.full(nxprobe*nvprobe,t),
                                       np.repeat(np.asarray(x)[xi],nvprobe),
                                       np.tile(np.asarray(velocity)[vi],nxprobe))))
    points=np.concatenate(points)
    try:
        exact=np.asarray(query(points,reserve=20.)).reshape(4,nxprobe,nvprobe)
    except reserve_exception:
        info.update(accepted_modes=0,validation_reason='query_budget')
        return ordinary_coefficients,None,info
    if not np.isfinite(exact).all():
        info.update(accepted_modes=0,validation_reason='nonfinite_probe')
        return ordinary_coefficients,None,info
    spline=make_interp_spline(times-times[0],np.eye(nt),k=7)
    matrix=torch.as_tensor(BSpline.design_matrix(relative,spline.t,spline.k).toarray(),
                           device=device,dtype=torch.complex128)
    def evaluate(coefficients,carrier):
        modes=(matrix@coefficients.reshape(nt,-1)).reshape(4,nx,nv)
        if carrier is not None:
            if carrier.ndim==1:
                modes*=torch.exp(1j*torch.as_tensor(relative,device=device)[:,None]*carrier[None,:])[:,:,None]
            else:
                modes*=torch.exp(1j*torch.as_tensor(relative,device=device)[:,None,None]*carrier[None])
        values=torch.fft.ifft(modes,dim=1).real.cpu().numpy()
        return np.stack([values[j][xi][:,vi] for j,xi in enumerate(xindices)])
    original=evaluate(ordinary_coefficients,None)
    modeled=evaluate(coefficients,frequency)
    scale=weighted[vi][None,None,:]
    baseline_error=float(np.sum((original-exact)**2*scale))
    model_error=float(np.sum((modeled-exact)**2*scale))
    norm=float(np.sum(exact**2*scale))
    floor=1e-24*max(norm,1e-300)
    accepted=np.isfinite(model_error) and model_error<=1.01*baseline_error+floor
    failed_cells=[]
    if cell_carriers:
        for cell in selected_cells:
            mask=vi//6==cell
            original_defect=float(np.sum((original[:,:,mask]-exact[:,:,mask])**2*scale[:,:,mask]))
            modeled_defect=float(np.sum((modeled[:,:,mask]-exact[:,:,mask])**2*scale[:,:,mask]))
            cell_norm=float(np.sum(exact[:,:,mask]**2*scale[:,:,mask]))
            if not np.isfinite(modeled_defect) or modeled_defect>1.01*original_defect+1e-24*max(cell_norm,1e-300):
                failed_cells.append(int(cell))
        info.update(validation_cells=len(selected_cells),validation_failed_cells=failed_cells)
        if failed_cells:
            # Velocity cells are independent interpolation models. Restore
            # precisely the failed cells, retaining validated cells and their
            # unchanged ordinary fallback. This avoids allowing negligible
            # numerical tails to discard a well-resolved warm population.
            coefficients=coefficients.clone()
            frequency=frequency.clone()
            for cell in failed_cells:
                indices=slice(6*cell,6*(cell+1))
                coefficients[:,:,indices]=ordinary_coefficients[:,:,indices]
                frequency[:,indices]=0.
                modeled[:,:,vi//6==cell]=original[:,:,vi//6==cell]
            model_error=float(np.sum((modeled-exact)**2*scale))
            accepted=np.isfinite(model_error) and model_error<=1.01*baseline_error+floor
            info['accepted_modes']=int((frequency!=0).any(-1).sum().item())
            info['validated_nonzero_cells']=int((frequency!=0).reshape(nx,nv//6,6).any(-1).sum().item())
        if not bool((frequency!=0).any().item()):
            info.update(accepted_modes=0,validation_reason='all_cell_models_rejected')
            return ordinary_coefficients,None,info
    info.update(validation_points=len(points),validation_baseline_error=baseline_error,
                validation_model_error=model_error,validation_norm=norm,
                validation_reason='passed' if accepted else 'independent_source_defect')
    if not accepted:
        info['rejected_modes']=info['accepted_modes'];info['accepted_modes']=0
        return ordinary_coefficients,None,info
    return coefficients,frequency,info
