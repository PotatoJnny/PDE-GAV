"""CPU research: one validated temporal carrier per Fourier row/DG cell.

No solver integration is included. A future whole-state transform must undo
the full (k,v) phase before every physical DG/Poisson evaluation and restore
it on the nonlinear derivative. Returned coefficients demodulate source only
on accepted complete six-node cells; all other coefficients remain unchanged.
"""
import numpy as np
import torch
from scipy.interpolate import make_interp_spline

ORDER = 6
GATE_RATIO = .25
CELL_RESOLUTION = .03
MIN_RELATIVE_DEFECT = 1.e-10
MIN_RELATIVE_CELL_ENERGY = 1.e-20
MACRO_GUARD_ABSOLUTE = 1.e-10
MOMENT_SCALES = np.array([1.0041253663782175,.4788345668393978,
                          2.930763214320111,1.9801548942236534])


def improve_cells(panel, times, kx, velocity, quadrature,
                  coefficients, transform, validate=None, carrier_probe=None,
                  macro_guard=True):
    nt,nx,nv=panel.shape
    info=dict(method='cell_carrier_research',source_nodes=nt,accepted_cells=0,
              accepted_rows=0,validation_completed=False)
    mask=torch.zeros((nx,nv),dtype=torch.bool,device=panel.device)
    omega=np.zeros((nx,nv))
    if nt<33 or nt%2==0 or nv%ORDER or validate is None:
        info['reason']='complete_odd_panel_and_exact_validation_required'
        return coefficients,mask,info,omega
    nc=nv//ORDER
    q=torch.as_tensor(quadrature,dtype=torch.float64,device=panel.device).reshape(1,nc,ORDER)
    src=torch.fft.fft(panel,dim=1);src[:,nx//2]=0.
    cells=src.reshape(nt,nx,nc,ORDER)
    dt=float(times[1]-times[0])
    overlap=torch.sum(cells[0].conj()*cells[1]*q,dim=-1)
    frequency=-np.angle(overlap.cpu().numpy())/dt
    wraps=np.zeros((nx,nc))
    if carrier_probe is not None:
        probe=carrier_probe()
        if probe is not None:
            pt,pv=probe
            short_dt=float(pt-times[0])
            if 0.<short_dt<dt and np.isfinite(pv).all():
                early=torch.fft.fft(torch.as_tensor(pv,dtype=panel.dtype,device=panel.device),dim=0)
                early[nx//2]=0.
                short_overlap=torch.sum(cells[0].conj()*early.reshape(nx,nc,ORDER)*q,dim=-1)
                short_frequency=-np.angle(short_overlap.cpu().numpy())/short_dt
                wraps=np.rint((short_frequency-frequency)*dt/(2*np.pi))
                wraps=np.where(np.isfinite(wraps),wraps,0.)
                frequency=np.where(wraps!=0,frequency+2*np.pi*wraps/dt,frequency)
                info.update(early_probe_time=float(pt),early_probe_delta=short_dt)
    partners=(-np.arange(nx))%nx
    frequency=.5*(frequency-frequency[partners])
    frequency[0]=frequency[nx//2]=0.
    expanded=np.repeat(frequency,ORDER,axis=1)
    dtimes=torch.as_tensor(np.asarray(times)-times[0],device=panel.device)
    freq=torch.as_tensor(expanded,device=panel.device)
    demod=torch.exp(1j*dtimes[:,None,None]*freq[None])*src
    coarse=make_interp_spline(times[::2],np.eye((nt+1)//2),k=7)(times[1::2])
    coarse=torch.as_tensor(coarse,dtype=src.dtype,device=panel.device)
    raw_pred=(coarse@src[::2].reshape((nt+1)//2,-1)).reshape((nt-1)//2,nx,nv)
    frame_pred=(coarse@demod[::2].reshape((nt+1)//2,-1)).reshape_as(raw_pred)
    def energy(values):
        return torch.mean(torch.sum(abs(values.reshape(-1,nx,nc,ORDER))**2*q,dim=-1),dim=0)
    raw_error=energy(raw_pred-src[1::2])
    frame_error=energy(frame_pred-demod[1::2])
    norm=energy(src[1::2])
    significant=norm>MIN_RELATIVE_CELL_ENERGY*torch.max(norm)
    preliminary=(significant & torch.isfinite(raw_error) & torch.isfinite(frame_error)
                 & (frame_error<=GATE_RATIO**2*raw_error)
                 & (raw_error>MIN_RELATIVE_DEFECT**2*norm))
    info['coarse_cells']=int(preliminary.sum().item())
    if not bool(preliminary.any().item()):
        info['reason']='coarse_rejected'
        return coefficients,mask,info,omega
    checked=validate()
    if checked is None:
        info['reason']='exact_validation_unavailable'
        return coefficients,mask,info,omega
    check_times,check_values,diagnostic=checked
    if not np.isfinite(check_values).all():
        info['reason']='nonfinite_validation'
        return coefficients,mask,info,omega
    actual=torch.fft.fft(torch.as_tensor(check_values,dtype=panel.dtype,device=panel.device),dim=1)
    actual[:,nx//2]=0.
    fine=make_interp_spline(times,np.eye(nt),k=7)(check_times)
    fine=torch.as_tensor(fine,dtype=src.dtype,device=panel.device)
    raw_fine=(fine@src.reshape(nt,-1)).reshape(len(check_times),nx,nv)
    frame_fine=(fine@demod.reshape(nt,-1)).reshape_as(raw_fine)
    check_dt=torch.as_tensor(np.asarray(check_times)-times[0],device=panel.device)
    actual_frame=actual*torch.exp(1j*check_dt[:,None,None]*freq[None])
    raw_fine_error=energy(raw_fine-actual)
    frame_fine_error=energy(frame_fine-actual_frame)
    fine_norm=energy(actual)
    accepted=(preliminary & torch.isfinite(raw_fine_error) & torch.isfinite(frame_fine_error)
              & (fine_norm>MIN_RELATIVE_CELL_ENERGY*torch.max(fine_norm))
              & (frame_fine_error<=GATE_RATIO**2*raw_fine_error)
              & (frame_fine_error<=CELL_RESOLUTION**2*fine_norm)
              & (raw_fine_error>MIN_RELATIVE_DEFECT**2*fine_norm))
    accepted&=accepted[torch.as_tensor(partners,device=panel.device)]
    accepted[0]=accepted[nx//2]=False
    if macro_guard and bool(accepted.any().item()):
        mask=accepted.repeat_interleave(ORDER,dim=1)
        moment=torch.as_tensor(np.stack([np.asarray(quadrature)*np.asarray(velocity)**p/MOMENT_SCALES[p]
            for p in range(4)]),dtype=src.dtype,device=panel.device)
        streaming=torch.as_tensor(np.asarray(kx)[:,None]*np.asarray(velocity)[None,:],device=panel.device)
        row_pass=torch.ones(nx,dtype=torch.bool,device=panel.device)
        before=int(accepted.any(dim=1).sum().item())
        worst_excess=0.
        for label,test_times,raw_test,frame_test,truth in (
                ('coarse',np.asarray(times)[1::2],raw_pred,frame_pred,src[1::2]),
                ('fine',np.asarray(check_times),raw_fine,frame_fine,actual)):
            offset=torch.as_tensor(test_times-times[0],device=panel.device)
            frame_physical=frame_test*torch.exp(-1j*offset[:,None,None]*freq[None])
            hybrid=torch.where(mask[None],frame_physical,raw_test)
            raw_defect=raw_test-truth
            hybrid_defect=hybrid-truth
            for kind in ('direct','ballistic'):
                if kind=='ballistic':
                    remain=torch.as_tensor(times[-1]-test_times,device=panel.device)
                    ballistic=torch.exp(-1j*remain[:,None,None]*streaming[None])
                    raw_defect=raw_defect*ballistic
                    hybrid_defect=hybrid_defect*ballistic
                raw_macro=abs(raw_defect@moment.T)*(times[-1]-times[0])/nx
                hybrid_macro=abs(hybrid_defect@moment.T)*(times[-1]-times[0])/nx
                excess=hybrid_macro-raw_macro
                passed=torch.all(excess<=MACRO_GUARD_ABSOLUTE,dim=(0,2))
                info[f'{label}_{kind}_failed_rows']=int((~passed & accepted.any(dim=1)).sum().item())
                row_pass&=passed
                worst_excess=max(worst_excess,float(excess.max().item()))
        row_pass&=row_pass[torch.as_tensor(partners,device=panel.device)]
        accepted&=row_pass[:,None]
        info.update(macro_guard=True,macro_guard_absolute=MACRO_GUARD_ABSOLUTE,
                    macro_guard_rows_before=before,
                    macro_guard_rejected_rows=before-int(accepted.any(dim=1).sum().item()),
                    macro_guard_maximum_excess=worst_excess)
    # Compare the complete hybrid row against another whole-row source
    # representation: rejected cells retain their raw interpolation error.
    info['_validated_row_error']=len(check_times)*torch.where(
        accepted,frame_fine_error,raw_fine_error).sum(dim=1)
    info['_row_exception_bound']=torch.zeros(nx,dtype=torch.float64,device=panel.device)
    mask=accepted.repeat_interleave(ORDER,dim=1)
    if bool(mask.any().item()):
        fitted=transform.to(src.dtype)@demod[:,mask]
        coefficients[:,mask]=fitted
    omega=np.repeat(np.where(accepted.cpu().numpy(),frequency,0.),ORDER,axis=1)
    info.update(accepted_cells=int(accepted.sum().item()),accepted_rows=int(accepted.any(dim=1).sum().item()),
                accepted_unwrapped_cells=int(np.count_nonzero(accepted.cpu().numpy()&(wraps!=0))),
                validation_completed=True,validation_times=np.asarray(check_times).tolist(),
                maximum_accepted_carrier=float(np.max(abs(omega))),
                reason='accepted' if bool(mask.any().item()) else 'fine_rejected',
                source_energy_fraction=float(norm[accepted].sum()/norm.sum()) if float(norm.sum()) else 0.)
    return coefficients,mask,info,omega


def improve_coefficients(panel, times, kx, velocity, quadrature,
                         coefficients, transform, validate=None, carrier_probe=None):
    """Existing whole-row-selection ABI with a cellwise carrier matrix.

    The row mask means at least one cell uses a carrier. Coefficients for
    rejected cells remain raw, and their carrier is zero. Private fine-error
    diagnostics measure the complete hybrid row, in the production sum norm.
    """
    coefficients,cell_mask,info,omega=improve_cells(panel,times,kx,velocity,quadrature,
        coefficients,transform,validate=validate,carrier_probe=carrier_probe)
    return coefficients,cell_mask.any(dim=1,keepdim=True),info,omega
