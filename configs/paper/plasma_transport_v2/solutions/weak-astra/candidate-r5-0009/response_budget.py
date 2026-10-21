"""Per-case bounded response-cost probe, independent of scoring coordinates.

The default caller executes one full Green response on at most64 solver modes
and zero source. This measures the real selected velocity rule, OU parameters,
time quadrature, GPU transfer, and kernel construction. It makes no Oracle call.
"""
from __future__ import annotations
import math
import numpy as np
from green_ops import ou_parameters
from green_gpu import green_response as default_response,clear_cache as default_clear
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


def prospective_active_modes(spatial_n,source_mesh,taper_fraction=.9,threshold=1e-12):
    frequency=np.abs(np.fft.fftfreq(int(spatial_n))*int(spatial_n))
    radius=np.maximum(frequency[:,None],frequency[None,:])
    taper=np.exp(-(radius/(float(taper_fraction)*int(source_mesh)/2))**8)
    return np.flatnonzero(taper.ravel()>=float(threshold))


def _temporal_order(beta):
    return 4 if beta<=4 else 8 if beta<=12 else 12 if beta<=20 else int(np.ceil(beta/2))+2


def choose_probe_block(model,evolution_times,wavevectors,physical_velocity,active_modes,mode_block=64):
    """Choose a worst-cost contiguous block in the actual active-mode ordering."""
    times=np.asarray(evolution_times,float);k=np.asarray(wavevectors,float);v=np.asarray(physical_velocity,float)
    active=np.asarray(active_modes,int);step=float(times[1]-times[0])
    # Largest lag used by a half-hat kernel is just beyond the final output.
    Amax=float(ou_parameters(np.array([times[-1]-times[0]+step]),model.nu,model.theta)[1][0])
    zspacing=float(np.max(np.diff(model.z)));order=int(len(model.z));blocks=[]
    for start in range(0,len(active),int(mode_block)):
        indices=active[start:start+int(mode_block)];kb=k[indices]
        beta=float(np.max(abs(kb@v.T)))*step
        temporal_order=_temporal_order(beta)
        rotated=kb@model.transform
        velocity_orders=[max(model.quadrature_order,int(np.ceil(Amax*np.max(abs(rotated[:,axis]))*zspacing/2))+4)
                         for axis in range(2)]
        # Tensor moment products dominate at high Q. The second term covers
        # per-cell oscillatory cardinal quadrature and breaks phase ties.
        relative_work=temporal_order*(108*order**2+16*order*sum(velocity_orders))
        relative_work*=len(indices)/min(int(mode_block),len(active))
        blocks.append((relative_work,indices,{'first_active_index':start,'mode_count':len(indices),
            'max_hat_phase':beta,'temporal_gauss_order':temporal_order,
            'velocity_cell_orders':velocity_orders,'relative_work_indicator':relative_work}))
    if not blocks:raise ValueError('nonempty prospective mode set required')
    _,indices,diagnostics=max(blocks,key=lambda block:block[0])
    return indices,diagnostics


def measure_response_reserves(model,evolution_times,wavevectors,physical_velocity,integration,
                              source_meshes,spatial_n,bulk_center,nu,theta,
                              *,check=lambda:None,response=None,clear=None,clock=candidate_solve_clock,
                              mode_block=64,taper_fraction=.9,reserve_floor=140.,
                              kernel_margin=1.3,cpu_floor=25.):
    """Measure one worst block and estimate conservative core reserves.

    `evolution_times` is the final65-output grid, not the33 acquired source
    times. Optional analytic force passes are excluded and remain separately
    budgeted. The probe is included in the same SDK charged case clock.

    Dependencies are injectable for CPU control-flow tests. Production callers
    can omit them; defaults are the current CUDA response and cache clearer.
    Cache clearing is attempted even when the probe fails or is interrupted.
    """
    if response is None or clear is None:
        if response is None:response=default_response
        if clear is None:clear=default_clear
    check();times=np.asarray(evolution_times,float);wave=np.asarray(wavevectors,float)
    velocity=np.asarray(physical_velocity,float);integration=np.asarray(integration,float)
    meshes=tuple(map(int,source_meshes));nv=len(velocity)
    if (not meshes or times.ndim!=1 or len(times)<2 or times[1]<=times[0]
            or not np.allclose(np.diff(times),times[1]-times[0]) or wave.shape!=(int(spatial_n)**2,2)
            or velocity.shape!=(nv,2) or integration.shape!=(nv,) or mode_block<1):
        raise ValueError('uniform evolution times, source meshes, and consistent solver axes required')
    modes={mesh:prospective_active_modes(spatial_n,mesh,taper_fraction) for mesh in meshes}
    largest=max(meshes,key=lambda mesh:len(modes[mesh]))
    indices,block_diagnostics=choose_probe_block(model,times,wave,velocity,modes[largest],mode_block)
    zero_source=np.zeros((len(times),len(indices),nv),complex)
    before=clock()
    try:
        answer=response(zero_source,source_times=times,velocity=velocity,integration=integration,
            wavevectors=wave[indices],bulk_center=bulk_center,nu=nu,theta=theta,
            density_only=False,mode_block=mode_block,check=check,kernel_model=model)
        if answer.shape!=(len(times),10,len(indices)) or not np.isfinite(answer).all():
            raise FloatingPointError('invalid cost probe response')
        check()
    finally:
        clear(model)
    elapsed=float(clock()-before)
    if elapsed<0 or not np.isfinite(elapsed):raise ValueError('valid SDK probe duration required')
    # Two Picard products/FFTs scale with the velocity axis even though only
    # the density is returned. Keep that CPU overhead separate from GPU work.
    cpu_reserve=float(cpu_floor)*max(1.,nv/576.)*(len(times)/65.)*(int(spatial_n)/64.)**2
    reserves={mesh:max(float(reserve_floor),float(kernel_margin)*elapsed*math.ceil(len(modes[mesh])/len(indices))+cpu_reserve)
              for mesh in meshes}
    return reserves,{'measured_probe_seconds':elapsed,'probe_block':block_diagnostics,
             'probe_zero_source_entries':int(zero_source.size),'prospective_taper_fraction':float(taper_fraction),
             'prospective_active_modes':{str(mesh):len(modes[mesh]) for mesh in meshes},
             'kernel_time_margin':float(kernel_margin),'cpu_reserve_seconds':cpu_reserve,
             'core_reserve_seconds':{str(mesh):reserve for mesh,reserve in reserves.items()},
             'limitations':['worst-block extrapolation is a conservative estimate, not a wall-time guarantee',
                 'GPU contention can change after the probe','optional force correction remains separately budgeted']}
