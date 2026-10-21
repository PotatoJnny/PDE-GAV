"""Reserve measured evolution cost after committing the source mesh.

This pure helper neither chooses a source mesh nor reads a clock. The caller
must retain its original deadline and reserve while choosing the source mesh.
All absolute times use the SDK solve clock.
"""
from __future__ import annotations
import math


def _finite_positive(value,name,*,allow_zero=False):
    if isinstance(value,bool):raise ValueError(name+' must be numeric')
    result=float(value)
    if not math.isfinite(result) or result<(0. if allow_zero else 1e-300):
        raise ValueError(name+' must be finite and '+('nonnegative' if allow_zero else 'positive'))
    return result


def _positive_integer(value,name):
    result=_finite_positive(value,name)
    integer=int(result)
    if integer!=result:raise ValueError(name+' must be an integer')
    return integer


def committed_mesh_response_budget(probe_diagnostics,*,started,solve_budget,
        original_deadline,original_evolution_reserve,active_mode_count,velocity_count,
        mesh_selected=False,spatial_n=64,output_time_count=65,
        optional_prepare_cap=850.,return_reserve=50.,reserve_floor=80.,
        kernel_margin=1.5,cpu_floor=25.,guard_reserve=15.):
    """Return optional deadline/reserve without changing the original schedule.

    The measured full-response probe is scaled to the actual active-mode
    count, not the larger prospective .9-taper count used for mesh selection.
    Existing measured CPU allowance is retained, or increased to the explicit
    grid/time/velocity floor. A separate guard allowance covers final local
    realization and construction of a completed handle. Optional analytic
    force correction still runs only after that handle exists.

    Missing or invalid probe evidence retains the original evolution reserve.
    No deadline or reserve changes until mesh_selected is exactly True.
    """
    started=float(started);original_deadline=float(original_deadline)
    if not math.isfinite(started) or not math.isfinite(original_deadline):
        raise ValueError('finite SDK-clock start and original deadline required')
    original_reserve=_finite_positive(original_evolution_reserve,'original evolution reserve')
    diagnostics={'status':'unchanged','mesh_selected':mesh_selected is True,
        'original_deadline':original_deadline,'original_evolution_reserve_seconds':original_reserve,
        'scope':'source mesh fixed by original schedule; measured reserve funds acquisition and completion'}
    if mesh_selected is not True:
        diagnostics['reason']='the spatial source mesh has not been selected'
        return original_deadline,original_reserve,diagnostics
    budget=_finite_positive(solve_budget,'remaining shared solve budget')
    cap=_finite_positive(optional_prepare_cap,'optional prepare cap')
    returning=_finite_positive(return_reserve,'return reserve',allow_zero=True)
    floor=_finite_positive(reserve_floor,'reserve floor')
    margin=_finite_positive(kernel_margin,'kernel margin')
    cpu_minimum=_finite_positive(cpu_floor,'CPU floor',allow_zero=True)
    guard=_finite_positive(guard_reserve,'guard reserve',allow_zero=True)
    active=_positive_integer(active_mode_count,'actual active-mode count')
    nv=_positive_integer(velocity_count,'velocity-node count')
    nx=_positive_integer(spatial_n,'spatial grid count')
    nt=_positive_integer(output_time_count,'output-time count')
    if active>nx*nx or nt<2:raise ValueError('consistent active-mode and output-time counts required')
    deadline=started+min(cap,budget-returning)
    diagnostics.update(status='mesh_committed',optional_deadline=deadline,
        optional_prepare_cap_seconds=cap,return_reserve_seconds=returning,
        actual_active_modes=active,velocity_nodes=nv,output_times=nt,
        spatial_n=nx,kernel_time_margin=margin,guard_reserve_seconds=guard,
        minimum_evolution_reserve_seconds=floor)
    try:
        elapsed=_finite_positive(probe_diagnostics['measured_probe_seconds'],'probe duration')
        block=_positive_integer(probe_diagnostics['probe_block']['mode_count'],'probe mode count')
        entries=_positive_integer(probe_diagnostics['probe_zero_source_entries'],'probe source entries')
        if block>nx*nx or entries%(block*nv):raise ValueError('probe axes do not match the selected velocity rule')
        probe_nt=entries//(block*nv)
        if probe_nt<2:raise ValueError('probe must include at least two output times')
        previous_cpu=_finite_positive(probe_diagnostics['cpu_reserve_seconds'],'existing CPU reserve',allow_zero=True)
        time_ratio=nt/probe_nt
        # Equal output grids have factor1. For a longer grid, include both
        # kernel length and modest FFT growth rather than linear scaling alone.
        time_factor=time_ratio*max(1.,math.log2(2*nt)/math.log2(2*probe_nt))
        blocks=math.ceil(active/block)
        kernel=margin*elapsed*blocks*time_factor
        cpu=max(previous_cpu*time_ratio,cpu_minimum*max(1.,nv/576.)*(nt/65.)*(nx/64.)**2)
        estimate=kernel+cpu+guard
        if not math.isfinite(estimate):raise ValueError('nonfinite extrapolated response cost')
        reserve=max(floor,estimate)
        diagnostics.update(reserve_status='measured',measured_probe_seconds=elapsed,
            probe_modes=block,probe_output_times=probe_nt,actual_mode_blocks=blocks,
            probe_time_scaling=time_factor,kernel_reserve_seconds=kernel,
            cpu_reserve_seconds=cpu,estimated_evolution_seconds=estimate,
            optional_evolution_reserve_seconds=reserve,
            limitations=['measured extrapolation is not a wall-time guarantee',
                'subsequent GPU/CPU contention can change cost',
                'optional analytic force work remains separately deadline-guarded'])
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        reserve=original_reserve
        diagnostics.update(reserve_status='fallback',optional_evolution_reserve_seconds=reserve,
            reason='invalid response probe: '+type(error).__name__+': '+str(error))
    return deadline,reserve,diagnostics
