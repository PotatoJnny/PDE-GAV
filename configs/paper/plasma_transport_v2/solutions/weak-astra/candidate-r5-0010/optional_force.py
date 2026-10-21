"""Atomic optional force refinement of a fully prepared macro representation."""
from __future__ import annotations
import numpy as np
from gaussian_force import GaussianLift,nonlinear_force_correction
from moments import project,resample_modes,evaluate_grid
from macro_limiter import LocalPrepared,pointwise_weights


def refine_prepared_force(prepared,raw_initial_modes,used_density_modes,origin_relative,
                          nu,theta,taper,check=lambda:None,*,iterations=3):
    """Return a new prepared object, or the exact completed input on failure.

    The closure sees the actual guarded baseline moments. Its increment is
    guarded against that completed state, preserving the baseline atomically
    if the optional SDK-clock check, numerics or representation build fails.
    """
    diagnostics=dict(prepared.diagnostics)
    try:
        check();n=np.asarray(raw_initial_modes).shape[-1]
        baseline=prepared.baseline
        base_modes=resample_modes(project(prepared.raw_grid[:,:10]),n)
        base_delta=base_modes-raw_initial_modes[None,:10]
        lift=GaussianLift(baseline.thermal_covariance,origin_relative,
                          baseline.velocity_center,nu,theta)
        updated,iteration_diagnostics=nonlinear_force_correction(
            lift,raw_initial_modes,base_delta,used_density_modes,
            prepared.times,baseline.periods,taper,check,iterations=iterations)
        extra=updated-base_delta
        corrected=np.empty_like(prepared.raw_grid);weights=[]
        for frame in range(len(corrected)):
            check();base=prepared.raw_grid[frame]
            increment=evaluate_grid(extra[frame],base.shape[-1])
            # Initial limiting can scale its nominal floors slightly when
            # restoring mass. Preserve the actual completed anchor's floors.
            rho=base[0]
            px=base[3]-base[1]**2/rho;py=base[5]-base[2]**2/rho
            pxy=base[4]-base[1]*base[2]/rho
            mineigen=.5*(px+py-np.sqrt((px-py)**2+4*pxy**2))
            density_floor=min(.15,.99*float(rho.min()))
            pressure_floor=min(.01,.99*float(mineigen.min()))
            weight=pointwise_weights(base,increment,density_floor=density_floor,
                                     pressure_floor=pressure_floor)
            corrected[frame]=base+weight[None]*increment
            corrected[frame]*=base[0].mean()/corrected[frame,0].mean()
            weights.append([float(weight.mean()),float(weight.min())])
        diagnostics['nonlinear_force_refinement']={
            'status':'completed','iterations':iterations,
            'closure':'degree3 Gaussian lift at physical measured thermal covariance',
            'coherent_density_feedback':True,
            'iteration_diagnostics':iteration_diagnostics,
            'guard_weight_mean_min_by_time':weights}
        check();result=LocalPrepared(baseline,prepared.times,corrected,diagnostics,check)
        check();return result
    except (TimeoutError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        diagnostics['nonlinear_force_refinement']={
            'status':'retained_completed_baseline','reason':type(error).__name__+': '+str(error)}
        prepared.diagnostics=diagnostics
        return prepared
