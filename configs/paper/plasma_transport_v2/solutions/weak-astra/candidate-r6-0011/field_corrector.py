"""Optional macro-only correction for electric force acting on delta.

Uses no numerical Oracle and returns the original prepared handle on a
cooperative numerical/deadline rejection. Caller supplies the same SDK check
as preparation, or an earlier optional deadline which reserves serialization.
"""
from __future__ import annotations
import numpy as np
from gaussian_green import GaussianHermiteOUKernel
from green_ops import green_response
from moments import project,evaluate_grid,resample_modes,electric_modes
from macro_limiter import LocalPrepared,pointwise_weights


def correct_prepared(prepared,raw_initial_modes,used_density_modes,taper,nu,theta,
                     *,check=lambda:None,passes=3,max_workers=4):
    """Complete Picard feedback with an analytic degree-three velocity lift.

`used_density_modes` is the *last intermediate* density passed to the force
on f0 in the incumbent, before its full final macro convolution. It includes
the incumbent spatial taper. `taper` is the same NX-by-NX array used for the
base response. The current field is rebuilt from the current limited macros.

The additional forcing in each pass is
 -E_current.grad(delta_lift) -(E_delta,current-E_delta,used).grad(f0_lift).
Its OU response is added to the fixed completed base response. Thus both
force terms use one coherent current field, without doubling the existing
force on f0. The physical initial thermal covariance sets the lift width.
"""
    try:
        check()
        baseline=prepared.baseline;times=prepared.times
        n=raw_initial_modes.shape[-1];fine=prepared.raw_grid.shape[-1]
        prior=evaluate_grid(raw_initial_modes[:10],fine)
        fixed_base=np.asarray(prepared.raw_grid).copy();current=fixed_base.copy()
        relative_origin=np.real(raw_initial_modes[1:3,0,0]/raw_initial_modes[0,0,0])
        model=GaussianHermiteOUKernel(baseline.thermal_covariance,
                 baseline.velocity_center+relative_origin,baseline.velocity_center,nu,theta)
        initial_lift=model.lift(prior)
        wave=np.stack(np.meshgrid(*(2*np.pi*np.fft.fftfreq(n,d=l/n)
                     for l in baseline.periods),indexing='ij'),axis=-1).reshape(-1,2)
        used_field=np.stack([evaluate_grid(electric_modes(d,baseline.periods),fine)
                             for d in used_density_modes])
        active=np.flatnonzero(np.asarray(taper).ravel()>=1e-12)
        args=dict(source_times=times,velocity=model.phase_probes,integration=np.ones(15),
                  wavevectors=wave[active],bulk_center=baseline.velocity_center,nu=nu,theta=theta,
                  check=check,kernel_model=model,mode_block=64,max_workers=max_workers)
        records=[]
        for iteration in range(int(passes)):
            check();source=np.empty((len(times),n*n,15),complex)
            for frame in range(len(times)):
                check()
                delta=current[frame]-prior
                total_field=evaluate_grid(electric_modes(project(current[frame,0]),baseline.periods),fine)
                delta_field=evaluate_grid(electric_modes(project(delta[0]),baseline.periods),fine)
                force=model.force(model.lift(delta),total_field)
                force+=model.force(initial_lift,delta_field-used_field[frame])
                source[frame]=resample_modes(project(force),n).reshape(15,-1).T
            response=green_response(source[:,active],density_only=False,**args)
            modes=np.zeros((len(times),10,n*n),complex);modes[:,:,active]=response
            modes=modes.reshape(len(times),10,n,n)*taper
            modes[:,0,0,0]=0.;modes[...,n//2,:]=0.;modes[...,:,n//2]=0.
            updated=np.empty_like(current);weights=[]
            for frame in range(len(times)):
                check();correction=evaluate_grid(modes[frame],fine)
                total_delta=fixed_base[frame]+correction-prior
                weight=pointwise_weights(prior,total_delta)
                updated[frame]=prior+weight[None]*total_delta
                updated[frame]*=prior[0].mean()/updated[frame,0].mean()
                weights.append(float(weight.min()))
            records.append({'iteration':iteration+1,
                            'raw_step_rms':np.sqrt(np.mean((updated-current)**2,axis=(0,2,3))).tolist(),
                            'raw_correction_rms':np.sqrt(np.mean((updated-fixed_base)**2,axis=(0,2,3))).tolist(),
                            'minimum_guard_weight':float(min(weights))})
            current=updated
        diagnostics=dict(prepared.diagnostics)
        diagnostics['analytic_force_correction']={'passes':int(passes),'lift_degree':3,
                 'source_degree':4,'physical_covariance':model.covariance.tolist(),
                 'records':records,'method':'analytic Gaussian-Hermite OU response, coherent force field'}
        answer=LocalPrepared(baseline,times,current,diagnostics,check)
        check();return answer
    except (TimeoutError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        prepared.diagnostics['analytic_force_correction_rejected']=type(error).__name__+': '+str(error)
        return prepared
