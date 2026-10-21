"""Continuous OU macro response with cubic histories and electric feedback.

Two density Picard passes include -E_delta.grad_v f0; an optional analytic
Gaussian moment lift supplies the field acting on the evolving perturbation.
"""
from __future__ import annotations
import numpy as np
from torch.cuda import OutOfMemoryError as CUDAOutOfMemoryError
from pde_agent_verifier_solver.sdk import open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from base import Prepared as BrownianPrepared,FIELDS
from thermal_mixing import initial_thermal_covariance
from moments import INDICES,project,electric_modes,evaluate_grid,resample_modes,embed_modes,central_values,central_tangent
from macro_limiter import local_guard_grid,LocalPrepared
from kinetic_ops import HermiteTensor
from green_ops import available_workers,refine_source_history
from green_gpu import green_response,clear_cache
from spline_kernel import SplineOUKernel
from field_corrector import correct_prepared
from adaptive_taper import calibration_plan,select_taper
from response_budget import measure_response_reserves
from cuda_force_projector import CudaForceProjector
from same_grid_midpoints import retain_coarse_original_samples,optional_same_grid_refinement
from adaptive_initial import pilot_positions,select_initial_rule

NX=64
INITIAL_NV=64
INITIAL_HALF=18.
SOURCE_NV=128
SOURCE_HALF=24.
SOURCE_PILOT=64
SOURCE_RULES=((24,.70),(32,.70),(36,.70),(40,.65),(48,.70))
SOURCE_SPATIAL_COUNTS=tuple(range(48,15,-2))
SOURCE_TIME_COUNT=33
SOURCE_CALIBRATION_GRID=16
SOURCE_CALIBRATION_COUNT=64
SOURCE_REQUEST_LIMIT=65536
SOURCE_COST_MARGIN=1.30
SOURCE_DEGREE_TOLERANCE=.05
SOURCE_CENTRAL_TOLERANCE=.04
RETURN_RESERVE=50.
MAX_PREPARE_SECONDS=850.
EVOLUTION_RESERVE=140.
SPLINE_MODE_COST_24=.25
SPLINE_MODE_COST_32=.55
SPLINE_RESPONSE_OVERHEAD=25.
VELOCITY_DIFFERENCE_STEP=.005
PICARD_DENSITY_PASSES=2
SPATIAL_TAPER_FRACTION=.60
TAPER_MODE_THRESHOLD=1e-12
RETAIN_REPLAY=False
P5=tuple((degree-b,b) for degree in range(6) for b in range(degree+1))
P5_INDEX={pair:i for i,pair in enumerate(P5)}


def point_values(modes,xy,periods):
    n=modes.shape[-1]
    wave=[2*np.pi*np.fft.fftfreq(n,d=l/n) for l in periods]
    bx=np.exp(1j*np.asarray(xy)[:,0,None]*wave[0]);by=np.exp(1j*np.asarray(xy)[:,1,None]*wave[1])
    return np.einsum('cij,ni,nj->cn',modes,bx,by,optimize=True).real


def initial_operator_at_sites(raw_modes,xy,periods,center,nu,theta):
    n=raw_modes.shape[-1]
    kx,ky=np.meshgrid(*(2*np.pi*np.fft.fftfreq(n,d=l/n) for l in periods),indexing='ij')
    linear=np.zeros((15,n,n),complex)
    raw=point_values(raw_modes,xy,periods);field=point_values(electric_modes(raw_modes[0],periods),xy,periods)
    for row,(a,b) in enumerate(INDICES):
        linear[row]=(-1j*kx*(raw_modes[P5_INDEX[a+1,b]]+center[0]*raw_modes[row])
                     -1j*ky*(raw_modes[P5_INDEX[a,b+1]]+center[1]*raw_modes[row])-nu*(a+b)*raw_modes[row])
        if a>1:linear[row]+=nu*theta*a*(a-1)*raw_modes[P5_INDEX[a-2,b]]
        if b>1:linear[row]+=nu*theta*b*(b-1)*raw_modes[P5_INDEX[a,b-2]]
    result=point_values(linear,xy,periods)
    for row,(a,b) in enumerate(INDICES):
        if a:result[row]+=a*(field[0]-nu*center[0])*raw[P5_INDEX[a-1,b]]
        if b:result[row]+=b*(field[1]-nu*center[1])*raw[P5_INDEX[a,b-1]]
    return result


def baseline_from_initial(raw_modes,periods,interval,center,nu,theta):
    initial=evaluate_grid(raw_modes,NX)
    raw8=np.stack((initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],
                   .5*(initial[6]+initial[8]),.5*(initial[7]+initial[9])))
    covariance=initial_thermal_covariance(np.moveaxis(raw8,0,-1))
    modes=np.concatenate((project(raw8),electric_modes(raw_modes[0],periods)))
    return BrownianPrepared(modes,periods,interval,center,covariance,nu,theta,
                             project(central_values(initial,center)[-2:]))


def preview_baseline(arrays,periods,interval,center,nu,theta):
    """Finite startup handle from the public initial macro preview only.

The available initial kinetic energy sets isotropic central stress. Missing
or invalid energy uses public theta; the central heat closure is zero at the
preview nodes. It is only used if resolved initial acquisition is interrupted.
"""
    rho=np.asarray(arrays.get('rho0',[]),float)
    momentum=np.asarray(arrays.get('momentum0',[]),float)
    valid=(rho.ndim==2 and rho.shape[0]==rho.shape[1] and rho.size>0
           and momentum.shape==rho.shape+(2,) and np.isfinite(rho).all()
           and np.all(rho>0) and np.isfinite(momentum).all())
    if not valid:
        rho=np.ones((1,1));momentum=rho[...,None]*np.asarray(center)
    energy=np.asarray(arrays.get('kinetic_energy0',[]),float)
    if energy.shape==rho.shape and np.isfinite(energy).all():
        pressure=energy-.5*np.sum(momentum*momentum,axis=-1)/rho
        pressure=np.maximum(pressure,.01*float(theta)*rho)
    else:pressure=float(theta)*rho
    relative=momentum-rho[...,None]*np.asarray(center)
    ux=relative[...,0]/rho;uy=relative[...,1]/rho;speed=ux*ux+uy*uy
    raw=np.stack((rho,relative[...,0],relative[...,1],pressure+rho*ux*ux,
                  rho*ux*uy,pressure+rho*uy*uy,
                  ux*(.5*rho*speed+2*pressure),uy*(.5*rho*speed+2*pressure)))
    raw_modes=project(raw)
    modes=np.concatenate((raw_modes,electric_modes(raw_modes[0],periods)))
    covariance=np.eye(2)*(pressure.mean()/rho.mean())
    return BrownianPrepared(modes,periods,interval,center,covariance,nu,theta,
                            np.zeros((2,)+rho.shape,complex))


class FallbackPrepared:
    def __init__(self,baseline,diagnostics):
        self.baseline=baseline;self.diagnostics=diagnostics
    def evaluate(self,query):return self.baseline.evaluate(query)
    def close(self):self.baseline.close()


def source_validation(approximate,reference,raw,full_tangent,center,theta,duration):
    difference=approximate-reference;degree_errors=[]
    for degree in range(5):
        group=np.asarray([a+b==degree for a,b in INDICES])
        floor=.001*theta**(degree/2)/max(float(duration),.25)
        numerator=float(np.sqrt(np.mean(difference[group]**2)))
        denominator=max(float(np.sqrt(np.mean(reference[group]**2))),floor)
        degree_errors.append(numerator/denominator)
    transformed=central_tangent(raw,difference[:10],center);central_errors=[]
    for group,weight in (((0,),(1,)),((1,2),(1,1)),((3,4,5),(1,2,1)),((6,7),(1,1))):
        group=list(group);weight=np.asarray(weight)[:,None]
        numerator=float(np.sqrt(np.mean(weight*transformed[group]**2)))
        denominator=max(float(np.sqrt(np.mean(weight*full_tangent[group]**2))),.02)
        central_errors.append(numerator/denominator)
    return {'degree_scaled_errors':degree_errors,'central_tangent_relative_errors':central_errors,
            'accepted':max(degree_errors)<=SOURCE_DEGREE_TOLERANCE and max(central_errors)<=SOURCE_CENTRAL_TOLERANCE,
            'degree_tolerance':SOURCE_DEGREE_TOLERANCE,'central_tolerance':SOURCE_CENTRAL_TOLERANCE}


def prepare_solution(case,solve_budget=None):
    started=candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget,bool) or not isinstance(solve_budget,(int,float))
            or not np.isfinite(solve_budget) or solve_budget<=0):raise ValueError('finite positive shared solve budget required')
    budget=850. if solve_budget is None else float(solve_budget)
    deadline=started+min(MAX_PREPARE_SECONDS,budget-RETURN_RESERVE)
    evolution_reserve=EVOLUTION_RESERVE
    def check():
        if candidate_solve_clock()>=deadline:raise TimeoutError('kinetic macro return reserve')
    question=case.public_data['question']
    periods=np.asarray(question['grid']['x_domain'],float);interval=np.asarray(question['time_domain'],float)
    nu,theta=float(question['collision_nu']),float(question['temperature'])
    if (periods.shape!=(2,) or interval.shape!=(2,) or np.any(periods<=0) or interval[1]<=interval[0]
            or not np.isfinite(periods).all() or not np.isfinite(interval).all()):raise ValueError('valid public space/time domain required')
    limit=question.get('numerical_access',{}).get('max_points_per_request',262144)
    if type(limit) is not int or limit<=0:raise ValueError('positive public request limit required')
    limit=min(limit,262144);source_limit=min(limit,SOURCE_REQUEST_LIMIT)
    arrays=case.public_data.get('arrays',{})
    rp=np.asarray(arrays.get('rho0',[]),float);jp=np.asarray(arrays.get('momentum0',[]),float)
    center=np.zeros(2)
    if rp.ndim==2 and jp.shape==rp.shape+(2,) and np.isfinite(rp).all() and np.isfinite(jp).all() and rp.mean()>0:
        center=jp.mean(axis=(0,1))/rp.mean()
    startup=FallbackPrepared(preview_baseline(arrays,periods,interval,center,nu,theta),
                            {'initial_representation':'public preview isotropic pressure and zero nodal heat'})
    session=None
    timings={'source_points':0,'source_seconds':0.,'source_calls':0,'source_max_batch_rate':0.,
             'source_calibration_points':0,'source_calibration_seconds':0.}
    def source_rate(reserve=False):
        rate=max(timings['source_seconds']/max(timings['source_points'],1),
                 timings['source_calibration_seconds']/max(timings['source_calibration_points'],1))
        return max(rate,timings['source_max_batch_rate']) if reserve else rate
    def sample(sites,relative_velocity,method,calibration=False):
        sites=np.asarray(sites,float);velocity=np.asarray(relative_velocity,float).reshape(-1,2)
        request_limit=source_limit if method=='source' else limit
        block=max(1,request_limit//len(velocity));out=[]
        for offset in range(0,len(sites),block):
            check();cells=sites[offset:offset+block]
            points=np.column_stack((np.repeat(cells,len(velocity),axis=0),np.tile(velocity+center,(len(cells),1))))
            if method=='source':points=points.reshape(len(cells),len(velocity),-1).transpose(1,0,2).reshape(-1,points.shape[-1])
            pieces=[]
            for first in range(0,len(points),request_limit):
                check();batch=points[first:first+request_limit]
                if method=='source' and candidate_solve_clock()+SOURCE_COST_MARGIN*source_rate(True)*len(batch)+evolution_reserve>deadline:
                    raise TimeoutError('source acquisition would consume kinetic evolution reserve')
                before=candidate_solve_clock()
                try:answer=np.asarray(getattr(session,method)(batch),float)
                finally:
                    if method=='source':
                        duration=candidate_solve_clock()-before
                        timings['source_points']+=len(batch);timings['source_seconds']+=duration;timings['source_calls']+=1
                        timings['source_max_batch_rate']=max(timings['source_max_batch_rate'],duration/len(batch))
                        if calibration:
                            timings['source_calibration_points']+=len(batch);timings['source_calibration_seconds']+=duration
                if answer.shape!=(len(batch),) or not np.isfinite(answer).all():raise FloatingPointError('invalid public numerical response')
                pieces.append(answer)
            values=np.concatenate(pieces)
            if method=='source':values=values.reshape(len(velocity),len(cells)).T
            else:values=values.reshape(len(cells),len(velocity))
            out.append(values)
        check();return np.concatenate(out)
    try:
        check();session=open_public_case_session(case)
        xy=np.stack(np.meshgrid(*(np.arange(NX)*l/NX for l in periods),indexing='ij'),-1).reshape(-1,2)
        density=[]
        for first in range(0,len(xy),limit):
            check();density.append(np.asarray(session.rho0(xy[first:first+limit]),float))
        density=np.concatenate(density)
        if density.shape!=(len(xy),) or np.any(density<=0) or not np.isfinite(density).all():raise ValueError('positive initial density required')
        pilot_xy=pilot_positions(periods)
        pilot_density=[]
        for first in range(0,len(pilot_xy),limit):
            check();pilot_density.append(np.asarray(session.rho0(pilot_xy[first:first+limit]),float))
        pilot_density=np.concatenate(pilot_density)
        initial_nv,initial_half,initial_rule_diagnostics=select_initial_rule(
            pilot_xy,pilot_density,center,theta,sample,check=check)
        axis=(np.arange(initial_nv)-initial_nv//2)*2*initial_half/initial_nv;spacing=2*initial_half/initial_nv
        velocity=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        initial_values=sample(xy,velocity,'g0')
        weights=np.stack([velocity[:,0]**a*velocity[:,1]**b*spacing**2 for a,b in P5])
        initial=weights@initial_values.T
        if np.any(initial[0]<=0):raise ValueError('positive initial quadrature mass required')
        initial*=density[None]/initial[0];initial[0]=density
        initial=initial.reshape(21,NX,NX);raw_initial_modes=project(initial);del initial_values
        baseline=baseline_from_initial(raw_initial_modes,periods,interval,center,nu,theta)
        check()
    except (TimeoutError,ValueError,FloatingPointError,np.linalg.LinAlgError,CUDAOutOfMemoryError) as error:
        if isinstance(error,CUDAOutOfMemoryError):clear_cache(None)
        if 'baseline' in locals():
            startup=FallbackPrepared(baseline,{'initial_representation':'resolved initial moments'})
        startup.diagnostics.update(timings)
        startup.diagnostics['fallback_stage']='initial acquisition'
        startup.diagnostics['fallback_reason']=type(error).__name__+': '+str(error)
        startup.diagnostics['total_seconds']=candidate_solve_clock()-started
        return startup
    parallel_speedup=(1.,1.3,1.6,2.)[available_workers()-1]
    diagnostics={'initial_seconds':candidate_solve_clock()-started,'initial_nv':initial_nv,'initial_half':initial_half,
                 'initial_rule':initial_rule_diagnostics,'initial_nx':NX,
                 'green_parallel_workers':available_workers(),'green_parallel_mode_block':64,
                 'green_parallel_reserve_speedup':parallel_speedup}
    fallback=FallbackPrepared(baseline,diagnostics)
    try:
        row=np.arange(SOURCE_PILOT,dtype=float)+.5;fraction=row/SOURCE_PILOT;fraction[0]=0.;fraction[-1]=1.
        sites=np.column_stack((interval[0]+fraction*(interval[1]-interval[0]),
                               np.mod(row*np.sqrt(2.),1.)*periods[0],np.mod(row*np.sqrt(3.),1.)*periods[1]))
        axis=(np.arange(SOURCE_NV)-SOURCE_NV//2)*2*SOURCE_HALF/SOURCE_NV;spacing=2*SOURCE_HALF/SOURCE_NV
        uniform_velocity=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        pilot=sample(sites,uniform_velocity,'source')
        uniform_weights=np.stack([uniform_velocity[:,0]**a*uniform_velocity[:,1]**b*spacing**2 for a,b in INDICES])
        full_moments=uniform_weights@pilot.T
        pilot_raw=point_values(raw_initial_modes,sites[:,1:],periods)
        pilot_operator=initial_operator_at_sites(raw_initial_modes,sites[:,1:],periods,center,nu,theta)
        full_tangent=central_tangent(pilot_raw,pilot_operator[:10]+full_moments[:10],center)
        origin=pilot_raw[1:3].sum(axis=1)/pilot_raw[0].sum()
        selected=None;trials=[];pilot_rules={}
        for order,scale in SOURCE_RULES:
            check();operator=HermiteTensor(order,baseline.thermal_covariance,scale,origin,center,nu,theta)
            values=sample(sites,operator.relative_velocity,'source')
            flat=operator.relative_velocity.reshape(-1,2);integration=operator.integration.ravel()
            weights=np.stack([integration*flat[:,0]**a*flat[:,1]**b for a,b in INDICES])
            approximate=weights@values.T
            trial=source_validation(approximate,full_moments,pilot_raw,full_tangent,center,theta,interval[1]-interval[0])
            trial.update({'family':'common_hermite','order':order,'scale':scale,'nodes':order**2})
            trials.append(trial);pilot_rules[order]=(operator,values,approximate)
            if trial['accepted']:
                selected=(operator,trial)
                break
        diagnostics['source_rule_trials']=trials
        if selected is None:raise ValueError('all-time common velocity source validation failed')
        operator,trial=selected;diagnostics['source_rule']=trial;order=operator.order;node_count=order**2
        spline_kernel=SplineOUKernel(operator.z,operator.rotation*operator.sigma[None,:],
                                     operator.center,center,nu,theta,bc_type='natural',gh_blend=True)
        diagnostics['green_velocity_rule']='geometric Nyquist blend of common GH and natural cardinal cubic'
        evolution_reserve=EVOLUTION_RESERVE
        diagnostics['kinetic_relative_origin']=origin.tolist();diagnostics['kinetic_physical_center']=operator.center.tolist()
        diagnostics['kinetic_sigma']=operator.sigma.tolist();diagnostics['kinetic_rotation']=operator.rotation.tolist()
        calibration_sites,calibration_frames=calibration_plan(interval,periods,SOURCE_TIME_COUNT,SOURCE_CALIBRATION_COUNT)
        calibration_xy=calibration_sites[:,1:]
        calibration_values=sample(calibration_sites,operator.relative_velocity,'source',True)
        check();derivative_started=candidate_solve_clock()
        relative_velocity=operator.relative_velocity.reshape(-1,2)
        physical_velocity=relative_velocity+center
        integration=operator.integration.ravel()
        # Public g0 is a conditional velocity density. Keep its actual sampled
        # values here, rather than a quadrature normalization or Hermite patch.
        f0=sample(xy,relative_velocity,'g0')*density[:,None]
        f0_mass=f0@integration
        if np.any(f0_mass<=0):raise ValueError('positive Green initial quadrature mass required')
        diagnostics['green_initial_mass_rms']=float(np.sqrt(np.mean((f0_mass-density)**2)))
        dense_initial=evaluate_grid(raw_initial_modes,NX)
        moment_weights=np.stack([integration*relative_velocity[:,0]**a*relative_velocity[:,1]**b
                                  for a,b in INDICES[:10]])
        initial_difference=(moment_weights@f0.T).reshape(10,NX,NX)-dense_initial[:10]
        diagnostics['green_initial_macro_rms']=np.sqrt(np.mean(initial_difference**2,axis=(1,2))).tolist()
        h=VELOCITY_DIFFERENCE_STEP;velocity_gradient=[];velocity_laplacian=np.zeros_like(f0)
        for direction in range(2):
            shift=np.zeros(2);shift[direction]=h
            plus=sample(xy,relative_velocity+shift,'g0')*density[:,None]
            minus=sample(xy,relative_velocity-shift,'g0')*density[:,None]
            velocity_gradient.append((plus-minus)/(2*h))
            velocity_laplacian+=(plus-2*f0+minus)/(h*h)
        velocity_gradient=np.stack(velocity_gradient)
        f0_modes=project(f0.T.reshape(node_count,NX,NX))
        kx,ky=np.meshgrid(*(2*np.pi*np.fft.fftfreq(NX,d=l/NX) for l in periods),indexing='ij')
        wavevectors=np.stack((kx,ky),axis=-1).reshape(-1,2)
        spatial_gradient=np.stack([evaluate_grid(1j*k[None]*f0_modes,NX).reshape(node_count,-1).T
                                   for k in (kx,ky)])
        field0=evaluate_grid(electric_modes(raw_initial_modes[0],periods),NX).reshape(2,-1)
        fine_operator=-np.sum(spatial_gradient*physical_velocity.T[:,None,:],axis=0)
        fine_operator-=np.sum(velocity_gradient*field0[:,:,None],axis=0)
        fine_operator+=nu*(2*f0+np.sum(velocity_gradient*physical_velocity.T[:,None,:],axis=0)
                            +theta*velocity_laplacian)
        fine_operator_modes=project(fine_operator.T.reshape(node_count,NX,NX))
        gradient_modes=project(velocity_gradient.transpose(0,2,1).reshape(2,node_count,NX,NX))
        gradient_oversampled=evaluate_grid(gradient_modes,2*NX)
        diagnostics['green_velocity_difference_step']=h
        diagnostics['green_initial_derivative_seconds']=candidate_solve_clock()-derivative_started
        del plus,minus,velocity_laplacian,spatial_gradient,f0,velocity_gradient
        response_reserves,response_budget_diagnostics=measure_response_reserves(
            spline_kernel,np.linspace(*interval,2*SOURCE_TIME_COUNT-1),wavevectors,
            physical_velocity,integration,SOURCE_SPATIAL_COUNTS,NX,center,nu,theta,check=check,
            taper_fraction=SPATIAL_TAPER_FRACTION,reserve_floor=80.,cpu_floor=35.)
        diagnostics['green_response_cost_probe']=response_budget_diagnostics
        def estimate(mesh):return SOURCE_COST_MARGIN*source_rate()*SOURCE_TIME_COUNT*mesh*mesh*node_count
        def response_reserve(mesh):
            return response_reserves[mesh]
        diagnostics['source_mesh_estimated_seconds']={str(mesh):estimate(mesh) for mesh in SOURCE_SPATIAL_COUNTS}
        diagnostics['source_mesh_response_reserve']={str(mesh):response_reserve(mesh) for mesh in SOURCE_SPATIAL_COUNTS}
        n=next((mesh for mesh in SOURCE_SPATIAL_COUNTS if candidate_solve_clock()+estimate(mesh)+response_reserve(mesh)<deadline),None)
        if n is None:raise TimeoutError('remaining time cannot resolve common kinetic source frames')
        evolution_reserve=response_reserve(n)
        diagnostics['green_response_reserved_seconds']=evolution_reserve
        diagnostics['source_nx']=n;diagnostics['source_nt']=SOURCE_TIME_COUNT
        coarse_xy=np.stack(np.meshgrid(*(np.arange(n)*l/n for l in periods),indexing='ij'),-1).reshape(-1,2)
        times=np.linspace(*interval,SOURCE_TIME_COUNT)
        source_sites=np.column_stack((np.repeat(times,n*n),np.tile(coarse_xy,(len(times),1))))
        values=sample(source_sites,relative_velocity,'source').reshape(len(times),n,n,node_count)
        # Preserve raw S on a divisor subgrid before the in-place addition of
        # Lf0. The optional new-time correction compares raw S on this exact
        # same grid, so stationary spatial aliases cancel without an operator fit.
        coarse_original_source=None
        if n>=16 and n%2==0:
            try:
                coarse_original_source=retain_coarse_original_samples(values,n//2)
            except (ValueError,MemoryError,CUDAOutOfMemoryError) as error:
                diagnostics['green_same_grid_retention_failure']=type(error).__name__+': '+str(error)
        fine_on_coarse=point_values(fine_operator_modes,coarse_xy,periods).T.reshape(n,n,node_count)
        values+=fine_on_coarse[None]
        defect_modes=np.empty((len(times),NX*NX,node_count),complex)
        for frame in range(len(times)):
            check();coarse_modes=project(values[frame].transpose(2,0,1))
            fine_modes=embed_modes(coarse_modes,NX)
            defect_modes[frame]=fine_modes.reshape(node_count,-1).T
        del values
        # Record residual quadrature mass forcing. Final density's zero mode
        # is fixed to the initial mass independently of higher raw moments.
        mass_residual=defect_modes[:,0]@integration
        diagnostics['green_defect_global_mass_rms']=float(np.sqrt(np.mean(abs(mass_residual)**2)))
        frequency=np.abs(np.fft.fftfreq(NX)*NX)
        radius=np.maximum(frequency[:,None],frequency[None,:])
        calibration_defect=calibration_values+point_values(fine_operator_modes,calibration_xy,periods).T
        # Source-value fits do not control the oscillatory propagated heat
        # error. Keep a conservative fixed spatial bandwidth for all cases.
        taper_fraction=SPATIAL_TAPER_FRACTION
        taper_diagnostics={'selected_fraction':taper_fraction,
            'selection':'fixed conservative spatial bandwidth; no source-fit widening'}
        diagnostics['green_adaptive_taper']=taper_diagnostics
        taper=np.exp(-(radius/(taper_fraction*n/2))**8)
        active_modes=np.flatnonzero(taper.ravel()>=TAPER_MODE_THRESHOLD)
        diagnostics['green_active_modes']=int(len(active_modes))
        diagnostics['green_total_modes']=NX*NX
        diagnostics['green_mode_taper_threshold']=TAPER_MODE_THRESHOLD
        # Commit only a fully completed optional refinement. The helper keeps
        # the 33-frame source intact and returns ordinary natural refinement
        # after any optional acquisition, deadline, validation, or memory failure.
        refined_modes,refined_times,midpoint_diagnostics=optional_same_grid_refinement(
            defect_modes,times,coarse_original_source,source_nx=n,periods=periods,
            relative_velocity=relative_velocity,sample=sample,source_rate=source_rate,
            solve_clock=candidate_solve_clock,deadline=deadline,evolution_reserve=evolution_reserve,
            oom_errors=(CUDAOutOfMemoryError,),check=check)
        defect_modes,times=refined_modes,refined_times
        del refined_modes,refined_times,coarse_original_source
        diagnostics['green_same_grid_midpoints']=midpoint_diagnostics
        diagnostics['green_output_time_count']=len(times)
        diagnostics['green_source_time_interpolation']=('natural cubic with same-grid low-mode midpoint residual'
            if midpoint_diagnostics['status']=='applied' else 'natural cubic, exact midpoint knot insertion')
        diagnostics['green_max_hat_phase']=float(np.max(abs(wavevectors[active_modes]@physical_velocity.T))*(times[1]-times[0]))
        evolution_started=candidate_solve_clock()
        response_arguments=dict(source_times=times,velocity=physical_velocity,integration=integration,
                                wavevectors=wavevectors[active_modes],bulk_center=center,nu=nu,theta=theta,
                                check=check,kernel_model=spline_kernel)
        effective_modes=defect_modes
        picard_diagnostics=[]
        force_projector=CudaForceProjector(gradient_oversampled,NX,check=check)
        try:
            for iteration in range(PICARD_DENSITY_PASSES):
                check();density_response=green_response(effective_modes[:,active_modes],density_only=True,**response_arguments)
                density_modes=np.zeros((len(times),NX*NX),complex)
                density_modes[:,active_modes]=density_response[:,0]
                density_modes=density_modes.reshape(len(times),NX,NX)
                density_modes[:,0,0]=0.
                density_modes*=taper
                updated_modes=np.empty_like(defect_modes)
                field_norm=[]
                for frame in range(len(times)):
                    check();field=evaluate_grid(electric_modes(density_modes[frame],periods),2*NX)
                    field_norm.append(float(np.sqrt(np.mean(np.sum(field**2,axis=0)))))
                    force_modes=force_projector(field)
                    updated_modes[frame]=defect_modes[frame]+force_modes.reshape(node_count,-1).T
                picard_diagnostics.append({'iteration':iteration+1,'field_rms_by_time':field_norm})
                effective_modes=updated_modes
        finally:
            diagnostics['green_force_projector']=force_projector.diagnostics
            force_projector.close()
        corrections=green_response(effective_modes[:,active_modes],density_only=False,**response_arguments)
        diagnostics['green_hermitian_reduction']=getattr(spline_kernel,'_hermitian_reduction_diagnostics',{})
        clear_cache(spline_kernel)
        correction_modes=np.zeros((len(times),10,NX*NX),complex)
        correction_modes[:,:,active_modes]=corrections
        correction_modes=correction_modes.reshape(len(times),10,NX,NX)
        correction_modes[:,0,0,0]=0.
        correction_modes[...,NX//2,:]=0.;correction_modes[...,:,NX//2]=0.
        diagnostics['green_picard_passes']=picard_diagnostics
        diagnostics['green_evolution_seconds']=candidate_solve_clock()-evolution_started
        diagnostics['green_untapered_max_macro_mode']=np.max(abs(correction_modes),axis=(0,2,3)).tolist()
        correction_modes*=taper
        diagnostics['green_spatial_taper_fraction']=taper_fraction
        diagnostics['green_spatial_taper_nyquist']=n/2
        diagnostics['green_approximation']='free OU plus two density Picard passes; field on delta omitted'
        check()
        prior_modes=np.broadcast_to(raw_initial_modes[:10],correction_modes.shape)
        limited,weights=local_guard_grid(prior_modes,correction_modes,check,oversampling=4)
        diagnostics['local_guard_weight_mean']=weights.mean(axis=(1,2)).tolist()
        diagnostics['local_guard_weight_min']=weights.min(axis=(1,2)).tolist()
        diagnostics.update(timings);diagnostics['total_seconds']=candidate_solve_clock()-started
        prepared=LocalPrepared(baseline,times,limited,diagnostics,check)
        fallback=prepared
        def optional_check():
            if candidate_solve_clock()>=deadline-20.:
                raise TimeoutError('preserve completed Green handle before optional force deadline')
        prepared=correct_prepared(prepared,raw_initial_modes,density_modes,taper,nu,theta,check=optional_check,passes=3)
        fallback=prepared
        prepared.diagnostics['total_seconds']=candidate_solve_clock()-started
        if RETAIN_REPLAY:
            prepared.raw_initial_modes=raw_initial_modes;prepared.green_correction_modes=correction_modes
            prepared.source_pilot_sites=sites;prepared.source_pilot_full_moments=full_moments
        check();return prepared
    except (TimeoutError,ValueError,FloatingPointError,np.linalg.LinAlgError,CUDAOutOfMemoryError) as error:
        if 'spline_kernel' in locals() or isinstance(error,CUDAOutOfMemoryError):
            clear_cache(locals().get('spline_kernel'))
        diagnostics.update(timings);diagnostics['fallback_reason']=type(error).__name__+': '+str(error)
        diagnostics['total_seconds']=candidate_solve_clock()-started
        return fallback
