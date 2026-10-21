"""Validated original-source cubature with stationary-defect cancellation.

The original public source trains/selects the velocity basis.  Online values
are sampled on that same rule, their fine initial stationary source is
subtracted at exactly those nodes, and the reconstructed residual is added to
the exact coarse kinetic stationary source.  No solve is deferred to queries.
"""
import numpy as np
from scipy.stats import qmc
from pde_agent_verifier_solver.sdk import open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from tensor_cubature import fit
from base import NV,VMAX
from goal_tensor import fit_goal,validate_charge


# Source allocation can be compared by changing these two constants; the
# velocity validation and all physical evolution remain identical.
SOURCE_NX=16
SOURCE_NT=65


def choose_rule(samples,velocity,weights,nu,theta,*,max_wave=8.,horizon=1.,deadline=None):
    """Select the cheapest heldout-validated density-response interpolant.

    Transport observables do not use this kinetic correction, so degree3/4
    accuracy is diagnostic rather than an acceptance criterion. We do retain
    charge, current, full reconstruction, and phase-mixed charge checks.
    """
    train=samples[:128];heldout=samples[128:]
    records=[]
    for cap in (400,576,784,1024,1296,1600):
        if deadline is not None and candidate_solve_clock()>deadline-15:return None
        rules=[fit_goal(train,velocity,weights,count=int(np.sqrt(cap)),
                       nu=nu,theta=theta,max_wave=max_wave,horizon=horizon,validation=heldout)]
        for tail,over in ((0,0),(1,2),(2,0)):
            if deadline is not None and candidate_solve_clock()>deadline-15:return None
            rules.append(fit(train,velocity,weights,max_rank=48,max_nodes=cap,
                degree=4,tail_power=tail,oversample=over,normalize_rows=False,
                validation=heldout))
        accepted=[]
        for rule in rules:
            if deadline is not None and candidate_solve_clock()>deadline-15:return None
            response=validate_charge(rule,heldout,nu=nu,theta=theta,max_wave=max_wave,horizon=horizon)
            diag=rule.diagnostics['heldout'];degree=diag['degree_relative_rms']
            rule.diagnostics['heldout_charge_response']=response
            reconstruction=diag['weighted_reconstruction_relative_rms']
            merit=response['charge_response_relative_rms']
            records.append({'nodes':rule.node_count,'response':merit,
                'density':degree['0'],'current':degree['1'],
                'reconstruction':reconstruction})
            if (merit<=.055 and degree['0']<=.05 and degree['1']<=.35
                    and reconstruction<=.18):
                # Conditional covariance and POD can trade a little response
                # precision for substantial current precision. Current enters
                # the short-lag derivative of the charge response.
                quality=merit+.025*degree['1']+.02*reconstruction
                accepted.append((quality,rule))
        if accepted:
            smallest=min(item[1].node_count for item in accepted)
            economical=[item for item in accepted if item[1].node_count<=1.10*smallest]
            _,rule=min(economical,key=lambda item:item[0])
            rule.diagnostics['selection_records']=records
            return rule
    return None


def prepare_source(case,center,deadline,nx=None,nt=None,balance=None):
    """Return prepared source frames, or None when validation/budget fails.

    ``balance.coarse(nx)`` must return cached arrays
    ``(initial_f, S0_fine, S0_coarse)`` on the same uniform spatial mesh and
    relative-velocity lattice used below.  The caller supplies an absolute
    SDK charged-clock deadline, including its desired evolution reserve.
    """
    q=case.public_data['question']
    if nx is None:nx=SOURCE_NX
    if nt is None:nt=SOURCE_NT
    if nt not in (17,33,65) or nx not in (16,20,24):
        raise ValueError('source allocation must use supported nested time grid')
    periods=np.asarray(q['x_domain']);interval=np.asarray(q['time_domain'])
    center=np.asarray(center,dtype=float)
    session=open_public_case_session(case)
    request_limit=q.get('numerical_access',{}).get('max_points_per_request',262144)
    if type(request_limit) is not int or request_limit<=0:raise ValueError('positive public request limit required')
    request_limit=min(request_limit,262144)
    def source_values(points):
        values=[]
        for offset in range(0,len(points),request_limit):
            if candidate_solve_clock()>deadline-5:raise TimeoutError('source return reserve')
            part=points[offset:offset+request_limit]
            result=np.asarray(session.source(part),dtype=float)
            if result.shape!=(len(part),) or not np.isfinite(result).all():raise ValueError('invalid source values')
            values.append(result)
        return np.concatenate(values)
    nv=NV;vmax=VMAX;v=(np.arange(nv)-nv//2)*2*vmax/nv
    vv=np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)
    sites=qmc.Sobol(3,scramble=True,seed=7103).random_base2(8)[:192]
    sites[:,0]=interval[0]+sites[:,0]*np.ptp(interval)
    sites[:,1:]*=periods
    samples=[]
    pilot_start=candidate_solve_clock()
    for offset in range(0,len(sites),16):
        if candidate_solve_clock()>deadline-60:return None
        xyts=sites[offset:offset+16]
        points=np.column_stack((np.repeat(xyts,len(vv),axis=0),np.tile(vv+center,(len(xyts),1))))
        values=source_values(points)
        if values.shape!=(len(points),) or not np.isfinite(values).all():raise ValueError('invalid source pilot')
        samples.append(values.reshape(len(xyts),nv,nv))
    samples=np.concatenate(samples)
    point_rate=(candidate_solve_clock()-pilot_start)/(len(sites)*nv*nv)
    max_wave=max(8.,2*np.pi*(nx//2-1)/float(np.min(periods)))
    rule=choose_rule(samples,v,np.full(nv,2*vmax/nv),float(q['collision_nu']),float(q['temperature']),
                     max_wave=max_wave,horizon=float(np.ptp(interval)),deadline=deadline)
    if rule is None:return None
    # The pilot can overestimate online frame cost because its sparse-site
    # batches carry different fixed overhead. Require only a minimum17-frame
    # solve to be plausible; nested refinement then uses observed frame costs.
    available=max(0.,deadline-candidate_solve_clock()-15.)
    planned_nt=nt
    minimum_estimated=point_rate*nx*nx*17*rule.node_count*1.10
    if minimum_estimated>=available:return None
    # The basis is still trained on ORIGINAL source samples.  Subtract the
    # fine defect only after selecting this rule, at its exact pivot indices.
    fine_nodes=None;coarse_source=None
    if balance is not None:
        if candidate_solve_clock()>deadline-15:return None
        initial_f,fine_source,coarse_source=balance.coarse(nx)
        expected=(nx,nx,nv,nv)
        if any(np.asarray(a).shape!=expected or not np.isfinite(a).all()
               for a in (initial_f,fine_source,coarse_source)):
            raise ValueError('invalid cached stationary-defect arrays')
        if hasattr(balance,'velocity') and not np.allclose(np.asarray(balance.velocity),v,rtol=0.,atol=1e-12):
            raise ValueError('stationary-defect velocity grid differs from source grid')
        if hasattr(balance,'center') and not np.allclose(np.asarray(balance.center),center,rtol=0.,atol=1e-12):
            raise ValueError('stationary-defect velocity center differs from source center')
        ix=np.asarray(rule.index_x,dtype=int);iy=np.asarray(rule.index_y,dtype=int)
        if (not np.array_equal(rule.vx,v[ix]) or not np.array_equal(rule.vy,v[iy])):
            raise ValueError('cubature node order differs from its velocity indices')
        fine=np.asarray(fine_source).reshape(nx*nx,nv,nv)
        fine_nodes=np.take(np.take(fine,ix,axis=-2),iy,axis=-1)
        coarse_source=np.asarray(coarse_source,dtype=np.float32)
        if candidate_solve_clock()>deadline-15:return None
    vx,vy=np.meshgrid(rule.vx,rule.vy,indexing='ij')
    nodes=np.column_stack((vx.ravel(),vy.ravel()))+center
    xy=np.stack(np.meshgrid(*[np.arange(nx)*l/nx for l in periods],indexing='ij'),-1).reshape(-1,2)
    times=np.linspace(*interval,nt)
    stride=(nt-1)//8
    order=list(range(0,nt,stride))
    for step in [stride//2,stride//4,stride//8]:
        if step>=1:order.extend(range(step,nt,2*step))
    source={};last_cost=point_rate*len(xy)*len(nodes)
    count=max(1,request_limit//len(nodes))
    gaussian=np.exp(-.5*(vv[:,0]**2+vv[:,1]**2)).reshape(nv,nv)
    gaussian/=gaussian.sum()*(2*vmax/nv)**2
    removed_mean_mass=[];online_frame_timeout=False
    for it in order:
        if candidate_solve_clock()+max(10.,1.6*last_cost)>deadline:break
        before=candidate_solve_clock();values=[]
        try:
            for ix in range(0,len(xy),count):
                cells=xy[ix:ix+count]
                points=np.column_stack((np.full(len(cells)*len(nodes),times[it]),
                        np.repeat(cells,len(nodes),axis=0),np.tile(nodes,(len(cells),1))))
                result=source_values(points)
                if result.shape!=(len(points),) or not np.isfinite(result).all():raise ValueError('invalid source')
                sampled_values=result.reshape(len(cells),len(rule.vx),len(rule.vy))
                if fine_nodes is not None:
                    sampled_values=sampled_values-fine_nodes[ix:ix+len(cells)]
                values.append(rule.reconstruct(sampled_values).astype(np.float32))
        except TimeoutError:
            # A frame enters ``source`` only after every spatial batch is
            # complete. Keep earlier frames and discard this partial frame.
            online_frame_timeout=True;values=[]
            break
        residual=np.concatenate(values).reshape(nx,nx,nv,nv)
        # Correct residual mean mass before adding the exact coarse source.
        # Thus a zero residual returns S0_coarse without changing its shape.
        mass_mean=float(residual.sum(dtype=np.float64)*(2*vmax/nv)**2/(nx*nx))
        residual-=mass_mean*gaussian
        removed_mean_mass.append(mass_mean)
        if coarse_source is not None:
            residual+=coarse_source
        source[it]=residual
        last_cost=candidate_solve_clock()-before
    used=[]
    for step in [stride,stride//2,stride//4,stride//8]:
        if step>=1:
            trial=list(range(0,nt,step))
            if all(i in source for i in trial):used=trial
    if len(used)<17:return None
    used=sorted(source)
    array=np.asarray([source[i] for i in used],dtype=np.float32)
    diagnostics=dict(rule.diagnostics)
    diagnostics['source_space_count']=nx
    diagnostics['planned_source_time_count']=planned_nt
    diagnostics['target_source_time_count']=nt
    diagnostics['minimum_source_estimated_seconds']=minimum_estimated
    diagnostics['completed_source_time_count']=len(used)
    diagnostics['online_frame_timeout']=online_frame_timeout
    diagnostics['pilot_seconds_per_point']=point_rate
    diagnostics['stationary_defect_balanced']=balance is not None
    diagnostics['residual_mean_mass_removed']=removed_mean_mass
    return times[used],array,v,diagnostics
