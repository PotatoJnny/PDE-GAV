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


def prepare_source(case,center,deadline,nx=16,nt=33,balance=None):
    """Return prepared source frames, or None when validation/budget fails.

    ``balance.coarse(nx)`` must return cached arrays
    ``(initial_f, S0_fine, S0_coarse)`` on the same uniform spatial mesh and
    relative-velocity lattice used below.  The caller supplies an absolute
    SDK charged-clock deadline, including its desired evolution reserve.
    """
    q=case.public_data['question']
    periods=np.asarray(q['x_domain']);interval=np.asarray(q['time_domain'])
    center=np.asarray(center,dtype=float)
    session=open_public_case_session(case)
    nv=80;vmax=16.;v=(np.arange(nv)-nv//2)*2*vmax/nv
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
        values=np.asarray(session.source(points),dtype=float)
        if values.shape!=(len(points),) or not np.isfinite(values).all():raise ValueError('invalid source pilot')
        samples.append(values.reshape(len(xyts),nv,nv))
    samples=np.concatenate(samples)
    point_rate=(candidate_solve_clock()-pilot_start)/(len(sites)*nv*nv)
    options=[]
    for cap in (625,900,1225,1600):
        rule=fit(samples[:128],v,np.full(nv,2*vmax/nv),max_rank=48,max_nodes=cap,
                 degree=4,tail_power=2,normalize_rows=False,validation=samples[128:])
        errors=rule.diagnostics['heldout']['degree_relative_rms']
        merit=max(errors[str(d)] for d in range(5))
        options.append((merit,rule))
        if merit<.03:break
    # Preserve the validated temporal/velocity allocation of the parent.
    available=max(0.,deadline-candidate_solve_clock()-15.)
    best=None
    for merit,rule in options:
        for trial_nt in (33,):
            estimated=point_rate*nx*nx*trial_nt*rule.node_count*1.03
            if estimated<available:
                quality=merit+(.015 if trial_nt==33 else 0.)
                if best is None or quality<best[0]:best=(quality,rule,trial_nt)
    if best is None:return None
    _,rule,nt=best
    if max(rule.diagnostics['heldout']['degree_relative_rms'].values())>.12:
        return None
    if rule.diagnostics['heldout']['weighted_reconstruction_relative_rms']>.10:
        return None
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
    count=max(1,262144//len(nodes))
    gaussian=np.exp(-.5*(vv[:,0]**2+vv[:,1]**2)).reshape(nv,nv)
    gaussian/=gaussian.sum()*(2*vmax/nv)**2
    removed_mean_mass=[]
    for it in order:
        if candidate_solve_clock()+max(10.,1.6*last_cost)>deadline:break
        before=candidate_solve_clock();values=[]
        for ix in range(0,len(xy),count):
            cells=xy[ix:ix+count]
            points=np.column_stack((np.full(len(cells)*len(nodes),times[it]),
                    np.repeat(cells,len(nodes),axis=0),np.tile(nodes,(len(cells),1))))
            result=np.asarray(session.source(points),dtype=float)
            if result.shape!=(len(points),) or not np.isfinite(result).all():raise ValueError('invalid source')
            sampled_values=result.reshape(len(cells),len(rule.vx),len(rule.vy))
            if fine_nodes is not None:
                sampled_values=sampled_values-fine_nodes[ix:ix+len(cells)]
            values.append(rule.reconstruct(sampled_values).astype(np.float32))
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
    diagnostics['stationary_defect_balanced']=balance is not None
    diagnostics['residual_mean_mass_removed']=removed_mean_mass
    return times[used],array,v,diagnostics
