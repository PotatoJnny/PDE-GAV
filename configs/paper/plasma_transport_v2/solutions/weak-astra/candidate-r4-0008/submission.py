"""Locally adapted degree4 GH source moments and prepared D4 Hermite response.

Every numerical source site belongs to the solver-owned preparation mesh.
The later POINTS query only projects prepared raw-moment Fourier snapshots.
"""
from __future__ import annotations
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerQuery,AnswerView,QueryKind,open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from base import Prepared as BrownianPrepared,FIELDS
from thermal_mixing import initial_thermal_covariance
from moments import INDICES,INDEX,project,electric_modes,evaluate_grid,central_values,central_tangent
from adaptive_velocity import local_hermite
from hermite_response_h4 import brownian_prior5 as brownian_prior,moment_rhs,evolve
from realizability import gram
from initial_quadrature import plan_initial_quadrature,integrate_initial,initial_error_metrics
from convex_guard import guard_corrections as convex_guard_corrections
from initial_acquisition import local_initial as acquire_local_initial,repair_initial_mass


NX=32
INITIAL_NV=96
INITIAL_HALF=18.
INITIAL_SEED_NX=16
INITIAL_VALIDATION_COUNT=64
INITIAL_GH_RULES=((32,.65),(40,.65),(48,.65),(64,.60))
SOURCE_NV=128
SOURCE_HALF=24.
SOURCE_PILOT=64
SOURCE_SPATIAL_COUNTS=(32,30,28,26,24,22,20,18,16)
SOURCE_COUNTS=(33,65)
SOURCE_PLANNING_COUNT=65
SOURCE_GH_RULES=((24,.70),(28,.70),(32,.65),(40,.65))
SOURCE_CENTRAL_TOLERANCE=.04
SOURCE_DEGREE_TOLERANCE=.05
SOURCE_GH24_CENTRAL_TOLERANCE=.06
SOURCE_GH24_DEGREE_TOLERANCE=.08
SOURCE_ABSOLUTE_FLOOR=.001
SOURCE_COST_MARGIN=1.15
SOURCE_REQUEST_LIMIT=65536
SOURCE_CALIBRATION_GRID=16
SOURCE_CALIBRATION_COUNT=64
SOURCE_CALIBRATION_FRAMES=12
SOURCE_FINALIZE_RESERVE=35.
RETURN_RESERVE=50.
MAX_PREPARE_SECONDS=800.
RELAXATION=1.
CLOSURE_RELAXATION=1.
TAPER_FRACTION=.6
FIELD_FEEDBACK=True
BASIS_FROM_INITIAL_COVARIANCE=True
RETAIN_REPLAY=True
RETAIN_HIGHER_REPLAY=True
DEGREE5_INDICES=tuple((degree-b,b) for degree in range(6) for b in range(degree+1))
DEGREE5_INDEX={pair:i for i,pair in enumerate(DEGREE5_INDICES)}


def source_cost_rate(timings,reserve=False):
    """Measured charged seconds per source point, never an Oracle wait clock.

    The uniform-velocity pilot has a different point layout from production.
    A mid-time local-node calibration therefore supplies an independent floor
    for mesh planning. Request-level reserves also respect the slowest batch.
    """
    if timings.get('source_local_points',0):
        weighted=timings['source_local_seconds']/timings['source_local_points']
    else:weighted=timings['source_seconds']/max(timings['source_points'],1)
    calibration=timings['source_calibration_seconds']/max(timings['source_calibration_points'],1)
    rate=max(weighted,calibration)
    recent=timings.get('source_recent_local',[])
    if len(recent)>=12:
        recent_rate=sum(item[1] for item in recent)/max(sum(item[0] for item in recent),1)
        rate=max(recent_rate,.5*weighted)
    if reserve:
        maximum=(timings['source_local_max_batch_rate'] if timings.get('source_local_points',0)
                 else timings['source_max_batch_rate'])
        rate=max(rate,maximum)
    return rate


def record_source_cost(timings,points,duration,calibration=False,local=False):
    """Account every attempted source call, including calibration/refinement."""
    duration=max(float(duration),0.)
    if points<=0:raise ValueError('positive source request point count required')
    timings['source_points']+=int(points)
    timings['source_seconds']+=duration
    timings['source_calls']+=1
    timings['source_max_batch_rate']=max(timings['source_max_batch_rate'],duration/points)
    timings['source_max_batch_seconds']=max(timings['source_max_batch_seconds'],duration)
    timings['source_max_batch_points']=max(timings['source_max_batch_points'],int(points))
    if local:
        timings['source_local_points']=timings.get('source_local_points',0)+int(points)
        timings['source_local_seconds']=timings.get('source_local_seconds',0.)+duration
        timings['source_local_calls']=timings.get('source_local_calls',0)+1
        timings['source_local_max_batch_rate']=max(timings.get('source_local_max_batch_rate',0.),duration/points)
        recent=timings.setdefault('source_recent_local',[])
        recent.append((int(points),duration))
        del recent[:-16]
    if calibration:
        timings['source_calibration_points']+=int(points)
        timings['source_calibration_seconds']+=duration
        timings['source_calibration_calls']+=1


def higher_source_moments(values,velocity,integration,indices=None):
    """Replay-only degree4 integrals from already acquired local source values."""
    if indices is None:indices=np.arange(len(values))
    answer=np.empty((len(values),5))
    for offset in range(0,len(values),256):
        stop=min(offset+256,len(values));ids=indices[offset:stop]
        local=velocity[ids];weighted=values[offset:stop]*integration[ids]
        for m,(a,b) in enumerate(INDICES[10:]):
            answer[offset:stop,m]=np.sum(weighted*local[:,:,0]**a*local[:,:,1]**b,axis=1)
    return answer


def point_values(modes,xy,periods):
    n=modes.shape[-1]
    wave=[2*np.pi*np.fft.fftfreq(n,d=l/n) for l in periods]
    bx=np.exp(1j*np.asarray(xy)[:,0,None]*wave[0]);by=np.exp(1j*np.asarray(xy)[:,1,None]*wave[1])
    return np.einsum('cij,ni,nj->cn',modes,bx,by).real


def resize_modes(modes,n):
    old=modes.shape[-1];frequencies=np.rint(np.fft.fftfreq(old)*old).astype(int)
    if n<old:raise ValueError('resize only prolongs retained Fourier modes')
    out=np.zeros(modes.shape[:-2]+(n,n),complex)
    out[...,frequencies[:,None]%n,frequencies[None,:]%n]=modes
    return out


def initial_operator_at_sites(raw_modes,xy,periods,center,nu,theta):
    """Evaluate the fine initial moment operator before coarse-grid aliasing.

    Force products are formed at the requested solver sites, retaining products
    above the fine Fourier cutoff instead of truncating them before sampling.
    """
    n=raw_modes.shape[-1]
    kx,ky=np.meshgrid(*(2*np.pi*np.fft.fftfreq(n,d=l/n) for l in periods),indexing='ij')
    linear=np.zeros((15,n,n),complex)
    raw=point_values(raw_modes,xy,periods);field=point_values(electric_modes(raw_modes[0],periods),xy,periods)
    for row,(a,b) in enumerate(INDICES):
        linear[row]=(-1j*kx*(raw_modes[DEGREE5_INDEX[a+1,b]]+center[0]*raw_modes[row])
                     -1j*ky*(raw_modes[DEGREE5_INDEX[a,b+1]]+center[1]*raw_modes[row])-nu*(a+b)*raw_modes[row])
        if a>1:linear[row]+=nu*theta*a*(a-1)*raw_modes[INDEX[a-2,b]]
        if b>1:linear[row]+=nu*theta*b*(b-1)*raw_modes[INDEX[a,b-2]]
    result=point_values(linear,xy,periods)
    for row,(a,b) in enumerate(INDICES):
        if a:result[row]+=a*(field[0]-nu*center[0])*raw[INDEX[a-1,b]]
        if b:result[row]+=b*(field[1]-nu*center[1])*raw[INDEX[a,b-1]]
    return result


def baseline_from_initial(raw_modes,periods,interval,center,nu,theta):
    initial=evaluate_grid(raw_modes,NX)
    raw8=np.stack((initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],
                   .5*(initial[6]+initial[8]),.5*(initial[7]+initial[9])))
    covariance=initial_thermal_covariance(np.moveaxis(raw8,0,-1))
    heat_modes=project(central_values(initial,center)[-2:])
    modes=np.concatenate((project(raw8),electric_modes(raw_modes[0],periods)))
    return BrownianPrepared(modes,periods,interval,center,covariance,nu,theta,heat_modes)


def guard_corrections(prior_modes,correction_modes,check):
    """One whole-state coefficient per prepared snapshot, checked at 2x mesh."""
    modes=prior_modes+correction_modes;weights=[];n=modes.shape[-1]
    for it in range(len(modes)):
        check();base=evaluate_grid(prior_modes[it],2*n);delta=evaluate_grid(correction_modes[it],2*n)
        anchor_psd=np.min(np.linalg.eigvalsh(gram(base,True)))>=-1e-10
        amount=1.
        for _ in range(18):
            trial=base+amount*delta;r=trial[0];safe=np.maximum(r,1e-12)
            pxx=trial[3]-trial[1]**2/safe;pxy=trial[4]-trial[1]*trial[2]/safe;pyy=trial[5]-trial[2]**2/safe
            valid=(np.isfinite(trial).all() and np.min(r)>.15 and np.min(pxx)>.01
                   and np.min(pyy)>.01 and np.min(pxx*pyy-pxy*pxy)>1e-5)
            if valid and anchor_psd:
                valid=np.min(np.linalg.eigvalsh(gram(trial,True)))>=-1e-10
            if valid:break
            amount*=.5
        else:amount=0.
        weights.append(amount);modes[it]=prior_modes[it]+amount*correction_modes[it]
    return modes,weights


class Prepared:
    def __init__(self,baseline,times=None,raw_modes=None,diagnostics=None):
        self.baseline=baseline;self.times=times;self.raw_modes=raw_modes
        self.diagnostics=diagnostics or {};self.closed=False
        if raw_modes is not None:
            self.field_modes=np.stack([electric_modes(r[0],baseline.periods) for r in raw_modes])

    def evaluate(self,query):
        if self.closed:raise RuntimeError('prepared solution closed')
        fallback=self.baseline.evaluate(query)
        if self.raw_modes is None:return fallback
        p=np.asarray([(v['t'],v['x'],v['y']) for v in query.points],float)
        left=np.clip(np.searchsorted(self.times,p[:,0],side='right')-1,0,len(self.times)-2)
        alpha=(p[:,0]-self.times[left])/(self.times[left+1]-self.times[left])
        n=self.raw_modes.shape[-1];raw=np.empty((10,len(p)));field=np.empty((2,len(p)))
        for i in np.unique(left):
            mask=left==i;xy=p[mask,1:];a=alpha[mask]
            raw[:,mask]=((1-a)*point_values(self.raw_modes[i],xy,self.baseline.periods)
                         +a*point_values(self.raw_modes[i+1],xy,self.baseline.periods))
            field[:,mask]=((1-a)*point_values(self.field_modes[i],xy,self.baseline.periods)
                           +a*point_values(self.field_modes[i+1],xy,self.baseline.periods))
        r=raw[0];safe=raw.copy();safe[0]=np.maximum(r,1e-8)
        macro=central_values(safe,self.baseline.velocity_center)
        values=np.concatenate((macro[:1],field,macro[1:]))
        bad=((r<=.1)|(macro[3]<=0)|(macro[5]<=0)|(macro[3]*macro[5]-macro[4]**2<=0)
             |(~np.isfinite(values).all(axis=0)))
        out={}
        for key in query.fields:
            value=values[FIELDS.index(key)].copy();value[bad]=np.asarray(fallback.fields[key])[bad]
            out[key]=tuple(map(float,value))
        return AnswerView(kind=query.kind,fields=out,points=query.points,shape=(len(p),))

    def close(self):
        self.closed=True;self.baseline.close()


def prepare_solution(case,solve_budget=None):
    started=candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget,bool) or not isinstance(solve_budget,(int,float))
            or not np.isfinite(solve_budget) or solve_budget<=0):raise ValueError('finite positive shared solve budget required')
    budget=850. if solve_budget is None else float(solve_budget)
    deadline=started+min(MAX_PREPARE_SECONDS,budget-RETURN_RESERVE)
    def check():
        if candidate_solve_clock()>=deadline:raise TimeoutError('dynamic macro return reserve')
    q=case.public_data['question'];periods=np.asarray(q['grid']['x_domain'],float);interval=np.asarray(q['time_domain'],float)
    nu,theta=float(q['collision_nu']),float(q['temperature'])
    if (periods.shape!=(2,) or interval.shape!=(2,) or not np.isfinite(periods).all()
            or not np.isfinite(interval).all() or np.any(periods<=0) or interval[1]<=interval[0]):
        raise ValueError('positive periods and ordered interval required')
    limit=q.get('numerical_access',{}).get('max_points_per_request',262144)
    if type(limit) is not int or limit<=0:raise ValueError('positive request point limit required')
    limit=min(limit,262144);preview=case.public_data.get('arrays',{})
    rho_preview=np.asarray(preview.get('rho0',[]),float);j_preview=np.asarray(preview.get('momentum0',[]),float)
    center=np.zeros(2)
    if (rho_preview.ndim==2 and j_preview.shape==rho_preview.shape+(2,)
            and np.isfinite(rho_preview).all() and np.isfinite(j_preview).all() and rho_preview.mean()>0):
        center=j_preview.mean(axis=(0,1))/rho_preview.mean()
    if type(SOURCE_REQUEST_LIMIT) is not int or SOURCE_REQUEST_LIMIT<=0:
        raise ValueError('positive internal source request limit required')
    source_limit=min(limit,SOURCE_REQUEST_LIMIT)
    session=open_public_case_session(case)
    timings={'source_points':0,'source_seconds':0.,'source_calls':0,
             'source_max_batch_rate':0.,'source_max_batch_seconds':0.,'source_max_batch_points':0,
             'source_calibration_points':0,'source_calibration_seconds':0.,'source_calibration_calls':0,
             'source_local_points':0,'source_local_seconds':0.,'source_local_calls':0,'source_local_max_batch_rate':0.}
    def sample(sites,vx,vy,method):
        velocity=np.stack(np.meshgrid(vx,vy,indexing='ij'),-1).reshape(-1,2)
        request_limit=source_limit if method=='source' else limit
        block=max(1,request_limit//len(velocity));parts=[]
        for offset in range(0,len(sites),block):
            check();cells=sites[offset:offset+block]
            points=np.column_stack((np.repeat(cells,len(velocity),axis=0),np.tile(velocity+center,(len(cells),1))))
            if method=='source':
                points=points.reshape(len(cells),len(velocity),-1).transpose(1,0,2).reshape(-1,points.shape[-1])
            values=[]
            for j in range(0,len(points),request_limit):
                check();batch=points[j:j+request_limit]
                if method=='source' and timings['source_points']:
                    rate=source_cost_rate(timings,reserve=True)
                    if candidate_solve_clock()+SOURCE_COST_MARGIN*rate*len(batch)+SOURCE_FINALIZE_RESERVE>deadline:
                        raise TimeoutError('source batch would consume return reserve')
                before=candidate_solve_clock()
                try:answer=np.asarray(getattr(session,method)(batch),float)
                finally:
                    if method=='source':record_source_cost(timings,len(batch),candidate_solve_clock()-before)
                if answer.shape!=(len(batch),) or not np.isfinite(answer).all():raise FloatingPointError('invalid numerical source/initial response')
                values.append(answer)
            values=np.concatenate(values)
            if method=='source':values=values.reshape(len(velocity),len(cells)).T
            parts.append(values.reshape(len(cells),len(vx),len(vy)))
        check();return np.concatenate(parts)
    def sample_local(sites,velocity,integration,indices=None,calibration=False,max_cells_per_batch=None):
        """Query arbitrary locally shifted/rotated velocity points per site."""
        if indices is None:indices=np.arange(len(sites))
        indices=np.asarray(indices,int);count=velocity.shape[1]
        if len(indices)!=len(sites):raise ValueError('one local velocity plan index per source site required')
        block=max(1,source_limit//count);moments=np.empty((len(sites),10));all_values=np.empty((len(sites),count))
        if max_cells_per_batch is not None:
            if type(max_cells_per_batch) is not int or max_cells_per_batch<=0:
                raise ValueError('positive local cell batch limit required')
            block=min(block,max_cells_per_batch)
        for offset in range(0,len(sites),block):
            check();stop=min(offset+block,len(sites));ids=indices[offset:stop]
            local=velocity[ids];cells=sites[offset:stop]
            points=np.column_stack((np.repeat(cells,count,axis=0),(local+center).reshape(-1,2)))
            points=points.reshape(len(cells),count,-1).transpose(1,0,2).reshape(-1,points.shape[-1])
            pieces=[]
            for j in range(0,len(points),source_limit):
                check();batch=points[j:j+source_limit]
                rate=source_cost_rate(timings,reserve=True)
                if candidate_solve_clock()+SOURCE_COST_MARGIN*rate*len(batch)+SOURCE_FINALIZE_RESERVE>deadline:
                    raise TimeoutError('local source batch would consume return reserve')
                before=candidate_solve_clock()
                try:answer=np.asarray(session.source(batch),float)
                finally:record_source_cost(timings,len(batch),candidate_solve_clock()-before,calibration=calibration,local=True)
                if answer.shape!=(len(batch),) or not np.isfinite(answer).all():raise FloatingPointError('invalid local source response')
                pieces.append(answer)
            values=np.concatenate(pieces).reshape(count,len(cells)).T;all_values[offset:stop]=values
            weighted=values*integration[ids]
            for m,(a,b) in enumerate(INDICES[:10]):
                moments[offset:stop,m]=np.sum(weighted*local[:,:,0]**a*local[:,:,1]**b,axis=1)
        check();return moments,all_values
    xy=np.stack(np.meshgrid(*(np.arange(NX)*l/NX for l in periods),indexing='ij'),-1).reshape(-1,2)
    density=[]
    for offset in range(0,len(xy),limit):
        check();density.append(np.asarray(session.rho0(xy[offset:offset+limit]),float))
    density=np.concatenate(density)
    if density.shape!=(len(xy),) or not np.isfinite(density).all() or np.any(density<=0):raise ValueError('positive initial density required')
    velocity=(np.arange(INITIAL_NV)-INITIAL_NV//2)*2*INITIAL_HALF/INITIAL_NV;dv=2*INITIAL_HALF/INITIAL_NV
    initial_weights=np.stack([velocity[:,None]**a*velocity[None,:]**b*dv**2 for a,b in DEGREE5_INDICES])
    def uniform_initial(sites,rho):
        values=sample(sites,velocity,velocity,'g0')
        raw=np.einsum('rxy,mxy->mr',values,initial_weights,optimize=True)
        if np.any(raw[0]<=0):raise ValueError('positive initial quadrature mass required')
        raw*=rho[None]/raw[0];raw[0]=rho
        return raw
    def local_initial(sites,nodes,weights,rho):
        return acquire_local_initial(sites,nodes,weights,rho,session=session,
                                     center=center,limit=limit,check=check)
    if NX%INITIAL_SEED_NX:raise ValueError('initial mesh must contain the quadrature seed mesh')
    stride=NX//INITIAL_SEED_NX
    seed_xy=xy.reshape(NX,NX,2)[::stride,::stride].reshape(-1,2)
    seed_rho=density.reshape(NX,NX)[::stride,::stride].ravel()
    seed_raw=uniform_initial(seed_xy,seed_rho).reshape(21,INITIAL_SEED_NX,INITIAL_SEED_NX)
    validation_ids=(5+73*np.arange(min(INITIAL_VALIDATION_COUNT,len(xy))))%len(xy)
    validation_xy=xy[validation_ids];validation_rho=density[validation_ids]
    reference=uniform_initial(validation_xy,validation_rho)
    initial_trials=[];chosen=None
    for order,scale in INITIAL_GH_RULES:
        nodes,weights,plan=plan_initial_quadrature(seed_raw,validation_xy,periods,order,scale)
        approximate,mass=local_initial(validation_xy,nodes,weights,validation_rho)
        errors=initial_error_metrics(reference,approximate,theta,center)
        maximum_mass_error=float(np.max(np.abs(mass-1.)))
        accepted=bool(max(errors['raw_degree_relative_rms'])<.005
                  and max(errors['central_degree_relative_rms'][2:])<.01
                  and maximum_mass_error<.003
                  and errors['heat']['physical_rms_error']<.003*np.sqrt(errors['heat']['reference_rms']**2+.9900774471118267**2))
        initial_trials.append(dict(plan,errors=errors,maximum_mass_error=maximum_mass_error,accepted=accepted))
        if accepted:chosen=(order,scale);break
    initial_mass_diagnostics={}
    if chosen is None:
        initial=uniform_initial(xy,density);initial_method='uniform_fallback'
    else:
        nodes,weights,plan=plan_initial_quadrature(seed_raw,xy,periods,*chosen)
        initial,mass=local_initial(xy,nodes,weights,density)
        initial_mass_diagnostics=repair_initial_mass(initial,mass,xy,density,
            uniform_initial=uniform_initial,check=check)
        initial_method='validated_local_hermite'
    raw_initial_modes=project(initial.reshape(21,NX,NX))
    raw_initial_degree5_modes=raw_initial_modes
    baseline=baseline_from_initial(raw_initial_modes,periods,interval,center,nu,theta)
    diagnostics={'initial_seconds':candidate_solve_clock()-started,'initial_nv':INITIAL_NV,
                 'initial_method':initial_method,'initial_rule_trials':initial_trials,
                 **initial_mass_diagnostics}
    fallback=Prepared(baseline,diagnostics=diagnostics)
    if RETAIN_REPLAY:fallback.raw_initial_modes=raw_initial_modes
    try:
        velocity=(np.arange(SOURCE_NV)-SOURCE_NV//2)*2*SOURCE_HALF/SOURCE_NV;dv=2*SOURCE_HALF/SOURCE_NV
        row=np.arange(SOURCE_PILOT,dtype=float)+.5;fraction=row/SOURCE_PILOT;fraction[0]=0.;fraction[-1]=1.
        sites=np.column_stack((interval[0]+fraction*(interval[1]-interval[0]),
                               np.mod(row*np.sqrt(2.),1.)*periods[0],np.mod(row*np.sqrt(3.),1.)*periods[1]))
        pilot_sites=sites.copy()
        pilot=sample(sites,velocity,velocity,'source')
        uniform_weights=np.stack([velocity[:,None]**a*velocity[None,:]**b*dv**2 for a,b in INDICES[:10]])
        full_moments=np.einsum('rxy,mxy->mr',pilot,uniform_weights,optimize=True)
        extra_uniform_weights=np.stack([velocity[:,None]**a*velocity[None,:]**b*dv**2 for a,b in INDICES[10:]])
        pilot_degree4_extra=np.einsum('rxy,mxy->mr',pilot,extra_uniform_weights,optimize=True)
        pilot_raw=point_values(raw_initial_modes,sites[:,1:],periods)
        pilot_operator=initial_operator_at_sites(raw_initial_modes,sites[:,1:],periods,center,nu,theta)
        full_tangent=central_tangent(pilot_raw,pilot_operator[:10]+full_moments,center)
        rules=[];pilot_gh={};selected=None
        for order,scale in SOURCE_GH_RULES:
            check();nodes,integration=local_hermite(pilot_raw,order=order,scale=scale,rotate=True)
            approximate,values=sample_local(sites,nodes,integration)
            error=approximate.T-full_moments
            approximate_extra=higher_source_moments(values,nodes,integration)
            full_degree4=np.concatenate((full_moments,pilot_degree4_extra))
            error_degree4=np.concatenate((approximate,approximate_extra),axis=1).T-full_degree4
            rms_error=np.sqrt(np.mean(error_degree4**2,axis=1));rms_reference=np.sqrt(np.mean(full_degree4**2,axis=1))
            degree_errors=[]
            for degree in range(5):
                mask=np.asarray([a+b==degree for a,b in INDICES])
                floor=SOURCE_ABSOLUTE_FLOOR*theta**(degree/2)/max(float(interval[1]-interval[0]),.25)
                degree_errors.append(float(np.linalg.norm(rms_error[mask])/max(np.linalg.norm(rms_reference[mask]),floor*np.sqrt(mask.sum()))))
            transformed=central_tangent(pilot_raw,error,center);central_errors=[]
            for group,weight in (((0,),(1,)),((1,2),(1,1)),((3,4,5),(1,2,1)),((6,7),(1,1))):
                group=list(group);weight=np.asarray(weight)[:,None]
                numerator=float(np.sqrt(np.mean(weight*transformed[group]**2)))
                denominator=max(float(np.sqrt(np.mean(weight*full_tangent[group]**2))),.02)
                central_errors.append(numerator/denominator)
            degree_tolerance=SOURCE_GH24_DEGREE_TOLERANCE if order==24 else SOURCE_DEGREE_TOLERANCE
            central_tolerance=SOURCE_GH24_CENTRAL_TOLERANCE if order==24 else SOURCE_CENTRAL_TOLERANCE
            degree_tolerances=[degree_tolerance]*5
            if order==28:degree_tolerances[1]=SOURCE_GH24_DEGREE_TOLERANCE
            accepted=all(e<=tol for e,tol in zip(degree_errors,degree_tolerances)) and max(central_errors)<=central_tolerance
            diagnostic={'order':order,'scale':scale,'rotate':True,'nodes':order**2,'degree_scaled_errors':degree_errors,
                        'central_tangent_relative_errors':central_errors,'accepted':accepted,
                        'degree_tolerance':degree_tolerance,'central_tolerance':central_tolerance}
            diagnostic['degree_tolerances']=degree_tolerances
            rules.append(diagnostic);pilot_gh[order]=(nodes,integration,values,approximate)
            if accepted and selected is None:selected=(order,scale,diagnostic)
        diagnostics['source_rule_trials']=rules
        diagnostics['source_pilot_seconds']=timings['source_seconds']
        if selected is None:raise ValueError('all-time local GH source validation failed')
        order,scale,diagnostic=selected;diagnostics['source_rule']=diagnostic
        node_count=order**2
        # The full uniform pilot underestimates the observed production rate.
        # Measure actual local-node work throughout the public time interval,
        # with at least one request per calibration frame before mesh planning.
        calibration_n=int(SOURCE_CALIBRATION_GRID)
        calibration_count=min(int(SOURCE_CALIBRATION_COUNT),calibration_n**2)
        calibration_grid=np.stack(np.meshgrid(*(np.arange(calibration_n)*l/calibration_n for l in periods),
                                               indexing='ij'),-1).reshape(-1,2)
        calibration_indices=(np.arange(calibration_count)*37)%(calibration_n**2)
        calibration_xy=calibration_grid[calibration_indices]
        calibration_frames=int(SOURCE_CALIBRATION_FRAMES)
        if calibration_frames<=0:raise ValueError('positive source calibration frame count required')
        calibration_times=interval[0]+(np.arange(calibration_frames)+.5)*(interval[1]-interval[0])/calibration_frames
        calibration_plan_indices=np.tile(np.arange(calibration_count),calibration_frames)
        calibration_sites=np.column_stack((np.repeat(calibration_times,calibration_count),
                                           np.tile(calibration_xy,(calibration_frames,1))))
        calibration_raw=point_values(raw_initial_modes,calibration_xy,periods)
        calibration_velocity,calibration_integration=local_hermite(calibration_raw,order=order,scale=scale,rotate=True)
        calibration_moments,calibration_values=sample_local(calibration_sites,calibration_velocity,
            calibration_integration,indices=calibration_plan_indices,calibration=True,
            max_cells_per_batch=calibration_count)
        diagnostics['source_calibration_time']=float(np.mean(interval))
        diagnostics['source_calibration_times']=calibration_times.tolist()
        diagnostics['source_calibration_frames']=calibration_frames
        diagnostics['source_calibration_grid']=calibration_n
        diagnostics['source_calibration_cells']=calibration_count
        diagnostics['source_planning_rate']=source_cost_rate(timings)
        diagnostics['source_reserve_rate']=source_cost_rate(timings,reserve=True)
        diagnostics['source_cost_margin']=SOURCE_COST_MARGIN
        diagnostics['source_internal_request_limit']=source_limit
        diagnostics['source_request_order']='velocity_major'
        def estimated_cost(count,mesh):
            return SOURCE_COST_MARGIN*source_cost_rate(timings)*count*mesh*mesh*node_count
        minimum=SOURCE_COUNTS[0]
        diagnostics['source_mesh_estimated_seconds']={str(mesh):estimated_cost(minimum,mesh) for mesh in SOURCE_SPATIAL_COUNTS}
        n=next((mesh for mesh in SOURCE_SPATIAL_COUNTS if candidate_solve_clock()+estimated_cost(SOURCE_PLANNING_COUNT,mesh)+35<deadline),None)
        if n is None:
            n=next((mesh for mesh in SOURCE_SPATIAL_COUNTS if candidate_solve_clock()+estimated_cost(minimum,mesh)+35<deadline),None)
        if n is None:raise TimeoutError('remaining time cannot resolve local GH source frames')
        diagnostics['source_nx']=n
        coarse_xy=np.stack(np.meshgrid(*(np.arange(n)*l/n for l in periods),indexing='ij'),-1).reshape(-1,2)
        local_raw=point_values(raw_initial_modes,coarse_xy,periods)
        local_velocity,local_integration=local_hermite(local_raw,order=order,scale=scale,rotate=True)
        times=np.linspace(*interval,minimum)
        source_sites=np.column_stack((np.repeat(times,n*n),np.tile(coarse_xy,(len(times),1))))
        moments,sampled=sample_local(source_sites,local_velocity,local_integration,np.tile(np.arange(n*n),len(times)))
        sources=moments.reshape(len(times),n,n,10).transpose(0,3,1,2)
        extra=higher_source_moments(sampled,local_velocity,local_integration,np.tile(np.arange(n*n),len(times)))
        source_degree4_extra=extra.reshape(len(times),n,n,5).transpose(0,3,1,2)
        sampled_frames=sampled.reshape(len(times),n*n,node_count);del sampled
        for target in SOURCE_COUNTS[1:]:
            if target!=2*len(times)-1:raise ValueError('source refinement must be nested doubling')
            extra=len(times)-1
            if candidate_solve_clock()+estimated_cost(extra,n)+35>deadline:break
            middle=(times[:-1]+times[1:])*.5
            sites_mid=np.column_stack((np.repeat(middle,n*n),np.tile(coarse_xy,(len(middle),1))))
            try:moments_mid,values_mid=sample_local(sites_mid,local_velocity,local_integration,np.tile(np.arange(n*n),len(middle)))
            except TimeoutError:
                diagnostics['refinement_interrupted']=True;break
            moments_mid=moments_mid.reshape(len(middle),n,n,10).transpose(0,3,1,2)
            extra_mid=higher_source_moments(values_mid,local_velocity,local_integration,np.tile(np.arange(n*n),len(middle)))
            new_extra=np.empty((target,5,n,n));new_extra[::2]=source_degree4_extra
            new_extra[1::2]=extra_mid.reshape(len(middle),n,n,5).transpose(0,3,1,2)
            source_degree4_extra=new_extra
            new_times=np.empty(target);new_times[::2]=times;new_times[1::2]=middle
            new_sources=np.empty((target,10,n,n));new_sources[::2]=sources;new_sources[1::2]=moments_mid
            new_values=np.empty((target,n*n,node_count));new_values[::2]=sampled_frames
            new_values[1::2]=values_mid.reshape(len(middle),n*n,node_count)
            times,sources,sampled_frames=new_times,new_sources,new_values
        # Source acquisition keeps its original deadline. Finalize the measured
        # representation using unused preparation headroom, while preserving
        # the same shared-Case reserve for evaluation and serialization.
        acquisition_deadline=deadline
        deadline=min(started+budget-RETURN_RESERVE,started+MAX_PREPARE_SECONDS+40.)
        diagnostics['finalization_extra_seconds']=float(deadline-acquisition_deadline)
        check()
        sources=np.concatenate((sources,source_degree4_extra),axis=1)
        fine_operator_grid=initial_operator_at_sites(raw_initial_modes,coarse_xy,periods,center,nu,theta).reshape(15,n,n)
        coarse_initial=point_values(raw_initial_modes,coarse_xy,periods).reshape(21,n,n)
        prior=brownian_prior(coarse_initial,times,periods,nu,theta,baseline.thermal_covariance)
        coarse_operator=moment_rhs(prior[:1],np.zeros((1,15,n,n)),periods,center,nu,theta)[0]
        corrected_source=sources+fine_operator_grid[None]-coarse_operator[None]
        removed_mass=corrected_source[:,0].mean(axis=(-2,-1))
        corrected_source[:,0]-=removed_mass[:,None,None]
        diagnostics['source_density_mean_removed']=removed_mass.tolist()
        response=evolve(times,corrected_source,prior,periods,center,nu,theta,
                        relaxation=RELAXATION,closure_relaxation=CLOSURE_RELAXATION,
                        taper_fraction=TAPER_FRACTION,field_feedback=FIELD_FEEDBACK,
                        basis_temperature=(float(np.trace(baseline.thermal_covariance))/2 if BASIS_FROM_INITIAL_COVARIANCE else theta),
                        deadline_check=check)
        fine_initial=evaluate_grid(raw_initial_modes,NX)
        fine_prior=brownian_prior(fine_initial,times,periods,nu,theta,baseline.thermal_covariance)
        prior_modes=project(fine_prior[:,:15]);correction=resize_modes(response['correction_modes'][:,:15],NX)
        guarded_modes,guard_weights=guard_corrections(prior_modes,correction,check)
        modes=guarded_modes[:,:10]
        if not np.isfinite(modes).all():raise FloatingPointError('nonfinite prepared moments')
        diagnostics.update(timings);diagnostics.update({'source_nt':len(times),'guard_weights':guard_weights,
                            'max_hermite_increment':response['max_increment'],'response_parameters':response['parameters'],
                            'state_degree':4,'total_seconds':candidate_solve_clock()-started})
        prepared=Prepared(baseline,times,modes,diagnostics)
        if RETAIN_REPLAY:
            prepared.source_calibration_sites=calibration_sites
            prepared.source_calibration_times=calibration_times
            prepared.source_calibration_spatial_sites=calibration_xy
            prepared.source_calibration_plan_indices=calibration_plan_indices
            prepared.source_calibration_grid_indices=calibration_indices
            prepared.source_calibration_velocity=calibration_velocity
            prepared.source_calibration_integration=calibration_integration
            prepared.source_calibration_values=calibration_values
            prepared.source_calibration_moments=calibration_moments
            prepared.guarded_degree4_modes=guarded_modes
            prepared.raw_initial_modes=raw_initial_modes;prepared.source_pilot_sites=pilot_sites
            prepared.source_pilot_values=pilot;prepared.source_times=times;prepared.source_moments=sources
            prepared.source_pilot_full_moments=full_moments;prepared.source_pilot_initial_raw=pilot_raw
            prepared.source_local_velocity=local_velocity;prepared.source_local_integration=local_integration
            prepared.source_local_order=order;prepared.source_local_scale=scale
            for gh_order,(nodes,integration,values,approximate) in pilot_gh.items():
                setattr(prepared,'source_pilot_gh'+str(gh_order)+'_velocity',nodes)
                setattr(prepared,'source_pilot_gh'+str(gh_order)+'_integration',integration)
                setattr(prepared,'source_pilot_gh'+str(gh_order)+'_values',values)
                setattr(prepared,'source_pilot_gh'+str(gh_order)+'_moments',approximate)
                if RETAIN_HIGHER_REPLAY:
                    extra=higher_source_moments(values,nodes,integration)
                    setattr(prepared,'source_pilot_gh'+str(gh_order)+'_degree4_moments',np.concatenate((approximate,extra),axis=1))
            prepared.corrected_source_moments=corrected_source;prepared.prior_moments=prior
            prepared.source_values=sampled_frames;prepared.source_spatial_sites=coarse_xy
            prepared.fine_operator_at_source_sites=fine_operator_grid;prepared.coarse_operator=coarse_operator
            if RETAIN_HIGHER_REPLAY:
                prepared.raw_initial_degree5_modes=raw_initial_degree5_modes
                prepared.source_degree4_moments=sources
                prepared.source_pilot_full_degree4_moments=np.concatenate((full_moments,pilot_degree4_extra),axis=0)
        check();return prepared
    except (TimeoutError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        diagnostics.update(timings);diagnostics['fallback_reason']=type(error).__name__+': '+str(error)
        diagnostics['total_seconds']=candidate_solve_clock()-started
        return fallback
