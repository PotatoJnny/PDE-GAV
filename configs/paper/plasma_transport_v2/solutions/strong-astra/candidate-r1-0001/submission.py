"""Fourier-Hermite moment discretization of the forced full-space VPFP PDE."""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.special import roots_hermitenorm
from scipy.fft import fft2, ifft2
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

CONFIG = dict(nx=33, degree=16, nq=32, quadrature_scale=1.0,
              time_count=1600, quad_thresholds=(.5,2.,4.), source_orders=(32,24,20,12),
              max_dt=0.0015, collision_cfl=1.5, hermite_temperature_factor=3.0)
FIELDS = ('rho','electric_x','electric_y','momentum_x','momentum_y',
          'pressure_xx','pressure_xy','pressure_yy','heat_flux_x','heat_flux_y')

def basis_values(v, scale, degree):
    out = np.empty((degree+1, len(v)))
    out[0] = 1.
    if degree:
        out[1] = v/scale
    for a in range(1, degree):
        out[a+1] = ((v/scale)*out[a]-np.sqrt(a)*out[a-1])/np.sqrt(a+1)
    return out

def validate_late_source_quadrature(source_values, periods, t0, t1, nu, theta, scale):
    """Choose a late velocity rule using a small public-source moment check."""
    if nu<=0 or 4./nu>=t1-t0:
        return 12,np.sqrt(theta),{'checked':False,'chosen_order':12}
    start=t0+4./nu
    sample_times=np.array([start,.5*(start+t1),t1])
    sites=np.array([[.071,.137],[.283,.417],[.593,.719],[.827,.941],
                    [.13,.67],[.67,.23],[.37,.87],[.91,.37],[.49,.53]])*periods
    tx=np.column_stack([np.repeat(sample_times,len(sites)),
                        np.tile(sites,(len(sample_times),1))])
    low_pairs=[(a,d-a) for d in range(13) for a in range(d,-1,-1)]
    def moments(order,qscale):
        z,w=roots_hermitenorm(order)
        v=qscale*z
        weights=w*np.exp(z*z/2)*qscale
        h=basis_values(v,scale,12)
        projection=np.array([np.outer(h[a]*weights,h[b]*weights).ravel()
                             for a,b in low_pairs]).T
        vv=np.stack(np.meshgrid(v,v,indexing='ij'),axis=-1).reshape(-1,2)
        pts=np.column_stack([np.repeat(tx,order*order,axis=0),
                             np.tile(vv,(len(tx),1))])
        values=np.asarray(source_values(pts),dtype=float)
        return values.reshape(len(tx),order*order)@projection
    reference=moments(32,1.)
    report={'checked':True,'times':sample_times.tolist(),'tested':[]}
    for order in (10,12,14,15,16,20,22,24):
        difference=np.abs(moments(order,np.sqrt(theta))-reference)
        density_error=float(difference[:,0].max())
        other_error=float(difference[:,1:10].max())
        intermediate_error=float(difference[:,10:45].max())
        high_error=float(difference[:,45:].max())
        passed=(density_error<=2.e-5 and other_error<=2.e-5 and
                intermediate_error<=8.e-5 and high_error<=5.e-4)
        report['tested'].append({'order':order,'density_error':density_error,
                                 'other_error':other_error,'degree4_8_error':intermediate_error,
                                 'degree9_12_error':high_error,'passed':passed})
        if passed:
            report['chosen_order']=order
            return order,np.sqrt(theta),report
    report['chosen_order']=32
    return 32,1.,report


def fit_source_cost(points0,calls0,seconds0,points1,calls1,seconds1):
    """A nonnegative point/call cost model bounding both measured samples."""
    seconds0=max(float(seconds0),1.e-6)
    seconds1=max(float(seconds1),1.e-6)
    determinant=points0*calls1-points1*calls0
    threshold=1.e-12*max(abs(points0*calls1),abs(points1*calls0),1.)
    if abs(determinant)<=threshold:
        return max(seconds0/points0,seconds1/points1),0.,'per_point_degenerate'
    per_point=(seconds0*calls1-seconds1*calls0)/determinant
    per_call=(points0*seconds1-points1*seconds0)/determinant
    if per_point<0:
        return 0.,max(seconds0/calls0,seconds1/calls1),'per_call_bound'
    if per_call<0:
        return max(seconds0/points0,seconds1/points1),0.,'per_point_bound'
    return per_point,per_call,'affine'


class Prepared:
    def __init__(self, times, coeffs, periods, pairs, scale):
        self.times = np.asarray(times)
        self.coeffs = np.asarray(coeffs)
        self.periods = periods
        self.pairs = pairs
        self.scale = scale
        self.index = {p:i for i,p in enumerate(pairs)}
        self.nx = self.coeffs.shape[-1]
        self.freq = np.fft.fftfreq(self.nx)*self.nx

    def evaluate(self, query):
        if query.kind != QueryKind.POINTS:
            raise ValueError('POINTS required')
        pts = np.array([(p['t'],p['x'],p['y']) for p in query.points], dtype=float).reshape(-1,3)
        ts = np.clip(pts[:,0], self.times[0], self.times[-1])
        lower = np.searchsorted(self.times,ts,side='right')-1
        first = np.clip(lower-1,0,len(self.times)-4)
        phase_x = np.exp(2j*np.pi*pts[:,1,None]/self.periods[0]*self.freq)
        phase_y = np.exp(2j*np.pi*pts[:,2,None]/self.periods[1]*self.freq)
        vals = np.empty((len(pts),self.coeffs.shape[1]))
        for n in range(len(pts)):
            ids=first[n]+np.arange(4)
            knots=self.times[ids]
            weights=np.ones(4)
            for j in range(4):
                for k in range(4):
                    if j!=k:
                        weights[j]*=(ts[n]-knots[k])/(knots[j]-knots[k])
            modes=np.einsum('t,taij->aij',weights,self.coeffs[ids])
            vals[n] = np.einsum('aij,i,j->a',modes,phase_x[n],phase_y[n]).real
        c = lambda a,b: vals[:,self.index[(a,b)]]
        s = self.scale
        rho = np.maximum(c(0,0),1.e-8)
        jx,jy = s*c(1,0),s*c(0,1)
        ux,uy = jx/rho,jy/rho
        mxx = s*s*(rho+np.sqrt(2)*c(2,0))
        myy = s*s*(rho+np.sqrt(2)*c(0,2))
        mxy = s*s*c(1,1)
        rxxx = s**3*(np.sqrt(6)*c(3,0)+3*c(1,0))
        rxyy = s**3*(np.sqrt(2)*c(1,2)+c(1,0))
        rxxy = s**3*(np.sqrt(2)*c(2,1)+c(0,1))
        ryyy = s**3*(np.sqrt(6)*c(0,3)+3*c(0,1))
        speed2=ux*ux+uy*uy
        qx=.5*(rxxx+rxyy-ux*(mxx+myy)-2*(ux*mxx+uy*mxy)+2*rho*speed2*ux)
        qy=.5*(rxxy+ryyy-uy*(mxx+myy)-2*(ux*mxy+uy*myy)+2*rho*speed2*uy)
        values = dict(rho=rho,electric_x=vals[:,-2],electric_y=vals[:,-1],
                      momentum_x=jx,momentum_y=jy,pressure_xx=mxx-jx*ux,
                      pressure_xy=mxy-jx*uy,pressure_yy=myy-jy*uy,
                      heat_flux_x=qx,heat_flux_y=qy)
        return AnswerView(kind=query.kind,points=query.points,shape=(len(pts),),
                          fields={k:tuple(float(x) for x in values[k]) for k in query.fields})

    def close(self):
        pass

def prepare_solution(case, solve_budget=None):
    started=candidate_solve_clock()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU experiment requires assigned CUDA device')
    device=torch.device('cuda:0')
    torch.cuda.synchronize(device)
    budget=float(solve_budget if solve_budget is not None else 890.)
    cfg=dict(CONFIG)
    cfg['source_orders']=list(cfg['source_orders'])
    nx,degree,nq=cfg['nx'],cfg['degree'],cfg['nq']
    question=case.public_data['question']
    periods=np.asarray(question['grid']['x_domain'],dtype=float)
    t0,t1=map(float,question['time_domain'])
    nu,theta=float(question['collision_nu']),float(question['temperature'])
    scale=np.sqrt(theta*cfg['hermite_temperature_factor'])
    pairs=[(a,b) for d in range(degree+1) for a in range(d,-1,-1) for b in [d-a]]
    idx={p:i for i,p in enumerate(pairs)}
    nm=len(pairs)
    orders=np.array([a+b for a,b in pairs])[:,None,None]
    axes=[np.arange(nx)*p/nx for p in periods]
    xy=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2)
    quad_cache={}
    def quadrature(order,qscale):
        key=(int(order),float(qscale))
        if key not in quad_cache:
            z,w=roots_hermitenorm(order)
            v=qscale*z
            weights=w*np.exp(z*z/2)*qscale
            h=basis_values(v,scale,degree)
            projection=np.array([np.outer(h[a]*weights,h[b]*weights).ravel()
                                 for a,b in pairs]).T
            vv=np.stack(np.meshgrid(v,v,indexing='ij'),axis=-1).reshape(-1,2)
            phase=np.column_stack([np.repeat(xy,order*order,axis=0),
                                   np.tile(vv,(nx*nx,1))])
            quad_cache[key]=(phase,torch.as_tensor(projection,device=device,dtype=torch.float64),int(order))
        return quad_cache[key]
    initial_rule=quadrature(nq,cfg['quadrature_scale'])
    max_points=min(int(question.get('numerical_access',{}).get('max_points_per_request',262144)),262144)
    session=open_public_case_session(case)
    def check():
        torch.cuda.synchronize(device)
        if candidate_solve_clock()-started>budget-15:
            raise TimeoutError('Fourier-Hermite solver exhausted remaining case budget')
    def ask(method,pts):
        vals=[]
        for start in range(0,len(pts),max_points):
            check()
            vals.append(np.asarray(method(pts[start:start+max_points]),dtype=float))
        return np.concatenate(vals)
    late_order,late_qscale,late_report=validate_late_source_quadrature(
        lambda pts:ask(session.source,pts),periods,t0,t1,nu,theta,scale)
    cfg['source_orders'][-1]=late_order
    # Sustained fine velocity structure must also be resolved before the late
    # interval; collision relaxation alone does not certify the forced source.
    cfg['source_orders'][1]=max(cfg['source_orders'][1],late_order)
    cfg['source_orders'][2]=max(cfg['source_orders'][2],late_order)
    def project(values,rule):
        _,projection,order=rule
        values=torch.as_tensor(values,device=device,dtype=torch.float64)
        return (values.reshape(nx*nx,order*order)@projection).T.reshape(nm,nx,nx)
    rho=ask(session.rho0,xy).reshape(nx,nx)
    rho += 1.-rho.mean()
    rho=torch.as_tensor(rho,device=device,dtype=torch.float64)
    c=project(ask(session.g0,initial_rule[0]),initial_rule)*rho[None]
    # The zeroth coefficient is known directly without velocity quadrature.
    c[0]=rho
    kx,ky=np.meshgrid(*(2*np.pi*np.fft.fftfreq(nx,d=p/nx) for p in periods),indexing='ij')
    k2=kx*kx+ky*ky
    invk2=np.zeros_like(k2)
    np.divide(1.,k2,out=invk2,where=k2>0)
    plusx=np.zeros((nm,nm));plusy=np.zeros((nm,nm))
    minusx=np.zeros((nm,nm));minusy=np.zeros((nm,nm))
    for i,(a,b) in enumerate(pairs):
        if a+b<degree:
            plusx[i,idx[(a+1,b)]]=np.sqrt(a+1)
            plusy[i,idx[(a,b+1)]]=np.sqrt(b+1)
        if a:
            minusx[i,idx[(a-1,b)]]=np.sqrt(a)
        if b:
            minusy[i,idx[(a,b-1)]]=np.sqrt(b)
    streamx=scale*(plusx+minusx)
    streamy=scale*(plusy+minusy)
    mx_indices=np.argmax(minusx,axis=1);mx_weights=minusx.max(axis=1)[:,None,None]
    my_indices=np.argmax(minusy,axis=1);my_weights=minusy.max(axis=1)[:,None,None]
    px_indices=np.argmax(plusx,axis=1);px_weights=plusx.max(axis=1)[:,None,None]
    py_indices=np.argmax(plusy,axis=1);py_weights=plusy.max(axis=1)[:,None,None]
    lower2x_indices=np.array([idx[(a-2,b)] if a>=2 else 0 for a,b in pairs])
    lower2y_indices=np.array([idx[(a,b-2)] if b>=2 else 0 for a,b in pairs])
    lower2x_weights=np.array([np.sqrt(a*(a-1)) if a>=2 else 0. for a,b in pairs])[:,None,None]
    lower2y_weights=np.array([np.sqrt(b*(b-1)) if b>=2 else 0. for a,b in pairs])[:,None,None]
    max_wave=max(np.max(np.abs(kx)),np.max(np.abs(ky)))
    kx=torch.as_tensor(kx,device=device,dtype=torch.float64)
    ky=torch.as_tensor(ky,device=device,dtype=torch.float64)
    invk2=torch.as_tensor(invk2,device=device,dtype=torch.float64)
    orders=torch.as_tensor(orders,device=device,dtype=torch.float64)
    mx_indices=torch.as_tensor(mx_indices,device=device,dtype=torch.long)
    my_indices=torch.as_tensor(my_indices,device=device,dtype=torch.long)
    px_indices=torch.as_tensor(px_indices,device=device,dtype=torch.long)
    py_indices=torch.as_tensor(py_indices,device=device,dtype=torch.long)
    lower2x_indices=torch.as_tensor(lower2x_indices,device=device,dtype=torch.long)
    lower2y_indices=torch.as_tensor(lower2y_indices,device=device,dtype=torch.long)
    mx_weights=torch.as_tensor(mx_weights,device=device,dtype=torch.float64)
    my_weights=torch.as_tensor(my_weights,device=device,dtype=torch.float64)
    px_weights=torch.as_tensor(px_weights,device=device,dtype=torch.float64)
    py_weights=torch.as_tensor(py_weights,device=device,dtype=torch.float64)
    lower2x_weights=torch.as_tensor(lower2x_weights,device=device,dtype=torch.float64)
    lower2y_weights=torch.as_tensor(lower2y_weights,device=device,dtype=torch.float64)
    collision_shift=nu*(theta/(scale*scale)-1.)
    def electric(modes):
        return torch.stack([torch.fft.ifft2(-1j*k*invk2*modes[0]).real for k in (kx,ky)])
    def rhs(state,src):
        modes=torch.fft.fft2(state)
        ex,ey=electric(modes)
        ax=scale*(px_weights*modes[px_indices]+mx_weights*modes[mx_indices])
        ay=scale*(py_weights*modes[py_indices]+my_weights*modes[my_indices])
        streaming=torch.fft.ifft2(-1j*(kx*ax+ky*ay)).real
        force=(ex*mx_weights*state[mx_indices]+ey*my_weights*state[my_indices])/scale
        collision=-nu*orders*state+collision_shift*(
            lower2x_weights*state[lower2x_indices]+lower2y_weights*state[lower2y_indices])
        return streaming+force+collision+src
    final_source_cache=None
    def source_rule(t):
        stage=int(np.searchsorted(cfg['quad_thresholds'],nu*(t-t0),side='right'))
        order=cfg['source_orders'][stage]
        qscale=(cfg['quadrature_scale'] if stage==0 else
                late_qscale if order==late_order else np.sqrt(theta))
        return quadrature(order,qscale)
    def source(t):
        if t==t1 and final_source_cache is not None:
            return final_source_cache
        rule=source_rule(t)
        phase=rule[0]
        pts=np.column_stack([np.full(len(phase),t),phase])
        result=project(ask(session.source,pts),rule)
        result[0]-=result[0].mean()
        return result
    output_indices=np.array([i for i,(a,b) in enumerate(pairs) if a+b<=3],dtype=int)
    output_pairs=[pairs[i] for i in output_indices]
    output_indices=torch.as_tensor(output_indices,device=device,dtype=torch.long)
    times=[];history=[]
    def save(t):
        modes=torch.fft.fft2(c[output_indices])
        emodes=torch.stack([-1j*k*invk2*modes[0] for k in (kx,ky)])
        times.append(t)
        history.append((torch.cat([modes,emodes],dim=0)/(nx*nx)).cpu().numpy())
    save(t0)
    # Dense source times resolve a short initial layer and subsequent evolution.
    # Only four consecutive projected sources are retained at any point.
    prefix=np.concatenate([[0.],np.geomspace(1.e-7,1.e-3,21),
                           np.linspace(.001,.02,39)[1:]])
    # Measure both a large early rule and the actual selected final rule.
    # The final source is retained and reused when the rolling window reaches it.
    torch.cuda.synchronize(device)
    pilot_started=candidate_solve_clock()
    window_values=[source(t0+(t1-t0)*f) for f in prefix[:4]]
    torch.cuda.synchronize(device)
    early_seconds=(candidate_solve_clock()-pilot_started)/4.
    final_started=candidate_solve_clock()
    final_source_cache=source(t1)
    torch.cuda.synchronize(device)
    final_seconds=candidate_solve_clock()-final_started
    early_order=cfg['source_orders'][0]
    final_stage=int(np.searchsorted(cfg['quad_thresholds'],nu*(t1-t0),side='right'))
    final_order=cfg['source_orders'][final_stage]
    early_points=nx*nx*early_order*early_order
    final_points=nx*nx*final_order*final_order
    early_calls=(early_points+max_points-1)//max_points
    final_calls=(final_points+max_points-1)//max_points
    per_point,per_call,cost_kind=fit_source_cost(
        early_points,early_calls,early_seconds,
        final_points,final_calls,final_seconds)
    def costs_at(fractions):
        fractions=np.asarray(fractions)
        stages=np.searchsorted(cfg['quad_thresholds'],nu*(t1-t0)*fractions,side='right')
        counts=np.asarray(cfg['source_orders'])[stages]
        points=nx*nx*counts*counts
        calls=(points+max_points-1)//max_points
        costs=per_point*points+per_call*calls
        return np.where(fractions==1.,0.,costs)
    fixed_cost=float(costs_at(prefix[4:]).sum())
    remaining=budget-(candidate_solve_clock()-started)
    available=max(0.,.88*remaining-100.)
    requested_count=max(1,int(cfg['time_count']))
    def tail_cost(count):
        return float(costs_at(np.linspace(.02,1.,count+1)[1:]).sum())
    low,high=0,requested_count
    while low<high:
        middle=(low+high+1)//2
        if fixed_cost+tail_cost(middle)<=available:
            low=middle
        else:
            high=middle-1
    # One final interval still covers the public domain and reuses the cached
    # endpoint, even if the fixed prefix itself exceeds the source reserve.
    planned_count=max(1,low)
    planning_report=dict(
        cost_kind=cost_kind,seconds_per_point=per_point,seconds_per_call=per_call,
        early_order=int(early_order),early_points=int(early_points),early_calls=int(early_calls),
        early_seconds=early_seconds,final_order=int(final_order),
        final_points=int(final_points),final_calls=int(final_calls),final_seconds=final_seconds,
        requested_tail_count=requested_count,planned_tail_count=planned_count,
        fixed_source_seconds=fixed_cost,predicted_tail_seconds=tail_cost(planned_count),
        available_source_seconds=available,remaining_case_seconds=remaining,
        fixed_prefix_over_budget=bool(fixed_cost>available),cached_final_source=True)
    fractions=np.concatenate([prefix,np.linspace(.02,1.,planned_count+1)[1:]])
    nodes=t0+(t1-t0)*fractions
    nodes[0],nodes[-1]=t0,t1
    future_sources={len(nodes)-1:final_source_cache}
    def next_source(index):
        if index in future_sources:
            return future_sources.pop(index)
        rule=source_rule(nodes[index])
        phase=rule[0]
        batch_size=max(1,max_points//len(phase))
        stop=index+1
        while (stop<min(index+batch_size,len(nodes)-1) and
               source_rule(nodes[stop]) is rule):
            stop+=1
        if stop==index+1:
            return source(nodes[index])
        selected=nodes[index:stop]
        pts=np.column_stack([np.repeat(selected,len(phase)),np.tile(phase,(len(selected),1))])
        values=ask(session.source,pts).reshape(len(selected),len(phase))
        for offset,raw in enumerate(values):
            result=project(raw,rule)
            result[0]-=result[0].mean()
            future_sources[index+offset]=result
        return future_sources.pop(index)
    window_first=0
    def source_at(t,first,values):
        knots=nodes[first:first+4]
        exact=np.flatnonzero(t==knots)
        if len(exact):
            return values[exact[0]]
        weights=np.ones(4)
        for j in range(4):
            for k in range(4):
                if j!=k:
                    weights[j]*=(t-knots[k])/(knots[j]-knots[k])
        value=weights[0]*values[0]
        for j in range(1,4):
            value+=weights[j]*values[j]
        return value
    dtmax=min(cfg['max_dt'],cfg['collision_cfl']/max(nu*degree,1.),
              .7/(max_wave*scale*np.sqrt(degree+1)*2))
    for interval,(ta,tb) in enumerate(zip(nodes[:-1],nodes[1:])):
        desired_first=int(np.clip(interval-1,0,len(nodes)-4))
        while window_first<desired_first:
            window_values.pop(0)
            next_index=window_first+4
            window_values.append(next_source(next_index))
            window_first+=1
        nsub=max(1,int(np.ceil((tb-ta)/dtmax)))
        dt=(tb-ta)/nsub
        s0=window_values[interval-window_first]
        for step in range(nsub):
            next_t=tb if step+1==nsub else ta+(step+1)*dt
            sh=source_at(ta+(step+.5)*dt,window_first,window_values)
            s1=(window_values[interval+1-window_first] if step+1==nsub else
                source_at(next_t,window_first,window_values))
            k1=rhs(c,s0);k2r=rhs(c+.5*dt*k1,sh)
            k3=rhs(c+.5*dt*k2r,sh);k4=rhs(c+dt*k3,s1)
            c=c+dt/6*(k1+2*k2r+2*k3+k4)
            save(next_t)
            s0=s1
        check()
    snapshots=np.stack(history)
    history.clear()
    quad_cache.clear()
    del window_values
    prepared=Prepared(times,snapshots,periods,output_pairs,scale)
    prepared.late_quadrature_validation=late_report
    prepared.source_budget_plan=planning_report
    torch.cuda.synchronize(device)
    return prepared
