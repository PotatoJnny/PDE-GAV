"""Shared fine initial state, prepared GP transport, and optional kinetic charge.

All source and initial-data acquisition happens during preparation. Completed
GP forecasts survive insufficient budget or failure of the later charge solve.
"""
from __future__ import annotations
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView,open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
import gp_engine
from tangent_model import triangular_weights,raw_tangent,BandForecast
from initial_defect import InitialDefect
from compressed_source import prepare_source
from kinetic_evolution import evolve
from fourier_helpers import resize_fourier
from charge_engine import charge_guarded_modes
from cubic_charge_projection import CubicChargePrepared as ChargePrepared


INITIAL_NX=64
NX=32
NV=80
VMAX=16.
GP_SOURCE_NX=16
GP_SOURCE_NV=64
ADAPTIVE_QUADRATURE=True
QUADRATURE_TOLERANCE=.02
BANDWISE=True
HEAT_TANGENT=False
MIN_CHARGE_BUDGET=250.


class Prepared:
    def __init__(self,transport,charge=None,diagnostics=None):
        self.transport=transport;self.charge=charge
        self.diagnostics={} if diagnostics is None else diagnostics

    def evaluate(self,query):
        transport=self.transport.evaluate(query)
        if self.charge is None:return transport
        charge=self.charge.evaluate(query)
        fields={key:charge.fields[key] if key in ('rho','electric_x','electric_y')
                else transport.fields[key] for key in query.fields}
        return AnswerView(kind=query.kind,fields=fields,points=query.points,shape=(len(query.points),))

    def close(self):
        # Both prepared branches share the same baseline object.
        self.transport.close()


def prepare_solution(case,solve_budget=None):
    started=candidate_solve_clock();budget=850. if solve_budget is None else float(solve_budget)
    if not np.isfinite(budget) or budget<=0:raise ValueError('positive finite solve budget required')
    deadline=started+budget
    def check(reserve=25.):
        if candidate_solve_clock()>deadline-reserve:raise TimeoutError('hybrid return reserve')
    if INITIAL_NX<NX or INITIAL_NX%NX:raise ValueError('fine grid must be divisible by projection grid')
    if INITIAL_NX%GP_SOURCE_NX:raise ValueError('fine initial grid must be divisible by GP source grid')
    q=case.public_data['question'];periods=np.asarray(q['x_domain'],float);interval=np.asarray(q['time_domain'],float)
    if (periods.shape!=(2,) or interval.shape!=(2,) or not np.isfinite(periods).all()
            or not np.isfinite(interval).all() or np.any(periods<=0) or interval[1]<=interval[0]):
        raise ValueError('positive periods and ordered time domain required')
    nu=float(q['collision_nu']);theta=float(q['temperature'])
    preview=case.public_data.get('arrays',{});center=np.zeros(2)
    if 'rho0' in preview and 'momentum0' in preview:
        r=np.asarray(preview['rho0'],float);j=np.asarray(preview['momentum0'],float)
        if r.ndim==2 and j.shape==r.shape+(2,) and np.isfinite(r).all() and np.isfinite(j).all() and r.mean()>0:
            center=j.mean(axis=(0,1))/r.mean()
    request_limit=q.get('numerical_access',{}).get('max_points_per_request',262144)
    if type(request_limit) is not int or request_limit<=0:raise ValueError('positive public request limit required')
    request_limit=min(request_limit,262144)
    session=open_public_case_session(case)
    def request(method,points,reserve=30.):
        parts=[]
        for offset in range(0,len(points),request_limit):
            check(reserve);batch=points[offset:offset+request_limit]
            values=np.asarray(getattr(session,method)(batch),float)
            if values.shape!=(len(batch),) or not np.isfinite(values).all():raise ValueError('invalid public initial values')
            parts.append(values);check(reserve)
        return np.concatenate(parts)
    def spatial_grid(n):
        return np.stack(np.meshgrid(*[np.arange(n)*l/n for l in periods],indexing='ij'),-1).reshape(-1,2)
    def velocity_grid(n):
        v=(np.arange(n)-n//2)*2*VMAX/n
        vv=np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)
        return v,vv

    # The only g0 pass produces both a fine kinetic state and degree4 moments.
    xy=spatial_grid(INITIAL_NX);velocity,vv=velocity_grid(NV)
    rho=request('rho0',xy)
    if np.any(rho<=0):raise ValueError('positive initial density required')
    weights=triangular_weights(velocity,4)
    fine_f=np.empty((len(xy),NV,NV),dtype=np.float32)
    fine_moments=np.empty((len(xy),len(weights)),dtype=float)
    block=max(1,request_limit//len(vv))
    for offset in range(0,len(xy),block):
        cells=xy[offset:offset+block]
        points=np.column_stack((np.repeat(cells,len(vv),axis=0),np.tile(vv+center,(len(cells),1))))
        values=request('g0',points).reshape(len(cells),len(vv))
        local=values@weights.T
        if np.any(local[:,0]<=0):raise ValueError('positive initial velocity mass required')
        factor=rho[offset:offset+len(cells)]/local[:,0]
        values*=factor[:,None];local*=factor[:,None];local[:,0]=rho[offset:offset+len(cells)]
        fine_f[offset:offset+len(cells)]=values.reshape(-1,NV,NV)
        fine_moments[offset:offset+len(cells)]=local
    fine_f=fine_f.reshape(INITIAL_NX,INITIAL_NX,NV,NV)
    fine_moments=fine_moments.T.reshape(len(weights),INITIAL_NX,INITIAL_NX)
    stride=INITIAL_NX//NX;initial=fine_moments[:,::stride,::stride]
    baseline=gp_engine.baseline_from_moments(initial,periods,interval,center,nu,theta)
    transport=gp_engine.Prepared(baseline)
    diagnostics={'initial_nx':INITIAL_NX,'initial_nv':NV,'gp_source_nx':GP_SOURCE_NX,
                 'gp_source_nv':GP_SOURCE_NV,'initial_seconds':candidate_solve_clock()-started,
                 'gp_complete':False,'charge_complete':False}
    gp_engine.HEAT_TANGENT=bool(HEAT_TANGENT)
    try:
        gp_started=candidate_solve_clock()
        sites=spatial_grid(GP_SOURCE_NX);source_nv=GP_SOURCE_NV
        if ADAPTIVE_QUADRATURE:
            pilot_started=candidate_solve_clock()
            pilot_indices=np.random.default_rng(27183).choice(len(sites),min(8,len(sites)),replace=False)
            pilot_sites=sites[pilot_indices];pilots=[];pilot_points=0
            for nv in (96,64):
                pv,pvv=velocity_grid(nv)
                points=np.column_stack((np.full(len(pilot_sites)*len(pvv),interval[0]),np.repeat(pilot_sites,len(pvv),axis=0),np.tile(pvv+center,(len(pilot_sites),1))))
                values=request('source',points).reshape(len(pilot_sites),len(pvv))
                pilots.append(values@triangular_weights(pv,2).T);pilot_points+=len(points)
            errors=[];first=0
            numerical_floor=max(1e-20,1e-12*float(np.linalg.norm(pilots[0])))
            for degree in range(3):
                last=first+degree+1
                errors.append(float(np.linalg.norm((pilots[1]-pilots[0])[:,first:last])/max(np.linalg.norm(pilots[0][:,first:last]),numerical_floor)))
                first=last
            source_nv=64 if max(errors)<=QUADRATURE_TOLERANCE else 96
            diagnostics.update(quadrature_degree_errors=errors,quadrature_pilot_points=pilot_points,
                               quadrature_seconds=candidate_solve_clock()-pilot_started,gp_source_nv=source_nv)
        source_v,source_vv=velocity_grid(source_nv)
        source_weights=triangular_weights(source_v,3)
        source_moments=np.empty((len(sites),len(source_weights)),dtype=float)
        source_block=max(1,request_limit//len(source_vv))
        for offset in range(0,len(sites),source_block):
            cells=sites[offset:offset+source_block]
            points=np.column_stack((np.full(len(cells)*len(source_vv),interval[0]),
                np.repeat(cells,len(source_vv),axis=0),np.tile(source_vv+center,(len(cells),1))))
            values=request('source',points).reshape(len(cells),len(source_vv))
            source_moments[offset:offset+len(cells)]=values@source_weights.T
        source_moments=source_moments.T.reshape(len(source_weights),GP_SOURCE_NX,GP_SOURCE_NX)
        # Differentiate fine moments before point restriction, then cancel
        # against source values sampled on the identical GP mesh.
        fine_operator=raw_tangent(fine_moments,np.zeros((10,INITIAL_NX,INITIAL_NX)),periods,center,nu,theta)
        gp_stride=INITIAL_NX//GP_SOURCE_NX
        gp_initial=fine_moments[:,::gp_stride,::gp_stride]
        tangent=source_moments+fine_operator[:,::gp_stride,::gp_stride]
        forecast=BandForecast(gp_initial,source_moments,periods,center,nu,theta,
                             bandwise=BANDWISE,heat=HEAT_TANGENT,tangent=tangent)
        check()
        transport=gp_engine.Prepared(baseline,forecast)
        diagnostics.update(gp_complete=True,gp_seconds=candidate_solve_clock()-gp_started)
    except (ValueError,FloatingPointError,TimeoutError,np.linalg.LinAlgError) as exc:
        diagnostics['gp_fallback']=type(exc).__name__
        return Prepared(transport,diagnostics=diagnostics)
    del fine_moments,fine_operator,initial,source_moments,gp_initial
    if candidate_solve_clock()>deadline-MIN_CHARGE_BUDGET:
        diagnostics['charge_fallback']='remaining_budget'
        return Prepared(transport,diagnostics=diagnostics)
    try:
        charge_started=candidate_solve_clock()
        balance=InitialDefect(fine_f,periods,velocity,center,nu,theta,deadline_check=check)
        del fine_f
        sampled=prepare_source(case,center,deadline-75.,balance=balance)
        if sampled is None:
            diagnostics['charge_fallback']='source_validation_or_budget'
            return Prepared(transport,diagnostics=diagnostics)
        times,source,source_velocity,source_diagnostics=sampled
        if not np.array_equal(source_velocity,velocity):raise ValueError('source and shared initial grids differ')
        coarse,_,_=balance.coarse(source.shape[1])
        check()
        result=evolve(coarse,source,times,periods,velocity,center,nu,theta,deadline_check=check)
        correction=resize_fourier(result['correction_modes'],(NX,NX),normalized=True)
        modes=charge_guarded_modes(baseline.modes,correction)
        kx,ky=np.meshgrid(*baseline.wave,indexing='ij');k2=kx*kx+ky*ky
        inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
        field=np.stack((-1j*kx*inv*modes[:,0],-1j*ky*inv*modes[:,0]),axis=1)
        modes=np.concatenate((modes,field),axis=1)
        if not np.isfinite(modes).all():raise FloatingPointError('nonfinite kinetic charge modes')
        check()
        charge=ChargePrepared(baseline,result['times'],modes)
        diagnostics.update(charge_complete=True,charge_seconds=candidate_solve_clock()-charge_started,
                           source=source_diagnostics)
        return Prepared(transport,charge,diagnostics)
    except (ValueError,FloatingPointError,TimeoutError,np.linalg.LinAlgError) as exc:
        diagnostics['charge_fallback']=type(exc).__name__
        return Prepared(transport,diagnostics=diagnostics)
