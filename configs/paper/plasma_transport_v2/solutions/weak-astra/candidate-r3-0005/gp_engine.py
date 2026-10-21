"""Band-resolved initial-tangent current/stress with Brownian heat fallback."""
from __future__ import annotations
import numpy as np
from scipy.stats import qmc
from pde_agent_verifier_solver.sdk import AnswerView, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from base import Prepared as BrownianPrepared, FIELDS
from thermal_mixing import initial_thermal_covariance
from tangent_model import triangular_weights, electric, BandForecast, resize_source_grid,raw_tangent

NX=32
INITIAL_NX=64
SOURCE_NX=32
NV=80
BANDWISE=True
HEAT_TANGENT=False
ADAPTIVE_QUADRATURE=False
SOURCE_NV=64
SOURCE_QUADRATURE_TOLERANCE=.02


class Prepared:
    def __init__(self,baseline,forecast=None):
        self.baseline=baseline;self.forecast=forecast

    def evaluate(self,query):
        fallback=self.baseline.evaluate(query)
        if self.forecast is None:return fallback
        points=np.asarray([(p['t'],p['x'],p['y']) for p in query.points],float)
        macro=self.forecast.project(points,self.baseline.interval[0])
        if HEAT_TANGENT:
            increment=self.forecast.heat_increment(points,self.baseline.interval[0])
            macro[:,5]=np.asarray(fallback.fields.get('heat_flux_x',np.zeros(len(points))))+increment[:,0]
            macro[:,6]=np.asarray(fallback.fields.get('heat_flux_y',np.zeros(len(points))))+increment[:,1]
        pxx,pxy,pyy=macro[:,2],macro[:,3],macro[:,4]
        used=macro if HEAT_TANGENT else macro[:,:5]
        valid=np.isfinite(used).all(axis=1)&(pxx>.02)&(pyy>.02)&(pxx*pyy>pxy*pxy)
        mapping=dict(zip(FIELDS[3:],macro.T))
        out={}
        for key in query.fields:
            if key in ('rho','electric_x','electric_y') or (key.startswith('heat_flux') and not HEAT_TANGENT):
                out[key]=fallback.fields[key]
            else:
                proposed=mapping[key];base=np.asarray(fallback.fields[key])
                out[key]=tuple(map(float,np.where(valid,proposed,base)))
        return AnswerView(kind=query.kind,fields=out,points=query.points,shape=(len(points),))

    def close(self):self.baseline.close()


def baseline_from_moments(m,periods,interval,center,nu,theta):
    raw=np.asarray([m[0],m[1],m[2],m[3],m[4],m[5],.5*(m[6]+m[8]),.5*(m[7]+m[9])])
    thermal=initial_thermal_covariance(np.moveaxis(raw,0,-1))
    r,jx,jy,xx,xy,yy,qx,qy=raw;ux=jx/r;uy=jy/r
    heat=np.asarray([qx-ux*(1.5*xx+.5*yy)-uy*xy+r*ux*(ux*ux+uy*uy),
                     qy-uy*(.5*xx+1.5*yy)-ux*xy+r*uy*(ux*ux+uy*uy)])
    heat_modes=np.fft.fft2(heat)/NX**2
    modes=np.fft.fft2(np.concatenate([raw,electric(r,periods)]))/NX**2
    modes[:,NX//2,:]=0.;modes[:,:,NX//2]=0.
    heat_modes[:,NX//2,:]=0.;heat_modes[:,:,NX//2]=0.
    return BrownianPrepared(modes,periods,interval,center,thermal,nu,theta,heat_modes)


def prepare_solution(case,solve_budget=None):
    started=candidate_solve_clock();budget=850. if solve_budget is None else float(solve_budget)
    if not np.isfinite(budget) or budget<=0:raise ValueError('positive finite solve budget required')
    def check(reserve=30.):
        if candidate_solve_clock()-started>budget-reserve:raise TimeoutError('initial tangent return reserve')
    q=case.public_data['question'];periods=np.asarray(q['x_domain'],float);interval=np.asarray(q['time_domain'],float)
    preview=case.public_data.get('arrays',{});center=np.zeros(2)
    if 'rho0' in preview and 'momentum0' in preview:
        center=np.mean(np.asarray(preview['momentum0']),axis=(0,1))/np.mean(np.asarray(preview['rho0']))
    nu=float(q['collision_nu']);theta=float(q['temperature'])
    if INITIAL_NX<NX or INITIAL_NX%NX:raise ValueError('initial grid must be a multiple of projection grid')
    xy=np.stack(np.meshgrid(*[np.arange(INITIAL_NX)*l/INITIAL_NX for l in periods],indexing='ij'),-1).reshape(-1,2)
    session=open_public_case_session(case)
    request_limit=int(q.get('numerical_access',{}).get('max_points_per_request',262144));request_limit=min(request_limit,262144)
    if request_limit<=0:raise ValueError('positive source request limit required')
    rho=np.asarray(session.rho0(xy),float)
    if rho.shape!=(len(xy),) or not np.isfinite(rho).all() or np.any(rho<=0):raise ValueError('invalid initial density')
    def integrate(method,nv,degree):
        spatial_n=SOURCE_NX if method=='source' else INITIAL_NX
        sites=xy if spatial_n==INITIAL_NX else np.stack(np.meshgrid(*[np.arange(spatial_n)*l/spatial_n for l in periods],indexing='ij'),-1).reshape(-1,2)
        v=np.arange(nv)*32/nv-16;vv=np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)
        weights=triangular_weights(v,degree);blocks=max(1,request_limit//len(vv));values=[]
        for start in range(0,len(sites),blocks):
            check();cells=sites[start:start+blocks]
            columns=[np.repeat(cells,len(vv),axis=0),np.tile(vv+center,(len(cells),1))]
            if method=='source':columns.insert(0,np.full((len(cells)*len(vv),1),interval[0]))
            points=np.column_stack(columns);parts=[]
            for offset in range(0,len(points),request_limit):
                part=np.asarray(getattr(session,method)(points[offset:offset+request_limit]),float)
                if part.shape!=(len(points[offset:offset+request_limit]),) or not np.isfinite(part).all():raise ValueError('invalid initial Oracle values')
                parts.append(part);check()
            sampled=np.concatenate(parts).reshape(len(cells),len(vv));local=sampled@weights.T
            if method=='g0':
                if np.any(local[:,0]<=0):raise ValueError('positive initial velocity mass required')
                local*=rho[start:start+len(cells),None]/local[:,0,None]
                local[:,0]=rho[start:start+len(cells)]
            values.append(local)
        result=np.concatenate(values).T.reshape(len(weights),spatial_n,spatial_n)
        return resize_source_grid(result,(NX,NX)) if method=='source' else result
    fine_initial=integrate('g0',NV,4)
    stride=INITIAL_NX//NX
    initial=fine_initial[:,::stride,::stride]
    baseline=baseline_from_moments(initial,periods,interval,center,nu,theta)
    try:
        diagnostics={'initial_nx':INITIAL_NX,'source_nx':SOURCE_NX,'source_nv':SOURCE_NV,'adaptive_quadrature':ADAPTIVE_QUADRATURE}
        source_nv=SOURCE_NV
        if ADAPTIVE_QUADRATURE:
            sites=qmc.Sobol(2,scramble=True,seed=13007).random_base2(4)*periods
            pilots=[]
            for nv in (96,64):
                v=np.arange(nv)*32/nv-16;vv=np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)
                points=np.column_stack((np.full(len(sites)*len(vv),interval[0]),np.repeat(sites,len(vv),axis=0),np.tile(vv+center,(len(sites),1))))
                pieces=[]
                for offset in range(0,len(points),request_limit):
                    check();part=np.asarray(session.source(points[offset:offset+request_limit]),float)
                    if part.shape!=(len(points[offset:offset+request_limit]),) or not np.isfinite(part).all():raise ValueError('invalid quadrature pilot')
                    pieces.append(part);check()
                values=np.concatenate(pieces).reshape(len(sites),len(vv))
                pilots.append(values@triangular_weights(v,3).T)
            errors=[];start=0
            for degree in range(4):
                end=start+degree+1
                errors.append(float(np.linalg.norm((pilots[1]-pilots[0])[:,start:end])/max(np.linalg.norm(pilots[0][:,start:end]),1e-20)))
                start=end
            source_nv=64 if max(errors)<=SOURCE_QUADRATURE_TOLERANCE else 96
            diagnostics.update(source_nv=source_nv,pilot_degree_relative_error=errors)
        source=integrate('source',source_nv,3)
        check()
        # Differentiate on the fine lattice and only then point-restrict.
        # No lowpass operation is inserted before source/operator cancellation.
        fine_operator=raw_tangent(fine_initial,np.zeros((10,INITIAL_NX,INITIAL_NX)),periods,center,nu,theta)
        tangent=source+fine_operator[:,::stride,::stride]
        forecast=BandForecast(initial,source,periods,center,nu,theta,bandwise=BANDWISE,heat=HEAT_TANGENT,tangent=tangent)
        forecast.quadrature_diagnostics=diagnostics
        check()
        return Prepared(baseline,forecast)
    except (ValueError,FloatingPointError,TimeoutError,np.linalg.LinAlgError):
        return Prepared(baseline)
