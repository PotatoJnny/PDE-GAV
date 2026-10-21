"""Balanced kinetic charge/field with direct Brownian transport forecasts."""
from __future__ import annotations
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView,open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from base import prepare_solution as prepare_initial,FIELDS
from compressed_source import prepare_source
from fourier_helpers import resize_fourier
from kinetic_evolution import evolve
from initial_defect import InitialDefect


class Prepared:
    def __init__(self,baseline,times=None,modes=None):
        self.baseline=baseline;self.times=times;self.modes=modes

    def evaluate(self,query):
        fallback=self.baseline.evaluate(query)
        if self.modes is None:return fallback
        p=np.asarray([(v['t'],v['x'],v['y']) for v in query.points])
        left=np.clip(np.searchsorted(self.times,p[:,0],side='right')-1,0,len(self.times)-2)
        alpha=(p[:,0]-self.times[left])/(self.times[left+1]-self.times[left])
        fm=(1-alpha[:,None,None,None])*self.modes[left]+alpha[:,None,None,None]*self.modes[left+1]
        bx=np.exp(1j*p[:,1,None]*self.baseline.wave[0])
        by=np.exp(1j*p[:,2,None]*self.baseline.wave[1])
        m=np.einsum('ncij,ni,nj->nc',fm,bx,by).real
        rho=np.maximum(m[:,0],1e-8);ux=m[:,1]/rho;uy=m[:,2]/rho
        pxx=m[:,3]-rho*ux*ux;pxy=m[:,4]-rho*ux*uy;pyy=m[:,5]-rho*uy*uy
        qx=m[:,6]-ux*(1.5*m[:,3]+.5*m[:,5])-uy*m[:,4]+rho*ux*(ux*ux+uy*uy)
        qy=m[:,7]-uy*(.5*m[:,3]+1.5*m[:,5])-ux*m[:,4]+rho*uy*(ux*ux+uy*uy)
        c=self.baseline.velocity_center
        values=dict(zip(FIELDS,(rho,m[:,8],m[:,9],m[:,1]+rho*c[0],m[:,2]+rho*c[1],pxx,pxy,pyy,qx,qy)))
        bad=(m[:,0]<.1)|(pxx<0)|(pyy<0)|(pxx*pyy-pxy*pxy<0)|(~np.isfinite(m).all(axis=1))
        out={}
        for key in query.fields:
            # Transport fields are already central macro forecasts. Preserve
            # their baseline values directly when replacing charge/field.
            if key not in ('rho','electric_x','electric_y'):
                out[key]=fallback.fields[key]
                continue
            prediction=values[key].copy();base=np.asarray(fallback.fields[key])
            prediction[bad]=base[bad]
            if not np.isfinite(prediction).all():prediction=base
            out[key]=tuple(map(float,prediction))
        return AnswerView(kind=query.kind,fields=out,points=query.points,shape=(len(p),))

    def close(self):self.baseline.close()


def prepare_solution(case,solve_budget=None):
    start=candidate_solve_clock();budget=850. if solve_budget is None else float(solve_budget)
    baseline=prepare_initial(case,budget)
    def check():
        if candidate_solve_clock()>start+budget-25:raise TimeoutError('kinetic return reserve')
    try:
        q=case.public_data['question']
        # Match the source pilot velocity lattice exactly. The fine initial
        # representation is evaluated pointwise, with all spatial modes kept,
        # when the source routine selects its affordable coarse mesh.
        fine_nx=32;nv=80;velocity=(np.arange(nv)-nv//2)*2*16./nv;dv=2*16./nv
        xy=np.stack(np.meshgrid(*[np.arange(fine_nx)*l/fine_nx for l in baseline.periods],indexing='ij'),-1).reshape(-1,2)
        vv=np.stack(np.meshgrid(velocity,velocity,indexing='ij'),-1).reshape(-1,2)+baseline.velocity_center
        check()
        session=open_public_case_session(case)
        rho=np.asarray(session.rho0(xy),float)
        request_limit=q.get('numerical_access',{}).get('max_points_per_request',262144)
        if type(request_limit) is not int or request_limit<len(vv):raise ValueError('initial request limit too small')
        blocks=max(1,min(262144,request_limit)//len(vv));values=[]
        for offset in range(0,len(xy),blocks):
            check()
            cells=xy[offset:offset+blocks]
            points=np.column_stack((np.repeat(cells,len(vv),axis=0),np.tile(vv,(len(cells),1))))
            g=np.asarray(session.g0(points),float).reshape(len(cells),nv,nv)
            norm=g.sum(axis=(-2,-1),keepdims=True)*dv*dv
            if not np.isfinite(g).all() or np.any(norm<=0):raise FloatingPointError('invalid fine initial distribution')
            g*=rho[offset:offset+len(cells),None,None]/norm
            values.append(g.astype(np.float32));check()
        fine_initial=np.concatenate(values).reshape(fine_nx,fine_nx,nv,nv)
        balance=InitialDefect(fine_initial,baseline.periods,velocity,baseline.velocity_center,
                              float(q['collision_nu']),float(q['temperature']),deadline_check=check)
        del fine_initial,values
        sampled=prepare_source(case,baseline.velocity_center,start+budget-75.,balance=balance)
        if sampled is None:return Prepared(baseline)
        times,source,source_velocity,diagnostics=sampled
        if not np.array_equal(source_velocity,velocity):raise ValueError('source and initial velocity grids differ')
        nx=source.shape[1];initial,_,_=balance.coarse(nx)
        check()
        result=evolve(initial,source,times,baseline.periods,velocity,baseline.velocity_center,
                      float(q['collision_nu']),float(q['temperature']),deadline_check=check)
        correction=resize_fourier(result['correction_modes'],(32,32),normalized=True)
        modes=baseline.modes[None,:8]+correction
        base_grid=np.fft.ifft2(baseline.modes[:8]*32**2).real
        for it in range(len(modes)):
            delta=np.fft.ifft2(correction[it]*32**2).real;weight=1.
            for _ in range(12):
                m=base_grid+weight*delta;r=m[0]
                pxx=m[3]-m[1]**2/np.maximum(r,.01)
                pxy=m[4]-m[1]*m[2]/np.maximum(r,.01)
                pyy=m[5]-m[2]**2/np.maximum(r,.01)
                if (np.min(r)>.2 and np.min(pxx)>.02 and np.min(pyy)>.02
                        and np.min(pxx*pyy-pxy*pxy)>.001):break
                weight*=.5
            modes[it]=baseline.modes[:8]+weight*correction[it]
        kx,ky=np.meshgrid(*baseline.wave,indexing='ij');k2=kx*kx+ky*ky
        inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
        electric=np.stack((-1j*kx*inv*modes[:,0],-1j*ky*inv*modes[:,0]),axis=1)
        modes=np.concatenate((modes,electric),axis=1)
        if not np.isfinite(modes).all():return Prepared(baseline)
        return Prepared(baseline,result['times'],modes)
    except (FloatingPointError,TimeoutError,np.linalg.LinAlgError,ValueError):
        return Prepared(baseline)
