"""Balanced kinetic charge/field with direct Brownian transport forecasts."""
from __future__ import annotations
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView,open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from base import prepare_with_distribution as prepare_initial,FIELDS,NX,NV,VMAX
from compressed_source import prepare_source
from fourier_helpers import resize_fourier
from kinetic_evolution import evolve
from initial_defect import InitialDefect


def charge_guarded_modes(baseline_modes,correction):
    """Blend toward initial charge only if the evolved density is invalid."""
    nx,ny=baseline_modes.shape[-2:]
    modes=baseline_modes[None,:8]+correction
    base_density=np.fft.ifft2(baseline_modes[0]*(nx*ny)).real
    for it in range(len(modes)):
        delta_density=np.fft.ifft2(correction[it,0]*(nx*ny)).real
        weight=1.
        for _ in range(12):
            density=base_density+weight*delta_density
            if np.isfinite(density).all() and np.min(density)>.2:break
            weight*=.5
        modes[it]=baseline_modes[:8]+weight*correction[it]
    return modes


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
        rho=np.maximum(m[:,0],1e-8)
        values={'rho':rho,'electric_x':m[:,8],'electric_y':m[:,9]}
        # Only charge and field are returned from this evolution. Their
        # admissibility must not depend on discarded kinetic transport.
        bad=(m[:,0]<.1)|(~np.isfinite(m[:,[0,8,9]]).all(axis=1))
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
    baseline,fine_initial=prepare_initial(case,budget)
    def check():
        if candidate_solve_clock()>start+budget-25:raise TimeoutError('kinetic return reserve')
    try:
        q=case.public_data['question']
        # One shared lattice supplies accurate initial macros and the kinetic
        # stationary defect. No second initial-data Oracle pass is needed.
        velocity=(np.arange(NV)-NV//2)*2*VMAX/NV
        balance=InitialDefect(fine_initial,baseline.periods,velocity,baseline.velocity_center,
                              float(q['collision_nu']),float(q['temperature']),deadline_check=check)
        del fine_initial
        sampled=prepare_source(case,baseline.velocity_center,start+budget-75.,balance=balance)
        if sampled is None:return Prepared(baseline)
        times,source,source_velocity,diagnostics=sampled
        if not np.array_equal(source_velocity,velocity):raise ValueError('source and initial velocity grids differ')
        nx=source.shape[1];initial,_,_=balance.coarse(nx)
        check()
        result=evolve(initial,source,times,baseline.periods,velocity,baseline.velocity_center,
                      float(q['collision_nu']),float(q['temperature']),deadline_check=check)
        correction=resize_fourier(result['correction_modes'],(NX,NX),normalized=True)
        modes=charge_guarded_modes(baseline.modes,correction)
        kx,ky=np.meshgrid(*baseline.wave,indexing='ij');k2=kx*kx+ky*ky
        inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
        electric=np.stack((-1j*kx*inv*modes[:,0],-1j*ky*inv*modes[:,0]),axis=1)
        modes=np.concatenate((modes,electric),axis=1)
        if not np.isfinite(modes).all():return Prepared(baseline)
        return Prepared(baseline,result['times'],modes)
    except (FloatingPointError,TimeoutError,np.linalg.LinAlgError,ValueError):
        return Prepared(baseline)
