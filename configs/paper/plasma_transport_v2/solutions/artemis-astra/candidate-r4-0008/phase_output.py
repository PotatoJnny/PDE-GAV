"""Guarded source-frequency interpolation of prepared macroscopic modes."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import make_interp_spline
from pde_agent_verifier_solver.sdk import AnswerView

FIELDS=('M0','M1','M2','M3','E')
FLOORS=np.array([1.0041253663782175,.4788345668393978,
                 2.930763214320111,1.9801548942236534,.0698185715388132])

def linear(times,values,query):
    query=np.asarray(query)
    index=np.clip(np.searchsorted(times,query,side='right')-1,0,len(times)-2)
    fraction=np.clip((query-times[index])/(times[index+1]-times[index]),0.,1.)
    return ((1-fraction[:,None,None])*values[index]
            +fraction[:,None,None]*values[index+1])

def _errors(predicted,truth):
    return np.sum(abs(predicted-truth)**2,axis=0)/FLOORS[:,None]**2

def central_defect(predicted,reference):
    """Exact transport objective on the independent uniform spatial probes."""
    def centered(modes):
        raw=np.fft.ifft(modes,axis=-1).real*modes.shape[-1]
        n,j,m2,m3,e=np.moveaxis(raw,1,0)
        if np.any(n<=0):return None
        u=j/n
        return np.stack((n,j,m2-j*u,.5*(m3-3*u*m2+2*n*u**3),e),axis=1)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        p=centered(predicted);r=centered(reference)
        if p is None or r is None or not np.isfinite(p).all() or not np.isfinite(r).all():
            return float('inf')
        floors=FLOORS.copy();floors[3]*=.5
        error=np.sqrt(np.mean((p-r)**2,axis=(0,2))/
                      (np.mean(r*r,axis=(0,2))+floors**2))
        return float(error@np.array([.125,.25,.25,.25,.125]))

class PhasePrepared:
    def __init__(self,prepared,frequency,info):
        self.prepared=prepared
        self.frequency=frequency
        self.indices=np.flatnonzero(frequency)
        self.phase_output_info=info
        self.temporal_phase_model=True
        self.envelope=(prepared.modes[:,:,self.indices]
                       *np.exp(-1j*(prepared.times-prepared.times[0])[:,None]
                               *frequency[self.indices][None,:])[:,None,:])
        controls=np.asarray(getattr(prepared,'time_coefficients',prepared.modes))
        self.projection_mode_bounds=np.max(abs(controls),axis=0)
        self.projection_mode_bounds[:,self.indices]=np.max(abs(self.envelope),axis=0)
        self.projection_density_mean_interval=(float(np.min(controls[:,0,0].real)),
                                               float(np.max(controls[:,0,0].real)))

    def __getattr__(self,name):
        return getattr(self.prepared,name)

    def evaluate(self,query):
        answer=self.prepared.evaluate(query)
        ts=np.clip(np.asarray([point['t'] for point in query.points]),
                   self.times[0],self.times[-1])
        xs=np.asarray([point['x'] for point in query.points])
        proposed=linear(self.times,self.envelope,ts)
        proposed*=np.exp(1j*(ts-self.times[0])[:,None]
                         *self.frequency[self.indices][None,:])[:,None,:]
        if hasattr(self.prepared,'temporal_spline'):
            original=self.prepared.temporal_spline(ts)[:,:,self.indices]
        else:
            original=linear(self.times,self.modes[:,:,self.indices],ts)
        k=np.fft.fftfreq(self.modes.shape[-1],1/self.modes.shape[-1])[self.indices]
        spatial=np.exp(2j*np.pi*(xs[:,None]-self.xmin)/self.length*k)
        delta=np.einsum('pfk,pk->pf',proposed-original,spatial).real
        fields={field:tuple(np.asarray(answer.fields[field])+delta[:,i])
                for i,field in enumerate(FIELDS)}
        return AnswerView(kind=answer.kind,fields=fields,axes=answer.axes,
                          points=answer.points,shape=answer.shape)

def wrap(prepared,frequency,probe_times,probe_modes,background=None):
    times=np.asarray(prepared.times);modes=np.asarray(prepared.modes)
    info=dict(accepted_modes=0,method='source_carrier_linear_envelope',
              independent_points=len(probe_times))
    prepared.phase_output_info=info
    prepared.source_frequency=frequency
    prepared.macro_probe_times=np.asarray(probe_times)
    prepared.macro_probe_modes=np.asarray(probe_modes)
    if frequency is None or len(times)<9 or len(probe_times)<4:
        info['reason']='insufficient_history_or_probes'
        return prepared
    frequency=np.asarray(frequency)
    candidate=frequency!=0
    nx=modes.shape[-1]
    candidate[0]=False;candidate[nx//2]=False
    if not np.any(candidate):
        info['reason']='no_source_carrier'
        return prepared
    coarse=np.arange(0,len(times),2)
    if coarse[-1]!=len(times)-1:coarse=np.r_[coarse,len(times)-1]
    hold=np.setdiff1d(np.arange(len(times)),coarse)
    relative=times-times[0]
    phase=np.exp(-1j*relative[:,None]*frequency[None,:])
    envelope=modes*phase[:,None,:]
    if hasattr(prepared,'temporal_spline'):
        degree=min(prepared.temporal_spline.k,len(coarse)-1)
        if degree%2==0:degree-=1
        baseline=make_interp_spline(times[coarse],modes[coarse],k=degree)(times[hold])
    else:
        baseline=linear(times[coarse],modes[coarse],times[hold])
    modeled=linear(times[coarse],envelope[coarse],times[hold])/phase[hold,None,:]
    truth=modes[hold]
    baseline_error=_errors(baseline,truth)
    model_error=_errors(modeled,truth)
    norm=np.sum(abs(truth)**2,axis=0)/FLOORS[:,None]**2
    norm_row=norm.sum(0)
    selected=candidate&(model_error.sum(0)<.01*baseline_error.sum(0))
    selected&=model_error.sum(0)<1e-8*norm_row
    selected&=baseline_error.sum(0)>1e-20*norm_row
    selected&=np.all(model_error<=1.01*baseline_error+1e-20*norm,axis=0)
    # These are actual intervening solver states, never training history.
    probe_times=np.asarray(probe_times);probe_modes=np.asarray(probe_modes)
    baseline=(prepared.temporal_spline(probe_times) if hasattr(prepared,'temporal_spline')
              else linear(times,modes,probe_times))
    modeled=linear(times,envelope,probe_times)*np.exp(
        1j*(probe_times-times[0])[:,None]*frequency[None,:])[:,None,:]
    baseline_probe=_errors(baseline,probe_modes)
    model_probe=_errors(modeled,probe_modes)
    probe_norm=np.sum(abs(probe_modes)**2,axis=0)/FLOORS[:,None]**2
    selected&=model_probe.sum(0)<=.1*baseline_probe.sum(0)+1e-20*probe_norm.sum(0)
    selected&=np.all(model_probe<=1.01*baseline_probe+1e-20*probe_norm,axis=0)
    negative=(-np.arange(nx))%nx
    selected&=selected[negative]
    info.update(accepted_modes=int(np.sum(selected)),reason='accepted' if np.any(selected) else 'holdout_or_independent_defect',
                selected_holdout_baseline=float(baseline_error[:,selected].sum()),
                selected_holdout_model=float(model_error[:,selected].sum()),
                selected_probe_baseline=float(baseline_probe[:,selected].sum()),
                selected_probe_model=float(model_probe[:,selected].sum()))
    if not np.any(selected):return prepared
    mixed=baseline.copy()
    mixed[:,:,selected]=modeled[:,:,selected]
    ordinary_score=central_defect(baseline,probe_modes)
    phase_score=central_defect(mixed,probe_modes)
    info.update(central_baseline_defect=ordinary_score,central_model_defect=phase_score)
    if (not np.isfinite(ordinary_score) or not np.isfinite(phase_score)
            or phase_score>1.01*ordinary_score+1e-12):
        info.update(rejected_modes=info['accepted_modes'],accepted_modes=0,
                    reason='central_transport_defect')
        return prepared
    candidate=PhasePrepared(prepared,np.where(selected,frequency,0.),info)
    mean_low=candidate.projection_density_mean_interval[0]
    floor=0.
    if background is not None:
        background=float(background)
        if not np.isfinite(background) or background<=0:
            info.update(rejected_modes=info['accepted_modes'],accepted_modes=0,
                        reason='invalid_background')
            return prepared
        mean_low=min(mean_low,background)
        floor=.05*background
    lower=mean_low-float(candidate.projection_mode_bounds[0,1:].sum())
    info['density_lower_bound']=lower
    if not np.isfinite(candidate.projection_mode_bounds).all() or not lower>floor:
        info.update(rejected_modes=info['accepted_modes'],accepted_modes=0,
                    reason='density_certificate')
        return prepared
    return candidate
