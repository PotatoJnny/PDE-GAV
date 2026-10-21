"""Canonical-view forced VPFP hierarchy with learned continuous macro repair."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from vpfp_hierarchy import solve_hierarchy
from vpfp_newrepair import correction

_NX=33;_NY=33;_NV=9;_NT=13

def _resize(z,n,axis):
 z=np.asarray(z,float);old=z.shape[axis]
 if old==n:return z.copy()
 h=np.fft.fftshift(np.fft.fft(z,axis=axis),axes=axis);shape=list(h.shape);shape[axis]=n;g=np.zeros(shape,complex)
 take=min(old,n);so=(old-take)//2;sn=(n-take)//2;src=[slice(None)]*h.ndim;dst=[slice(None)]*h.ndim;src[axis]=slice(so,so+take);dst[axis]=slice(sn,sn+take);g[tuple(dst)]=h[tuple(src)]
 return (np.fft.ifft(np.fft.ifftshift(g,axes=axis),axis=axis)*(n/old)).real

def _xy(z,ax,ay):return _resize(_resize(z,_NX,ax),_NY,ay)
def _interp(x,z,target,axis,log=False):
 x=np.asarray(x,float);target=np.asarray(target,float);z=np.asarray(z,float)
 if len(x)==len(target) and np.allclose(x,target,rtol=0,atol=1e-13):return z.copy()
 if log:return np.exp(np.clip(CubicSpline(x,np.log(np.maximum(z,1e-35)),axis=axis)(target),-700,50))
 return CubicSpline(x,z,axis=axis)(target)

def _canonical_case(case):
 a=case['arrays'];q0=case['question'];gq=q0['grid'];tq=q0['time'];Lx,Ly=map(float,gq['x_domain']);x0=float(a['x'][0]);y0=float(a['y'][0]);x=x0+Lx*np.arange(_NX)/_NX;y=y0+Ly*np.arange(_NY)/_NY
 vx=np.linspace(float(a['vx'][0]),float(a['vx'][-1]),_NV);vy=np.linspace(float(a['vy'][0]),float(a['vy'][-1]),_NV);t=np.linspace(float(tq['t0']),float(tq['t1']),_NT)
 rho=_xy(a['rho0'],0,1);mom=_xy(a['momentum0'],0,1);K=_xy(a['kinetic_energy0'],0,1)
 g=_xy(a['g0'],0,1);g=_interp(a['vx'],g,vx,-2,True);g=_interp(a['vy'],g,vy,-1,True)
 source=_xy(a['source'],1,2);source=_interp(a['vx'],source,vx,-2);source=_interp(a['vy'],source,vy,-1);source=_interp(a['t'],source,t,0)
 dx=(vx[-1]-vx[0])/8;dy=(vy[-1]-vy[0])/8;wx=np.full(9,dx);wy=np.full(9,dy);wx[[0,-1]]*=.5;wy[[0,-1]]*=.5
 arrays={'x':x,'y':y,'vx':vx,'vy':vy,'t':t,'rho0':rho,'momentum0':mom,'kinetic_energy0':K,'g0':g,'source':source,'velocity_weights':wx[:,None]*wy[None,:]}
 question={**q0,'grid':{**gq,'nx':_NX,'ny':_NY,'nvx':_NV,'nvy':_NV},'time':{**tq,'nt':_NT}}
 return {**case,'arrays':arrays,'question':question}

def _select(result,indices):return {'rho':result['rho'][indices].copy(),'momentum':result['momentum'][indices].copy(),'kinetic_energy':result['kinetic_energy'][indices].copy()}
def solve_with_budget(public_case,query_times,budget):
 q=np.asarray(query_times,float)
 if q.ndim!=1 or q.size==0 or not np.all(np.isfinite(q)):raise ValueError('query_times must be finite nonempty 1D')
 t0=float(public_case['question']['time']['t0']);t1=float(public_case['question']['time']['t1'])
 if np.any(q<t0-1e-12) or np.any(q>t1+1e-12):raise ValueError('query time outside public interval')
 original=public_case;nv0=min(len(original['arrays']['vx']),len(original['arrays']['vy']));public_case=_canonical_case(public_case);st=np.asarray(public_case['arrays']['t'],float);all_times=np.unique(np.concatenate((q,st)));tier=min(max(int(budget.get('tier',0)),0),2);step=(.024,.008,.004)[tier]
 complete=solve_hierarchy(public_case,all_times,maximum_step=step,return_state=True);qi=np.searchsorted(all_times,q);si=np.searchsorted(all_times,st);result=_select(complete,qi);shared={'_initial_state':complete['_initial_state'],'_source_moments':complete['_source_moments'],'_source_spline':complete['_source_spline']};sr={**shared,'_state':complete['_state'][si]};qr={**shared,'_state':complete['_state'][qi]}
 source_file='model_source_v13.npz' if nv0==13 else 'model_source.npz';sk=correction(public_case,st,sr,source_file);source=sk[0:1] if len(st)==1 else CubicSpline(st,sk,axis=0)(q);dense=.3*correction(public_case,q,qr,'model_dense_old.npz')+.7*correction(public_case,q,qr,'model_dense_rand.npz');corr=source+dense
 result['rho']+=corr[:,0];result['momentum'][...,0]+=corr[:,1];result['momentum'][...,1]+=corr[:,2];result['kinetic_energy']+=corr[:,3];result['rho']-=result['rho'].mean((-2,-1),keepdims=True)-1
 nx=len(original['arrays']['x']);ny=len(original['arrays']['y']);result['rho']=_resize(_resize(result['rho'],nx,-2),ny,-1);result['momentum']=_resize(_resize(result['momentum'],nx,-3),ny,-2);result['kinetic_energy']=_resize(_resize(result['kinetic_energy'],nx,-2),ny,-1)
 at0=np.isclose(q,t0,rtol=0,atol=1e-13);result['rho'][at0]=original['arrays']['rho0'];result['momentum'][at0]=original['arrays']['momentum0'];result['kinetic_energy'][at0]=original['arrays']['kinetic_energy0']
 for name,value in result.items():
  if not np.all(np.isfinite(value)):raise FloatingPointError(f'non-finite {name}')
 return result

def solve(public_case):return solve_with_budget(public_case,public_case['arrays']['t'],{'tier':2,'wall_time_seconds':900})
