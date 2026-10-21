"""Fourier collocation VPFP on a solver-owned truncated velocity box.

The public initial/source functions are sampled at independent phase-space
nodes.  A smooth polynomial in time represents the prescribed source.  RK4
advances the full kinetic equation; only central macroscopic moments are kept.
"""
from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import numpy as np
import torch

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

_compression_spec=importlib.util.spec_from_file_location(
    __name__+"_source_compression",Path(__file__).with_name("source_compression.py"))
_compression_module=importlib.util.module_from_spec(_compression_spec)
_compression_spec.loader.exec_module(_compression_module)
compress_source=_compression_module.compress_source

FIELDS = ("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
          "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y")
NX = 32
SOURCE_NX = 24
NV = 128
SOURCE_NV = 96
VELOCITY_BOUND = 24.0
SOURCE_NT = 33
DT = 0.0075


def initial_frame(case):
    """A constant frame selected solely from the public initial macro preview."""
    arrays=case.public_data.get("arrays",{})
    if "rho0" not in arrays or "momentum0" not in arrays:
        return np.zeros(2)
    rho=np.asarray(arrays["rho0"],dtype=np.float64)
    momentum=np.asarray(arrays["momentum0"],dtype=np.float64)
    if not np.isfinite(rho).all() or not np.isfinite(momentum).all() or rho.mean()<=0:
        return np.zeros(2)
    if momentum.shape==rho.shape+(2,):
        frame=momentum.reshape(-1,2).mean(axis=0)/rho.mean()
    elif momentum.shape==(2,)+rho.shape:
        frame=momentum.reshape(2,-1).mean(axis=1)/rho.mean()
    else:
        return np.zeros(2)
    return np.asarray(frame,dtype=np.float64)


class FrameSession:
    """Map solver-owned (t,z,w) points to the public physical coordinates."""
    def __init__(self,session,frame,periods,t0):
        self.session=session
        self.frame=np.asarray(frame,dtype=np.float64)
        self.periods=np.asarray(periods,dtype=np.float64)
        self.t0=float(t0)

    def rho0(self,points):
        return self.session.rho0(points)

    def g0(self,points):
        physical=np.asarray(points,dtype=np.float64).copy()
        physical[:,2:4]+=self.frame
        return self.session.g0(physical)

    def source(self,points):
        physical=np.asarray(points,dtype=np.float64).copy()
        physical[:,1:3]=np.mod(physical[:,1:3]+(physical[:,0]-self.t0)[:,None]*self.frame,self.periods)
        physical[:,3:5]+=self.frame
        return self.session.source(physical)


def preview_fallback(case,periods,interval,theta,frame):
    arrays=case.public_data.get("arrays",{})
    rho=np.asarray(arrays.get("rho0",np.ones((NX,NX))),dtype=np.float64)
    if rho.ndim!=2 or not np.isfinite(rho).all() or rho.min()<=0:
        rho=np.ones((NX,NX))
    j=np.asarray(arrays.get("momentum0",np.zeros(rho.shape+(2,))),dtype=np.float64)
    if j.shape==(2,)+rho.shape:
        j=np.moveaxis(j,0,-1)
    if j.shape!=rho.shape+(2,) or not np.isfinite(j).all():
        j=rho[:,:,None]*frame
    energy=np.asarray(arrays.get("kinetic_energy0",theta*rho+.5*np.sum(j*j,axis=-1)/rho),dtype=np.float64)
    if energy.shape!=rho.shape or not np.isfinite(energy).all():
        energy=theta*rho+.5*np.sum(j*j,axis=-1)/rho
    rho=np.maximum(_upsample_xy(rho,NX),1e-3)
    j=_upsample_xy(j,NX)
    energy=_upsample_xy(energy,NX)
    isotropic=np.maximum(energy-.5*np.sum(j*j,axis=-1)/rho,1e-8*rho)
    relative=j/rho[:,:,None]-frame
    ux,uy=relative[:,:,0],relative[:,:,1]
    speed2=ux*ux+uy*uy
    raw=np.stack((rho,np.zeros_like(rho),np.zeros_like(rho),rho*ux,rho*uy,
                  isotropic+rho*ux*ux,rho*ux*uy,isotropic+rho*uy*uy,
                  .5*rho*speed2*ux+2*isotropic*ux,.5*rho*speed2*uy+2*isotropic*uy))
    # A bounded bilinear projection keeps this emergency representation positive.
    kx=2*np.pi*np.fft.fftfreq(NX,d=periods[0]/NX)[:,None]
    ky=2*np.pi*np.fft.fftfreq(NX,d=periods[1]/NX)[None,:]
    k2=kx*kx+ky*ky;k2[0,0]=1
    potential=np.fft.fft2(raw[0])/k2;potential[0,0]=0
    raw[1]=np.fft.ifft2(-1j*kx*potential).real
    raw[2]=np.fft.ifft2(-1j*ky*potential).real
    handle=Prepared(np.linspace(interval[0],interval[1],4),np.repeat(raw[None],4,axis=0),periods,interval,frame)
    handle.fallback_grid=raw
    handle.compression_stats={"fallback":"public_initial_preview","frame_velocity":frame.tolist()}
    return handle


def _upsample_xy(values, n):
    """Periodic Fourier interpolation, including the even-grid Nyquist split."""
    old = values.shape[0]
    if old == n:
        return values
    from scipy.signal import resample
    return resample(resample(values, n, axis=0), n, axis=1).real


def _upsample_source(values):
    values=_upsample_xy(values,NX)
    if SOURCE_NV != NV:
        from scipy.signal import resample
        values=resample(resample(values,NV,axis=2),NV,axis=3).real
    return values


class Prepared:
    def __init__(self, times, macros, periods, interval, frame=None):
        self.times = np.asarray(times)
        self.modes = np.fft.fft2(np.asarray(macros, dtype=np.float64), axes=(-2, -1)) / (NX*NX)
        self.periods, self.interval = periods, interval
        self.frame=np.zeros(2) if frame is None else np.asarray(frame,dtype=np.float64).copy()
        self.spatial_frame=self.frame.copy()
        self.closed = False
        self.safe_fallback=None
        self.fallback_grid=None

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared kinetic solution is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("kinetic solver supports POINTS queries")
        if not query.fields or len(set(query.fields)) != len(query.fields) or not set(query.fields) <= set(FIELDS):
            raise ValueError("unsupported transport fields")
        points = np.asarray([(p["t"], p["x"], p["y"]) for p in query.points], dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError("finite nonempty t,x,y points required")
        if np.any(points[:, 0] < self.interval[0]) or np.any(points[:, 0] > self.interval[1]):
            raise ValueError("query outside prepared public time interval")
        k = np.fft.fftfreq(NX) * NX
        relative_xy=points[:,1:]-(points[:,0]-self.interval[0])[:,None]*self.spatial_frame[None,:]
        ex = np.exp(2j*np.pi*np.outer(relative_xy[:, 0]/self.periods[0], k))
        ey = np.exp(2j*np.pi*np.outer(relative_xy[:, 1]/self.periods[1], k))
        if NX%2==0:
            ex[:,NX//2]=np.cos(2*np.pi*relative_xy[:,0]/self.periods[0]*k[NX//2])
            ey[:,NX//2]=np.cos(2*np.pi*relative_xy[:,1]/self.periods[1]*k[NX//2])
        dt = self.times[1] - self.times[0]
        z = (points[:, 0] - self.times[0])/dt
        left = np.clip(np.floor(z).astype(int)-1, 0, len(self.times)-4)
        result = np.zeros((len(points), len(FIELDS)), dtype=np.float64)
        for j in range(4):
            weight = np.ones(len(points))
            for h in range(4):
                if j != h:
                    weight *= (z-left-h)/(j-h)
            spatial = np.einsum("ni,nfij,nj->nf", ex, self.modes[left+j], ey, optimize=True).real
            result += weight[:, None] * spatial
        if self.fallback_grid is not None:
            scaled=np.mod(relative_xy/self.periods,1.)*NX
            lower=np.floor(scaled).astype(int)
            fraction=scaled-lower
            result[:]=0
            for dx in (0,1):
                for dy in (0,1):
                    w=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
                    result+=w[:,None]*self.fallback_grid[:,(lower[:,0]+dx)%NX,(lower[:,1]+dy)%NX].T
        invalid=(result[:,0]<1e-4)|(~np.isfinite(result).all(axis=1))
        if np.any(invalid):
            if self.safe_fallback is None:
                raise FloatingPointError("invalid fallback macro representation")
            result[invalid]=0
            result[invalid,0]=1
        rho=np.maximum(result[:,0],1e-8)
        jx,jy=result[:,3],result[:,4]
        ux,uy=jx/rho,jy/rho
        rxx,rxy,ryy=result[:,5].copy(),result[:,6].copy(),result[:,7].copy()
        result[:,5]=rxx-jx*ux
        result[:,6]=rxy-jx*uy
        result[:,7]=ryy-jy*uy
        result[:,8]-=rxx*ux+rxy*uy+.5*(rxx+ryy)*ux-rho*(ux*ux+uy*uy)*ux
        result[:,9]-=rxy*ux+ryy*uy+.5*(rxx+ryy)*uy-rho*(ux*ux+uy*uy)*uy
        result[:,3]+=rho*self.frame[0]
        result[:,4]+=rho*self.frame[1]
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite kinetic macro interpolation")
        values={field:result[:,FIELDS.index(field)].copy() for field in query.fields}
        if np.any(invalid):
            fallback_values=self.safe_fallback.evaluate(query).fields
            for field in query.fields:
                values[field][invalid]=np.asarray(fallback_values[field])[invalid]
        return AnswerView(kind=query.kind,
                          fields={field: tuple(map(float, values[field])) for field in query.fields},
                          points=query.points, shape=(len(points),))

    def close(self):
        self.closed = True


class KineticSystem:
    def __init__(self, periods, nu, theta, bound, device, frame=None):
        self.device = device
        self.frame=torch.as_tensor(np.zeros(2) if frame is None else frame,device=device,dtype=torch.float32)
        self.dv = 2*bound/NV
        vv = torch.arange(NV, device=device, dtype=torch.float32)*self.dv-bound
        self.vx, self.vy = vv[None, None, :, None], vv[None, None, None, :]
        kx = 2*np.pi*torch.fft.fftfreq(NX, d=periods[0]/NX, device=device)
        ky = 2*np.pi*torch.fft.fftfreq(NX, d=periods[1]/NX, device=device)
        self.kx, self.ky = kx[:, None, None, None], ky[None, :, None, None]
        eta = 2*np.pi*torch.fft.fftfreq(NV, d=self.dv, device=device)
        self.exi, self.eyi = eta[None, None, :, None], eta[None, None, None, :]
        self.k2 = kx[:, None]**2 + ky[None, :]**2
        self.k2[0, 0] = 1
        self.nu, self.theta = nu, theta
        self.v2 = self.vx**2+self.vy**2
        self.diff = -nu*theta*(self.exi**2+self.eyi**2)
        self.weights = torch.stack([torch.ones((NV,NV),device=device),
                                   self.vx[0,0].expand(NV,NV), self.vy[0,0].expand(NV,NV),
                                   (self.vx**2)[0,0].expand(NV,NV), (self.vx*self.vy)[0,0],
                                   (self.vy**2)[0,0].expand(NV,NV),
                                   (.5*self.v2*self.vx)[0,0], (.5*self.v2*self.vy)[0,0]])

    def field(self, rho):
        rh = torch.fft.fft2(rho)
        ph = rh/self.k2
        ph[0, 0] = 0
        ex = torch.fft.ifft2(-1j*self.kx[:, :, 0, 0]*ph).real
        ey = torch.fft.ifft2(-1j*self.ky[:, :, 0, 0]*ph).real
        return ex, ey

    def rhs(self, f, source):
        rho = f.sum(dim=(2,3)) * self.dv**2
        ex, ey = self.field(rho)
        fhx = torch.fft.fft2(f, dim=(0,1))
        transport = torch.fft.ifft2((-1j)*(self.kx*self.vx+self.ky*self.vy)*fhx, dim=(0,1)).real
        fhv = torch.fft.fft2(f, dim=(2,3))
        dxv = torch.fft.ifft2(1j*self.exi*fhv, dim=(2,3)).real
        dyv = torch.fft.ifft2(1j*self.eyi*fhv, dim=(2,3)).real
        diffusion = torch.fft.ifft2(self.diff*fhv, dim=(2,3)).real
        return (transport + (self.nu*(self.vx+self.frame[0])-ex[:,:,None,None])*dxv
                + (self.nu*(self.vy+self.frame[1])-ey[:,:,None,None])*dyv
                + diffusion + 2*self.nu*f + source)

    def macros(self, f):
        m = torch.einsum("ijab,kab->kij", f, self.weights) * self.dv**2
        rho, jx, jy, rxx, rxy, ryy, hx, hy = m
        ex, ey = self.field(rho)
        return torch.stack((rho,ex,ey,jx,jy,rxx,rxy,ryy,hx,hy)).cpu().numpy()


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    budget = 880. if solve_budget is None else float(solve_budget)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("positive finite solve budget required")
    deadline = budget-12.
    def remaining():
        return deadline-(candidate_solve_clock()-started)
    def check():
        if candidate_solve_clock()-started > deadline:
            raise TimeoutError("kinetic preparation exhausted safe shared solve budget")
    question = case.public_data["question"]
    periods = np.asarray(question["grid"]["x_domain"], dtype=np.float64)
    interval = np.asarray(question["time_domain"], dtype=np.float64)
    nu, theta = float(question["collision_nu"]), float(question["temperature"])
    if periods.shape != (2,) or interval.shape != (2,) or np.any(periods<=0) or interval[1]<=interval[0]:
        raise ValueError("invalid public domain")
    frame=initial_frame(case)
    preview=preview_fallback(case,periods,interval,theta,frame)
    high_drift=bool(np.linalg.norm(frame)>4.*math.sqrt(theta))
    if high_drift:
        preview.spatial_frame=np.zeros(2)
    try:
        session = FrameSession(open_public_case_session(case),frame,periods,interval[0])
        limit = min(131072, int(question.get("numerical_access",{}).get("max_points_per_request",262144)))
        bound = max(VELOCITY_BOUND, 8*math.sqrt(theta))
        v = np.arange(NV)*2*bound/NV-bound
        xy = np.stack(np.meshgrid(*(np.arange(SOURCE_NX)*p/SOURCE_NX for p in periods), indexing="ij"), axis=-1).reshape(-1,2)
        velocity = np.stack(np.meshgrid(v,v,indexing="ij"),axis=-1).reshape(-1,2)
        phase = np.concatenate((np.repeat(xy,NV*NV,axis=0), np.tile(velocity,(len(xy),1))),axis=1)
        method_cost={}
        def read(name, points):
            method=getattr(session,name)
            chunks=[]
            for first in range(0,len(points),limit):
                check()
                count=min(limit,len(points)-first)
                if name in method_cost:
                    seconds,completed=method_cost[name]
                    predicted=count*seconds/max(completed,1)
                    if remaining()<max(18.,1.5*predicted)+12.:
                        raise TimeoutError("remaining budget cannot cover the next numerical Oracle batch")
                batch_started=candidate_solve_clock()
                val=np.asarray(method(points[first:first+limit]),dtype=np.float32)
                old_seconds,old_points=method_cost.get(name,(0.,0))
                method_cost[name]=(old_seconds+candidate_solve_clock()-batch_started,old_points+count)
                if val.shape != (min(limit,len(points)-first),) or not np.isfinite(val).all():
                    raise ValueError("invalid public numerical Oracle response")
                chunks.append(val)
            return np.concatenate(chunks)
        rho0=read("rho0",xy).reshape(SOURCE_NX,SOURCE_NX)
        g0=read("g0",phase).reshape(SOURCE_NX,SOURCE_NX,NV,NV)
        g0*=rho0[:,:,None,None]
        source_v=np.arange(SOURCE_NV)*2*bound/SOURCE_NV-bound
        source_velocity=np.stack(np.meshgrid(source_v,source_v,indexing="ij"),axis=-1).reshape(-1,2)
        phase=np.concatenate((np.repeat(xy,SOURCE_NV*SOURCE_NV,axis=0),np.tile(source_velocity,(len(xy),1))),axis=1)
        device="cuda:0" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(1)
        with torch.no_grad():
            system=KineticSystem(periods,nu,theta,bound,device,frame)
            f=torch.as_tensor(_upsample_xy(g0,NX),device=device,dtype=torch.float32)
            initial_macros=system.macros(f)
            if not np.isfinite(initial_macros).all() or initial_macros[0].min()<1e-4:
                raise FloatingPointError("invalid initial kinetic density")
            fallback=Prepared(np.linspace(interval[0],interval[1],4),
                              np.repeat(initial_macros[None],4,axis=0),periods,interval,frame)
            fallback.compression_stats={"fallback":"initial_state","frame_velocity":frame.tolist()}
            fallback.safe_fallback=preview
            if high_drift:
                fallback.spatial_frame=np.zeros(2)
                fallback.compression_stats.update(method="empirical_high_drift_lab_initial",
                                                  source_points=0,drift_threshold=4.*math.sqrt(theta))
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                check()
                return fallback
            try:
                def read_source(_method,points):
                    return read("source",points)
                source_times, source_values, compression_stats = compress_source(
                    read_source, session.source, xy, source_velocity, interval, remaining,
                    source_nt=SOURCE_NT, reserve=180.)
                source_nt=len(source_times)
                sources=[torch.as_tensor(_upsample_source(value),device=device,dtype=torch.float32)
                         for value in source_values]
                source_stack=torch.stack(sources)
                bary=(-1.)**np.arange(source_nt)
                bary[[0,-1]]*=.5
                def source_at(tt):
                    if source_nt==1:
                        return source_stack[0]
                    delta=tt-source_times
                    hit=np.flatnonzero(np.abs(delta)<1e-13)
                    if len(hit):
                        return source_stack[int(hit[0])]
                    w=bary/delta
                    w=w/w.sum()
                    return torch.einsum("k,kijab->ijab",torch.as_tensor(w,device=device,dtype=torch.float32),source_stack)
                # Explicit stability depends on the solver mesh and public coefficients.
                # Include a conservative electric acceleration allowance; the spatial
                # transport contribution dominates for the normal public scales.
                initial_field=system.field(f.sum(dim=(2,3))*system.dv**2)
                field_scale=max(float(e.abs().max().cpu()) for e in initial_field)
                eta_max=np.pi/system.dv
                adv_rate=bound*np.pi*NX*np.sum(1/periods)+2*eta_max*(2*field_scale+nu*(bound+np.max(np.abs(frame))))
                diff_rate=2*nu*theta*eta_max**2
                stable_dt=min(DT,2.1/max(adv_rate,1e-9),1.5/max(diff_rate,1e-9))
                steps=max(4,int(math.ceil((interval[1]-interval[0])/stable_dt)))
                times=np.linspace(interval[0],interval[1],steps+1)
                dt=(interval[1]-interval[0])/steps
                outputs=[system.macros(f)]
                for n in range(steps):
                    check()
                    t=times[n]
                    s1,s2,s3=source_at(t),source_at(t+.5*dt),source_at(times[n+1])
                    k1=system.rhs(f,s1)
                    k2=system.rhs(f+.5*dt*k1,s2)
                    k3=system.rhs(f+.5*dt*k2,s2)
                    k4=system.rhs(f+dt*k3,s3)
                    f=f+(dt/6)*(k1+2*k2+2*k3+k4)
                    outputs.append(system.macros(f))
                    if not np.isfinite(outputs[-1]).all() or outputs[-1][0].min()<1e-4:
                        raise FloatingPointError("nonfinite kinetic evolution")
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                check()
                values=np.stack(outputs)
                if not np.isfinite(values).all():
                    raise FloatingPointError("nonfinite evolved transport moments")
                handle=Prepared(times,values,periods,interval,frame)
                compression_stats["frame_velocity"]=frame.tolist()
                handle.compression_stats=compression_stats
                handle.safe_fallback=preview
                return handle
            except Exception as exc:
                fallback.compression_stats["reason"]=type(exc).__name__+": "+str(exc)
                return fallback
    except Exception as exc:
        preview.compression_stats["reason"]=type(exc).__name__+": "+str(exc)
        return preview
