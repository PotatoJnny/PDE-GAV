"""Gauss x / Chebyshev source time with full-line conservative CUDA VP.

Velocity acceleration enforces zero-boundary integration by parts for moments
0 through 7 by the quadrature-weighted minimum-L2 correction. No endpoint
trace is extrapolated; initial/source moment truncation is not corrected.
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, fftfreq
from scipy.special import roots_legendre
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import (
    candidate_solve_clock, candidate_solve_remaining_seconds,
)

NX = 512
NG = 1024
NV = 320
SOURCE_DT = 1.0 / 256
BASE_SOURCE_DT = 1.0 / 16
V_SCALE = 4.0
FIELDS = ('M0', 'M1', 'M2', 'M3', 'E')
MIN_INTEGRATION_RESERVE = 60.0
RETURN_RESERVE = 15.0
RK4_CFL = .9
RK4_MAX_DT = .0015


class _GaussFourierProjector:
    """Project public values at Gauss x nodes onto a uniform Fourier panel."""
    def __init__(self,x0,length,nodes=NG,state_size=NX):
        if not torch.cuda.is_available():
            raise RuntimeError('Assigned CUDA device unavailable')
        self.device=torch.device('cuda:0')
        self.state_size=int(state_size)
        z,w=roots_legendre(int(nodes))
        self.x=x0+(length/2)*(z+1)
        integer_modes=np.rint(np.fft.fftfreq(state_size)*state_size)
        phase=np.pi*integer_modes[:,None]*(z[None,:]+1)
        # Mapped Gauss weights divided by L equal w/2. Two real GEMMs
        # avoid promoting every real oracle plane to complex128.
        self.real=torch.as_tensor(np.cos(phase)*(w/2)[None,:],
                                  dtype=torch.float64,device=self.device)
        self.imag=torch.as_tensor(-np.sin(phase)*(w/2)[None,:],
                                  dtype=torch.float64,device=self.device)

    def project(self,values):
        sampled=torch.as_tensor(np.ascontiguousarray(values),
                                dtype=torch.float64,device=self.device)
        coefficients=torch.complex(self.real@sampled,self.imag@sampled)
        coefficients[self.state_size//2]=0
        uniform=torch.fft.ifft(self.state_size*coefficients,dim=0).real
        # Each accepted source plane remains an ordinary host array. This
        # transfer also synchronizes projection before its charged timing.
        return uniform.cpu().numpy()


def _lobatto_times(t0,t1,intervals):
    times=t0+(t1-t0)*(1-np.cos(np.arange(intervals+1)*np.pi/intervals))/2
    times[0]=t0;times[-1]=t1
    if not np.all(np.diff(times)>0):
        raise ValueError('Chebyshev source nodes must be distinct')
    return times


class _CudaForcing:
    def __init__(self, times, values, device):
        self.times=np.array(times,dtype=np.float64,copy=True)
        self.coefficient_count=len(self.times)
        self.shape=values.shape[1:]
        self.barycentric_weights=(-1.)**np.arange(self.coefficient_count)
        self.barycentric_weights[[0,-1]]*=.5
        planes=np.ascontiguousarray(values)
        if not planes.flags.writeable:
            planes=planes.copy()
        self.values=torch.as_tensor(planes,dtype=torch.float64,device=device)
        self.device=device

    def weights_at(self,time):
        if not self.times[0]<=time<=self.times[-1]:
            raise ValueError('Source interpolation leaves the causal prefix')
        delta=time-self.times
        nearest=int(np.argmin(abs(delta)))
        if delta[nearest]==0:
            row=np.zeros(self.coefficient_count)
            row[nearest]=1.
            return row
        # Scale every term by the nearest-node distance to avoid overflow
        # near a node. A shared scale cancels in the barycentric quotient.
        row=self.barycentric_weights*(delta[nearest]/delta)
        return row/np.sum(row)

    def stages(self,times):
        matrix=torch.as_tensor(np.stack([self.weights_at(t) for t in times]),
                               dtype=torch.float64,device=self.device)
        planes=matrix @ self.values.reshape(self.coefficient_count,-1)
        return planes.reshape((len(times),)+self.shape)

    def benchmark(self,t0,t1):
        times=tuple(t0+(t1-t0)*fraction for fraction in (.113,.517,.883))
        with torch.no_grad():
            for _ in range(2):
                probe=self.stages(times)
            torch.cuda.synchronize(self.device)
            started=candidate_solve_clock()
            for _ in range(8):
                probe=self.stages(times)
            torch.cuda.synchronize(self.device)
        return max(0.,candidate_solve_clock()-started)/8


class _CudaIntegrator:
    def __init__(self,f,v,jac,weights,quadrature,ikx,ikv,invk):
        if not torch.cuda.is_available():
            raise RuntimeError('Assigned CUDA device unavailable')
        self.device=torch.device('cuda:0')
        def gpu(value):
            return torch.as_tensor(np.ascontiguousarray(value),device=self.device)
        self.velocity=gpu(v)[None,:]
        self.jacobian=gpu(jac)[None,:]
        self.weights=gpu(weights)
        self.quadrature=gpu(quadrature)
        # Condition eight polynomial constraints using a Legendre basis.
        # The lift minimizes the quadrature-weighted L2 derivative change.
        center=(float(np.min(v))+float(np.max(v)))/2
        scale=float(np.max(abs(v-center)))
        z=(v-center)/scale
        polynomials=np.polynomial.legendre.legvander(z,7)
        moment_weights=weights[:,None]*polynomials
        gram=polynomials.T@moment_weights
        lift=np.linalg.solve(gram,polynomials.T).T
        self.ibp_weights=gpu(moment_weights)
        # d P_j(z)/dv = sum_(i<j, j-i odd) (2*i+1)/scale * P_i(z).
        generator=np.zeros((8,8))
        for j in range(8):
            for i in range(j-1,-1,-2):
                generator[i,j]=-(2*i+1)/scale
        self.ibp_generator=gpu(generator)
        self.ibp_lift=gpu(lift)
        self.ikx=gpu(ikx)[:,None]
        self.ikv=gpu(ikv)[None,:]
        self.invk=gpu(invk)
        integer_modes=np.rint(np.fft.fftfreq(len(ikx))*len(ikx))
        self.mask=gpu((abs(integer_modes)<len(ikx)/3).astype(float))[:,None]
        self.state=self.project(gpu(np.asarray(f,dtype=np.float64)))
        self.forcing=None
        self.advective_rate=float(np.max(abs(ikx))*np.max(abs(v)))
        self.acceleration_factor=float(np.max(abs(ikv))/np.min(jac))

    def project(self,state):
        return torch.fft.ifft(torch.fft.fft(state,dim=0)*self.mask,dim=0).real

    def field(self,state):
        density=state @ self.weights
        return torch.fft.ifft(torch.fft.fft(density)*self.invk).real

    def rhs(self,state,source,electric=None):
        if electric is None:
            electric=self.field(state)
        dx=torch.fft.ifft(self.ikx*torch.fft.fft(state,dim=0),dim=0).real
        dv=torch.fft.ifft(self.ikv*torch.fft.fft(state,dim=1),dim=1).real/self.jacobian
        target=(state@self.ibp_weights)@self.ibp_generator
        dv=dv+(target-dv@self.ibp_weights)@self.ibp_lift.T
        # Project the complete source-plus-transport RHS to the strict 2/3 band.
        return self.project(source-self.velocity*dx-electric[:,None]*dv)

    def macros(self,state):
        return torch.cat((state @ self.quadrature,self.field(state)[:,None]),dim=1)

    def step(self,state,t,end,cap=None):
        electric=self.field(state)
        force_max=float(torch.max(torch.abs(electric)).item())
        if not np.isfinite(force_max):
            raise FloatingPointError('CUDA kinetic field became nonfinite')
        h=min(RK4_MAX_DT,RK4_CFL/(self.advective_rate
                  +force_max*self.acceleration_factor+1.),end-t)
        if cap is not None:
            h=min(h,cap)
        source=self.forcing.stages((t,t+h/2,min(end,t+h)))
        k1=self.rhs(state,source[0],electric)
        k2=self.rhs(state+h*k1/2,source[1])
        k3=self.rhs(state+h*k2/2,source[1])
        k4=self.rhs(state+h*k3,source[2])
        return state+(h/6)*(k1+2*k2+2*k3+k4),h

    def benchmark(self,t0,t1):
        # Warm FFT kernels and measure complete GPU RK4 steps, including
        # source evaluation, synchronization, and host launch overhead.
        probe=self.state.clone();t=float(t0);cap=(t1-t0)/16
        with torch.no_grad():
            for _ in range(2):
                probe,h=self.step(probe,t,t1,cap);t+=h
            torch.cuda.synchronize(self.device)
            started=candidate_solve_clock()
            for _ in range(4):
                probe,h=self.step(probe,t,t1,cap);t+=h
            torch.cuda.synchronize(self.device)
            seconds=max(0.,candidate_solve_clock()-started)/4
        return seconds

    def integrate(self,t0,t1,remaining):
        times=[float(t0)];values=[self.macros(self.state)]
        t=float(t0);steps=0;state=self.state
        with torch.no_grad():
            while t<t1:
                state,h=self.step(state,t,t1)
                t=min(float(t1),t+h);steps+=1
                if steps%4==0 or t==t1:
                    if not bool(torch.isfinite(state).all().item()):
                        raise FloatingPointError('CUDA kinetic integration became nonfinite')
                    times.append(t);values.append(self.macros(state))
                if remaining()<=RETURN_RESERVE:
                    raise TimeoutError('CUDA kinetic integration exhausted shared budget')
            host=torch.stack(values).cpu().numpy()
            torch.cuda.synchronize(self.device)
        return times,host,steps


class Solution:
    def __init__(self, times, macros, x0, length):
        self.times = np.asarray(times)
        self.modes = np.fft.fft(np.asarray(macros), axis=1) / macros[0].shape[0]
        self.x0, self.length = x0, length
        self.ks = 2*np.pi*np.fft.fftfreq(self.modes.shape[1], d=length/self.modes.shape[1])
        self.spline = CubicSpline(self.times, self.modes, axis=0)

    def evaluate(self, query):
        if query.kind is not QueryKind.POINTS:
            raise ValueError('POINTS query required')
        t = np.asarray([p['t'] for p in query.points])
        x = np.asarray([p['x'] for p in query.points])
        modes = self.spline(t)
        values = np.einsum('nk,nkf->nf', np.exp(1j*(x-self.x0)[:,None]*self.ks), modes).real
        return AnswerView(kind=query.kind, fields={k:tuple(values[:,i]) for i,k in enumerate(FIELDS)}, points=query.points, shape=(len(t),))


def prepare_solution(case, solve_budget=None):
    began = candidate_solve_clock()
    budget = float(solve_budget) if solve_budget is not None else 900.0
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('solve_budget must be finite and positive')
    def remaining():
        local = budget-(candidate_solve_clock()-began)
        upstream = candidate_solve_remaining_seconds()
        return local if upstream is None else min(local, upstream)
    equation = case.public_data['equation']
    domain = equation['domain']
    x0, x1 = domain['x_interval']; t0,t1 = domain['time_interval']
    v0,v1 = domain['velocity_query_window']
    length = x1-x0
    projector = _GaussFourierProjector(x0,length,NG,NX)
    x = projector.x
    # Map a uniform periodic computational grid onto the finite query window.
    vc = (v0+v1)/2; half = (v1-v0)/2
    extent = np.arcsinh(half/V_SCALE)
    ds = 2*extent/NV
    z = -extent + ds*np.arange(NV)
    v = np.clip(vc + V_SCALE*np.sinh(z),v0,v1)
    jac = V_SCALE*np.cosh(z)
    weights = jac*ds
    quadrature = np.stack([weights*v**i for i in range(4)], axis=1)
    ikx = 1j*2*np.pi*fftfreq(NX, d=length/NX)
    ikv = 1j*2*np.pi*fftfreq(NV, d=ds)
    ikx[NX//2] = 0; ikv[NV//2] = 0
    invk = np.zeros(NX, dtype=complex)
    invk[1:] = 1/(1j*2*np.pi*fftfreq(NX, d=length/NX)[1:])
    invk[NX//2] = 0
    limit = int(equation['oracle_protocol']['limits']['max_points_per_request'])
    session = open_public_case_session(case)
    xv = np.column_stack((np.repeat(x,NV), np.tile(v,NG)))
    sampling_seconds = 0.0
    sampling_points = 0
    recent_rates = []
    projection_seconds = 0.0
    projection_planes = 0
    recent_projection_seconds = []
    def observed_point_cost():
        average = sampling_seconds/max(1, sampling_points)
        return max([average]+recent_rates)
    def observed_projection_cost():
        average=projection_seconds/max(1,projection_planes)
        return max([average]+recent_projection_seconds)
    def project_plane(values,initial=False):
        nonlocal projection_seconds,projection_planes
        started=candidate_solve_clock()
        projected=projector.project(values.reshape(NG,NV))
        elapsed=max(0.,candidate_solve_clock()-started)
        if not initial:
            projection_seconds+=elapsed
            projection_planes+=1
            recent_projection_seconds.append(elapsed)
            del recent_projection_seconds[:-4]
        return projected
    class StopRefinement(Exception):
        pass
    def sample(points, initial=False, reserve=MIN_INTEGRATION_RESERVE+RETURN_RESERVE,
               optional=False):
        nonlocal sampling_seconds, sampling_points
        out = np.empty(len(points))
        method = session.query_initial_distribution if initial else session.query_source
        for start in range(0,len(points),limit):
            chunk = points[start:start+limit]
            forecast = 1.5*observed_point_cost()*len(chunk)
            if remaining() <= reserve+forecast:
                if optional:
                    raise StopRefinement()
                raise TimeoutError('Insufficient budget to finish mandatory coarse source panel')
            call_started = candidate_solve_clock()
            values = np.asarray(method(chunk), dtype=float)
            charged = max(0.0, candidate_solve_clock()-call_started)
            if values.shape != (len(chunk),) or not np.isfinite(values).all():
                raise ValueError('Invalid public oracle values')
            out[start:start+len(chunk)] = values
            # Source and f0 can have very different execution costs. Forecast
            # source refinement only from measured source calls.
            if not initial:
                sampling_seconds += charged
                sampling_points += len(chunk)
                recent_rates.append(charged/len(chunk))
                del recent_rates[:-4]
        return out
    f = project_plane(sample(xv, True),initial=True)
    duration = t1-t0
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Positive finite time interval required')
    nt = max(1,int(np.ceil(duration/BASE_SOURCE_DT)))+1
    st = _lobatto_times(t0,t1,nt-1)
    source = np.empty((nt,NX,NV))
    points = np.empty((NG*NV,3)); points[:,1:] = xv
    for i,t in enumerate(st):
        points[:,0] = t
        source[i] = project_plane(sample(points))
    source.setflags(write=False)
    integrator = _CudaIntegrator(f,v,jac,weights,quadrature,ikx,ikv,invk)
    torch.cuda.synchronize(integrator.device)
    spline_started = candidate_solve_clock()
    forcing = _CudaForcing(st,source,integrator.device)
    integrator.forcing = forcing
    torch.cuda.synchronize(integrator.device)
    coarse_spline_seconds = max(0.0, candidate_solve_clock()-spline_started)
    coarse_nodes = len(st)
    # Benchmark complete synchronized GPU RK4 steps after a COMPLETE coarse
    # source panel exists. No numerical oracle is called during this benchmark.
    gpu_step_seconds = integrator.benchmark(t0,t1)
    gpu_forcing_seconds = forcing.benchmark(t0,t1)
    gpu_other_step_seconds = max(0.,gpu_step_seconds-gpu_forcing_seconds)
    advective_rate = np.max(abs(ikx))*np.max(abs(v))
    acceleration_factor = np.max(abs(ikv))/np.min(jac)
    initial_field_max = float(torch.max(abs(integrator.field(integrator.state))).item())
    initial_current_max = np.max(abs(f@(weights*v)))
    def estimate_steps(values):
        density_forcing = values@weights
        source_field = ifft(fft(density_forcing,axis=1)*invk[None,:],axis=1).real
        field_bound = max(1.0, 2*initial_field_max
                          + duration*(initial_current_max+np.max(abs(source_field))))
        estimated_h = min(RK4_MAX_DT, RK4_CFL/(advective_rate+field_bound*acceleration_factor+1))
        return max(1,int(np.ceil(duration/estimated_h)))
    estimated_steps_bound = estimate_steps(source)
    def integration_reserve(nodes):
        # Global barycentric evaluation reads every accepted source plane.
        # Forecast its measured cost linearly with the prospective node count.
        step_cost=gpu_other_step_seconds+gpu_forcing_seconds*nodes/coarse_nodes
        return max(MIN_INTEGRATION_RESERVE, 2*estimated_steps_bound*1.15*step_cost)
    def reserve_for(nodes):
        interpolation_reserve = max(3.0, 2*coarse_spline_seconds*nodes/coarse_nodes)
        return integration_reserve(nodes)+interpolation_reserve+RETURN_RESERVE
    attempted_refinements = 0
    discarded_refinements = 0
    target_intervals = max(len(st)-1,int(np.ceil(duration/SOURCE_DT)))
    while len(st)-1 < target_intervals:
        proposed_nodes = 2*len(st)-1
        new_planes = len(st)-1
        new_points = new_planes*NG*NV
        # Whole-level admission needs only a modest margin: every request
        # still checks a 1.5x forecast and preserves the complete accepted
        # panel plus measured integration/return reserves if service slows.
        forecast = 1.1*(observed_point_cost()*new_points
                        +observed_projection_cost()*new_planes)
        if remaining() <= reserve_for(proposed_nodes)+forecast:
            break
        attempted_refinements += 1
        proposed_times = _lobatto_times(t0,t1,proposed_nodes-1)
        proposed_times[::2] = st
        proposed_source = np.empty((proposed_nodes,NX,NV))
        proposed_source[::2] = source
        try:
            for i in range(1,proposed_nodes,2):
                points[:,0] = proposed_times[i]
                plane = sample(points, reserve=reserve_for(proposed_nodes)
                                +1.5*observed_projection_cost(),optional=True)
                proposed_source[i] = project_plane(plane)
        except StopRefinement:
            # Keep the previous full-horizon panel. No incomplete refinement
            # or partially integrated solution can be returned.
            discarded_refinements += 1
            del proposed_source, proposed_times
            break
        proposed_source.setflags(write=False)
        source, st = proposed_source, proposed_times
        del proposed_source, proposed_times
        forcing = None
        integrator.forcing = None
        estimated_steps_bound = max(estimated_steps_bound,estimate_steps(source))
    accepted_source_nodes = len(st)
    if forcing is None:
        forcing = _CudaForcing(st,source,integrator.device)
        integrator.forcing = forcing
        torch.cuda.synchronize(integrator.device)
    del source, forcing
    remaining_at_integration = remaining()
    times,values,steps = integrator.integrate(t0,t1,remaining)
    del integrator
    solution = Solution(times,values,x0,length)
    solution._diagnostics = {
        'spatial_oracle_nodes': NG,
        'spatial_fourier_nodes': NX,
        'spatial_projection_charged_seconds': projection_seconds,
        'source_nodes': accepted_source_nodes,
        'attempted_refinements': attempted_refinements,
        'discarded_refinements': discarded_refinements,
        'source_sampling_charged_seconds': sampling_seconds,
        'source_sampling_points': sampling_points,
        'integration_reserve_seconds': integration_reserve(accepted_source_nodes),
        'remaining_at_integration': remaining_at_integration,
        'gpu_rk4_step_benchmark_seconds': gpu_step_seconds,
        'gpu_forcing_stage_benchmark_seconds': gpu_forcing_seconds,
        'integration_steps': steps,
    }
    return solution
