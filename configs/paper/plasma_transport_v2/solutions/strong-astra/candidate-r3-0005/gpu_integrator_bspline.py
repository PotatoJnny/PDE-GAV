"""Optional CUDA Fourier--Hermite RK4 integration, with no Oracle access.

The argument source_coefficients is make_interp_spline(..., k=3).c,
and source_times is its knot vector .t. Pass the original sample times as
time_knots to retain every source sample as an integration boundary.
All arrays entering and leaving this function are ordinary NumPy host arrays. The CUDA graph is an optimization
of the same four-stage formula, not a different time integrator.
Every source knot is an integration boundary and a saved projection time.
"""
from __future__ import annotations

import math
import numpy as np
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


def integrate_gpu(initial, source_coefficients, source_times, *, n, degree,
                  variance, theta, nu, periods, t_interval, solve_budget=None,
                  started=None, max_dt=None, save_every=1, use_graph=True,
                  time_knots=None, source_degree=3):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA integration requested without an assigned CUDA device")
    if started is None:
        started = candidate_solve_clock()
    budget = 850. if solve_budget is None else float(solve_budget)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("solve_budget must be finite and positive")
    t0, t1 = map(float, t_interval)
    if not (t1 > t0 and variance > 0 and theta > 0 and nu >= 0):
        raise ValueError("invalid integration constants")
    indices = [(a, d-a) for d in range(degree+1) for a in range(d,-1,-1)]
    index = {ab:i for i,ab in enumerate(indices)}
    count = len(indices)
    if np.shape(initial) != (count,n,n):
        raise ValueError("initial Fourier array has the wrong shape")
    spline_degree = int(source_degree)
    if spline_degree not in (3,5):
        raise ValueError("source spline degree must be three or five")
    coefficient_count = len(source_times)-spline_degree-1
    if np.shape(source_coefficients) != (coefficient_count,count,n,n):
        raise ValueError("cubic B-spline coefficients have the wrong shape")
    if coefficient_count < spline_degree+1:
        raise ValueError("cubic B-spline requires at least four coefficients")
    if t0 < source_times[spline_degree] or t1 > source_times[-spline_degree-1]:
        raise ValueError("integration interval leaves the B-spline source domain")
    device = torch.device("cuda:0")
    complex_dtype = torch.complex128
    real_dtype = torch.float64

    def real(value):
        return torch.as_tensor(value,dtype=real_dtype,device=device)

    def integer(value):
        return torch.as_tensor(value,dtype=torch.long,device=device)

    def check_budget():
        torch.cuda.synchronize(device)
        if candidate_solve_clock() - started > budget - 20.:
            raise TimeoutError("CUDA VPFP integration exhausted remaining Case budget")

    check_budget()
    state = torch.as_tensor(np.asarray(initial),dtype=complex_dtype,device=device).clone()
    initial_gpu = state.clone()
    coefficients = torch.as_tensor(np.asarray(source_coefficients),dtype=complex_dtype,device=device)
    knots = real(source_times)
    clock = real(t0)
    ax = np.array([a for a,b in indices])
    ay = np.array([b for a,b in indices])
    xm = integer([index.get((a-1,b),count) for a,b in indices])
    ym = integer([index.get((a,b-1),count) for a,b in indices])
    xp = integer([index.get((a+1,b),count) for a,b in indices])
    yp = integer([index.get((a,b+1),count) for a,b in indices])
    xmm = integer([index.get((a-2,b),count) for a,b in indices])
    ymm = integer([index.get((a,b-2),count) for a,b in indices])
    sx, sy = real(np.sqrt(ax)[:,None,None]), real(np.sqrt(ay)[:,None,None])
    sxp, syp = real(np.sqrt(ax+1)[:,None,None]), real(np.sqrt(ay+1)[:,None,None])
    damp = real((-nu*(ax+ay))[:,None,None])
    cx = real((nu*(theta/variance-1)*np.sqrt(ax*np.maximum(ax-1,0)))[:,None,None])
    cy = real((nu*(theta/variance-1)*np.sqrt(ay*np.maximum(ay-1,0)))[:,None,None])
    freq = np.fft.fftfreq(n)*n
    kx_np,ky_np = np.meshgrid(2*np.pi*freq/periods[0],2*np.pi*freq/periods[1],indexing="ij")
    k2 = kx_np*kx_np + ky_np*ky_np
    px_np = np.divide(-1j*kx_np,k2,out=np.zeros_like(k2,dtype=complex),where=k2>0)
    py_np = np.divide(-1j*ky_np,k2,out=np.zeros_like(k2,dtype=complex),where=k2>0)
    kx,ky = real(kx_np),real(ky_np)
    px = torch.as_tensor(px_np,dtype=complex_dtype,device=device)
    py = torch.as_tensor(py_np,dtype=complex_dtype,device=device)
    padding = torch.zeros((1,n,n),dtype=complex_dtype,device=device)
    scale = math.sqrt(variance)
    if max_dt is None:
        largest_axis = max(float(np.max(abs(kx_np))),float(np.max(abs(ky_np))))
        max_dt = min(.0015,.35/max(nu*degree+scale*largest_axis*math.sqrt(degree),1.))
    if not math.isfinite(max_dt) or max_dt <= 0:
        raise ValueError("max_dt must be finite and positive")
    max_dt = min(float(max_dt),(t1-t0)/128)
    boundaries = np.asarray(source_times if time_knots is None else time_knots,dtype=float)
    if boundaries.ndim != 1 or not np.isfinite(boundaries).all():
        raise ValueError("integration time knots must be a finite vector")
    boundaries = boundaries[(boundaries>t0)&(boundaries<t1)]
    # Always retain source knots, including when additional knots are supplied.
    source_boundaries = np.asarray(source_times,dtype=float)
    source_boundaries = source_boundaries[(source_boundaries>t0)&(source_boundaries<t1)]
    boundaries = np.unique(np.concatenate(([t0],boundaries,source_boundaries,[t1])))
    segments = []
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        microsteps = max(1,int(math.ceil((right-left)/max_dt)))
        segments.append((float(left),float(right),microsteps,float((right-left)/microsteps)))
    step_dt = real(segments[0][3])
    save_every = max(1,int(save_every))

    offsets = integer(np.arange(-spline_degree,1))
    integer_offsets = integer(np.arange(1-spline_degree,spline_degree+1))

    def source_at(t):
        # de Boor's cubic recurrence needs four coefficient fields, compared
        # with four fields per interval in a power-basis cubic representation.
        slot = torch.clamp(torch.searchsorted(knots,t,right=True)-1,spline_degree,coefficient_count-1)
        c = torch.index_select(coefficients,0,slot+offsets)
        local_knots = torch.index_select(knots,0,slot+integer_offsets)
        if spline_degree == 5:
            # de Boor recurrence, fixed-size unrolled by CUDA graph capture.
            values = [c[j] for j in range(6)]
            for level in range(1,6):
                for j in range(5,level-1,-1):
                    left = local_knots[j-1]
                    right = local_knots[j+5-level]
                    alpha = (t-left)/(right-left)
                    values[j] = values[j-1]+alpha*(values[j]-values[j-1])
            return values[5]
        # local_knots has indices i-2, i-1, i, i+1, i+2, i+3.
        a1 = (t-local_knots[0])/(local_knots[3]-local_knots[0])
        a2 = (t-local_knots[1])/(local_knots[4]-local_knots[1])
        a3 = (t-local_knots[2])/(local_knots[5]-local_knots[2])
        d1 = c[0]+a1*(c[1]-c[0])
        d2 = c[1]+a2*(c[2]-c[1])
        d3 = c[2]+a3*(c[3]-c[2])
        b2 = (t-local_knots[1])/(local_knots[3]-local_knots[1])
        b3 = (t-local_knots[2])/(local_knots[4]-local_knots[2])
        e2 = d1+b2*(d2-d1)
        e3 = d2+b3*(d3-d2)
        gamma = (t-local_knots[2])/(local_knots[3]-local_knots[2])
        return e2+gamma*(e3-e2)

    def rhs(c,source):
        padded = torch.cat((c,padding),dim=0)
        linear = -1j*scale*(kx*(sxp*padded[xp]+sx*padded[xm])
                           +ky*(syp*padded[yp]+sy*padded[ym]))
        linear = linear+damp*c+cx*padded[xmm]+cy*padded[ymm]+source
        physical = torch.fft.ifft2(padded,dim=(-2,-1)).real
        ex = torch.fft.ifft2(px*c[0]).real
        ey = torch.fft.ifft2(py*c[0]).real
        force = (sx*physical[xm]*ex+sy*physical[ym]*ey)/scale
        return linear+torch.fft.fft2(force,dim=(-2,-1))

    def step():
        s0 = source_at(clock)
        sm = source_at(clock+.5*step_dt)
        s1 = source_at(clock+step_dt)
        a = rhs(state,s0)
        b = rhs(state+.5*step_dt*a,sm)
        c = rhs(state+.5*step_dt*b,sm)
        d = rhs(state+step_dt*c,s1)
        state.add_((step_dt/6)*(a+2*b+2*c+d))
        state[0,0,0].fill_(n*n)
        clock.add_(step_dt)

    graph = None
    if use_graph:
        # Warm cuFFT plans and allocator paths on a side stream before capture.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream(device).wait_stream(stream)
        state.copy_(initial_gpu)
        clock.fill_(t0)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        state.copy_(initial_gpu)
        clock.fill_(t0)
    check_budget()
    save_times = [t0]
    saved = [state[:10].clone()]
    total_steps = 0
    for left,right,microsteps,dt in segments:
        # These device updates preserve graph addresses and reset clock drift.
        step_dt.fill_(dt)
        clock.fill_(left)
        for it in range(microsteps):
            if graph is None:
                step()
            else:
                graph.replay()
            if (it+1)%save_every == 0 or it == microsteps-1:
                save_times.append(right if it == microsteps-1 else left+(it+1)*dt)
                saved.append(state[:10].clone())
            if total_steps%32 == 0:
                check_budget()
            total_steps += 1
    result = torch.stack(saved).cpu().numpy()
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite CUDA Hermite solve")
    save_times[-1] = t1
    check_budget()
    return np.asarray(save_times),result
