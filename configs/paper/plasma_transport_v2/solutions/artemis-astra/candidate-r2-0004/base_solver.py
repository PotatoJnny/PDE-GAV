"""DG velocity with fine-quadrature L2 data projection, Fourier x, ETDRK4."""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline, make_interp_spline
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

FIELDS = ("M0", "M1", "M2", "M3", "E")
NX = 1024
NX_QUERY = 1024
NCELLS = 160
NGAUSS = 6
NGAUSS_QUERY = 7
NT = 65
DT = 0.0025
GRID_UNIFORM_FRACTION = 0.25
MIN_CELL_WIDTH_FRACTION = 0.03
SOURCE_RESERVE = 100.0
COARSE_SOURCE_NODES = 17
EVOLUTION_RESERVE_FACTOR = 1.5
RETURN_RESERVE = 20.
STREAMING_PHASE_LIMIT = 3.0
MAX_STEPS = 100000
SAVE_DT = 0.001
PILOT_BUDGET = 30.
PILOT_X_COUNT = 8
PILOT_FLOORS = np.array([1.0041253663782175, .4788345668393978,
                        2.930763214320111, 1.9801548942236534])



class _SamplingReserve(Exception):
    pass


def _phi(z, order):
    """Stable entire phi functions for imaginary ETDRK4 frequencies."""
    large = np.abs(z) >= 1.
    small_z = np.where(large, 0., z)
    factorial = 1.
    for j in range(2, order+1):
        factorial *= j
    term = np.full(z.shape, 1./factorial, dtype=np.complex128)
    value = term.copy()
    for m in range(1, 32):
        term = term*small_z/(m+order)
        value += term
        if np.max(np.abs(term)) < 1e-18:
            break
    if np.any(large):
        zz = z[large]
        direct = np.expm1(zz)/zz
        denominator = 1.
        for j in range(1, order):
            denominator *= j
            direct = (direct-1./denominator)/zz
        value[large] = direct
    return value


class Prepared:
    def __init__(self, times, modes, xmin, length):
        self.times = np.asarray(times)
        self.modes = np.asarray(modes)
        self.xmin = xmin
        self.length = length

    def evaluate(self, query):
        if query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS required")
        ts = np.asarray([p["t"] for p in query.points], dtype=float)
        xs = np.asarray([p["x"] for p in query.points], dtype=float)
        ks = np.fft.fftfreq(self.modes.shape[-1], 1 / self.modes.shape[-1])
        phase = np.exp(2j * np.pi * (xs[:, None] - self.xmin) / self.length * ks)
        ix = np.clip(np.searchsorted(self.times, ts, side="right") - 1, 0, len(self.times) - 2)
        a = np.clip((ts - self.times[ix]) / (self.times[ix+1] - self.times[ix]), 0, 1)
        coeff = (1-a[:, None, None])*self.modes[ix]+a[:, None, None]*self.modes[ix+1]
        vals = np.einsum("pfk,pk->pf", coeff, phase).real
        return AnswerView(kind=query.kind, fields={name: tuple(vals[:, i]) for i, name in enumerate(FIELDS)},
                          points=query.points, shape=(len(query.points),))


def select_velocity_grid(case, ncells=None):
    """Convenience values choose cells only; every solve value is queried exactly."""
    domain = case.public_data["equation"]["domain"]
    ncells = NCELLS if ncells is None else ncells
    vmin, vmax = map(float, domain["velocity_query_window"])
    tmin, tmax = map(float, domain["time_interval"])
    arrays = case.public_data.get("arrays", {})
    native_v = np.asarray(arrays.get("v", ()), dtype=float).ravel()
    if len(native_v) < 4 or not np.isfinite(native_v).all():
        return np.linspace(vmin, vmax, ncells+1), False
    order = np.argsort(native_v)
    native_v = native_v[order]
    if np.any(np.diff(native_v) <= 0):
        return np.linspace(vmin, vmax, ncells+1), False
    profiles = []
    for name in ("f0", "source"):
        values = np.asarray(arrays.get(name, ()), dtype=float)
        if values.ndim < 2 or values.shape[-1] != len(native_v):
            continue
        if name == "source" and "t" in arrays:
            public_t = np.asarray(arrays["t"], dtype=float).ravel()
            if values.shape[0] == len(public_t):
                values = values[public_t <= tmax+1e-12]
        if values.size:
            profile = np.sqrt(np.mean(np.nan_to_num(values).reshape(-1, len(native_v))**2, axis=0))[order]
            if name == "source":
                profile *= max(tmax-tmin, .01)
            if np.max(profile) > 1e-30:
                profiles.append(profile)
    if not profiles:
        return np.linspace(vmin, vmax, ncells+1), False
    monitor = np.zeros_like(native_v)
    components = 0
    for profile in profiles:
        for power in range(4):
            component = np.sqrt(np.maximum(profile, 0.)*(1+np.abs(native_v)**power))
            integral = np.trapz(component, native_v)
            if integral > 1e-30 and np.isfinite(integral):
                monitor += component/integral
                components += 1
    if not components:
        return np.linspace(vmin, vmax, ncells+1), False
    fine_v = np.linspace(vmin, vmax, max(4097, min(32769, 8*len(native_v)+1)))
    monitor = np.interp(fine_v, native_v, monitor/components, left=0., right=0.)
    monitor /= np.trapz(monitor, fine_v)
    monitor = GRID_UNIFORM_FRACTION/(vmax-vmin)+(1-GRID_UNIFORM_FRACTION)*monitor
    # Capping density rather than shifting edges preserves narrow-feature positions.
    cap = 1./((vmax-vmin)*MIN_CELL_WIDTH_FRACTION)
    lo, hi = 0., 1.
    while np.trapz(np.minimum(cap, hi*monitor), fine_v) < 1.:
        hi *= 2
    for _ in range(50):
        mid = (lo+hi)/2
        if np.trapz(np.minimum(cap, mid*monitor), fine_v) < 1.:
            lo = mid
        else:
            hi = mid
    monitor = np.minimum(cap, hi*monitor)
    cdf = np.r_[0., np.cumsum((monitor[1:]+monitor[:-1])*np.diff(fine_v)/2)]
    cdf /= cdf[-1]
    edges = np.interp(np.linspace(0., 1., ncells+1), cdf, fine_v)
    edges[0], edges[-1] = vmin, vmax
    return edges, True

def refine_velocity_grid(case, query, began, allowance):
    """Use exact f0/source probes solely to select a quadrature-efficient grid."""
    fallback, guided = select_velocity_grid(case)
    start, _ = select_velocity_grid(case, max(2, NCELLS//2))
    domain = case.public_data["equation"]["domain"]
    xmin, xmax = map(float, domain["x_interval"])
    tmin, tmax = map(float, domain["time_interval"])
    horizon = tmax-tmin
    x = xmin+(xmax-xmin)*(np.arange(PILOT_X_COUNT)+.173)/PILOT_X_COUNT
    pilot_start = candidate_solve_clock()
    pilot_limit = min(PILOT_BUDGET, max(0., .1*allowance))
    count = 0
    trace = []

    def integrals(intervals):
        nonlocal count
        geometry = []
        for order in (NGAUSS, 2*NGAUSS):
            xi, w = np.polynomial.legendre.leggauss(order)
            half = (intervals[:, 1]-intervals[:, 0])/2
            velocity = intervals.mean(axis=1)[:, None]+half[:, None]*xi
            weight = half[:, None]*w
            geometry.append((velocity, weight))
        velocities = np.concatenate([item[0].ravel() for item in geometry])
        xx, vv = np.repeat(x, len(velocities)), np.tile(velocities, len(x))
        values = []
        for time in (None, tmin+.5*horizon, tmax):
            if candidate_solve_clock()-pilot_start >= pilot_limit:
                raise _SamplingReserve("Velocity pilot reached its budget")
            points = (np.column_stack((xx, vv)) if time is None else
                      np.column_stack((np.full(len(xx), time), xx, vv)))
            # Reserve the unspent case time beyond the pilot's small allowance.
            reserve = max(SOURCE_RESERVE, allowance-(pilot_start-began)-pilot_limit)
            value = query(points, initial=time is None, reserve=reserve).reshape(len(x), -1)
            count += len(points)
            values.append(value if time is None else horizon*value)
        values = np.concatenate(values, axis=0)
        moments = []
        absolute = None
        offset = 0
        for velocity, weight in geometry:
            size = velocity.size
            part = values[:, offset:offset+size].reshape(len(values), *velocity.shape)
            weighted = weight[None, :, :]*velocity[None, :, :]**np.arange(4)[:, None, None]
            moments.append(np.einsum("scn,pcn->scp", part, weighted))
            absolute = np.einsum("scn,pcn->scp", np.abs(part), np.abs(weighted))
            offset += size
        if not all(np.isfinite(item).all() for item in (*moments, absolute)):
            raise _SamplingReserve("Nonfinite velocity pilot")
        return moments[0], moments[1], absolute

    intervals = np.column_stack((start[:-1], start[1:]))
    try:
        low, high, absolute = integrals(intervals)
        while len(intervals) < NCELLS:
            scales = np.maximum(np.max(np.sum(absolute, axis=1), axis=0), PILOT_FLOORS)
            errors = np.max(np.abs(high-low)/scales[None, None, :], axis=(0, 2))
            add = min(max(1, NCELLS//16), NCELLS-len(intervals))
            selected = np.argpartition(errors, -add)[-add:]
            trace.append((len(intervals), float(np.max(errors)), float(np.sum(errors))))
            retained = np.ones(len(intervals), dtype=bool)
            retained[selected] = False
            chosen = intervals[selected]
            middle = chosen.mean(axis=1)
            children = np.concatenate((np.column_stack((chosen[:, 0], middle)),
                                       np.column_stack((middle, chosen[:, 1]))), axis=0)
            intervals = np.concatenate((intervals[retained], children), axis=0)
            if len(intervals) == NCELLS:
                break
            child_low, child_high, child_absolute = integrals(children)
            low = np.concatenate((low[:, retained], child_low), axis=1)
            high = np.concatenate((high[:, retained], child_high), axis=1)
            absolute = np.concatenate((absolute[:, retained], child_absolute), axis=1)
    except _SamplingReserve:
        return fallback, guided, dict(completed=False, points=count, trace=trace)
    edges = np.sort(np.r_[intervals[:, 0], np.max(intervals[:, 1])])
    return edges, guided, dict(completed=True, points=count, trace=trace)

class DGVelocity:
    def __init__(self, edges, device):
        if NGAUSS not in (4, 6, 8):
            raise ValueError("NGAUSS must be 4, 6, or 8")
        xi, gauss_weights = np.polynomial.legendre.leggauss(NGAUSS)
        differences = xi[:, None]-xi[None, :]
        np.fill_diagonal(differences, 1.)
        bary = 1./np.prod(differences, axis=1)
        derivative = (bary[None, :]/bary[:, None])/differences
        np.fill_diagonal(derivative, 0.)
        np.fill_diagonal(derivative, -np.sum(derivative, axis=1))
        left = bary/(-1.-xi); left /= np.sum(left)
        right = bary/(1.-xi); right /= np.sum(right)
        widths = np.diff(edges)
        self.nodes = ((edges[:-1, None]+edges[1:, None])/2+widths[:, None]*xi/2).ravel()
        self.quadrature = (widths[:, None]*gauss_weights/2).ravel()
        self.min_width = float(np.min(widths))
        self.cfl = {4: .070, 6: .035, 8: .022}[NGAUSS]
        self.edges = edges
        tensor = lambda values: torch.as_tensor(values, dtype=torch.float64, device=device)
        self.derivative = tensor(derivative)
        self.left, self.right = tensor(left), tensor(right)
        self.lift_left, self.lift_right = tensor(left/gauss_weights), tensor(right/gauss_weights)
        self.inverse_jacobian = tensor(2./widths)[None, :, None]

    def acceleration(self, values, electric):
        cells = values.reshape(values.shape[0], NCELLS, NGAUSS)
        field = electric[:, None]
        left = cells@self.left
        right = cells@self.right
        neighbor_left = torch.roll(right, 1, dims=1)
        neighbor_right = torch.roll(left, -1, dims=1)
        neighbor_left[:, 0] = 0.
        neighbor_right[:, -1] = 0.
        positive, negative = field.clamp_min(0), field.clamp_max(0)
        flux_left = positive*neighbor_left+negative*left
        flux_right = positive*right+negative*neighbor_right
        interior = field[:, :, None]*(cells@self.derivative.T)
        correction = ((flux_right-field*right)[:, :, None]*self.lift_right
                      -(flux_left-field*left)[:, :, None]*self.lift_left)
        return (-(interior+correction)*self.inverse_jacobian).reshape_as(values)


def velocity_data_projection(edges):
    """Cellwise fine Gauss data -> degree-(NGAUSS-1) L2 projection.

    The fine quadrature moments through degree NGAUSS-1 equal the moments of
    the projected DG polynomial under its own Gauss quadrature.
    """
    if NGAUSS_QUERY < NGAUSS:
        raise ValueError("Velocity query quadrature must resolve the DG basis")
    state_xi, _ = np.polynomial.legendre.leggauss(NGAUSS)
    query_xi, query_weight = np.polynomial.legendre.leggauss(NGAUSS_QUERY)
    state_basis = np.polynomial.legendre.legvander(state_xi, NGAUSS-1)
    query_basis = np.polynomial.legendre.legvander(query_xi, NGAUSS-1)
    inverse_legendre_mass = (2*np.arange(NGAUSS)+1)/2
    projection = ((state_basis*inverse_legendre_mass)@query_basis.T)*query_weight
    widths = np.diff(edges)
    query_nodes = ((edges[:-1, None]+edges[1:, None])/2
                   +widths[:, None]*query_xi/2).ravel()
    return query_nodes, projection


def prepare_solution(case, solve_budget=None):
    began = candidate_solve_clock()
    allowance = float(solve_budget) if solve_budget is not None else 900.
    if not torch.cuda.is_available():
        raise RuntimeError("This candidate requires its assigned CUDA device")
    device = torch.device("cuda:0")
    domain = case.public_data["equation"]["domain"]
    xmin, xmax = map(float, domain["x_interval"])
    tmin, tmax = map(float, domain["time_interval"])
    session = open_public_case_session(case)
    maximum = min(4096, int(case.public_data["equation"]["oracle_protocol"]["limits"]["max_points_per_request"]))

    def query(points, initial=False, reserve=20.):
        method = session.query_initial_distribution if initial else session.query_source
        values = np.empty(len(points), dtype=float)
        for offset in range(0, len(points), maximum):
            if candidate_solve_clock()-began >= allowance-reserve:
                raise _SamplingReserve("Sampling reached reserved case budget")
            part = points[offset:offset+maximum]
            values[offset:offset+len(part)] = np.asarray(method(part), dtype=float)
        return values

    edges, guided_grid, pilot_info = refine_velocity_grid(case, query, began, allowance)
    dg = DGVelocity(edges, device)
    v, quadrature = dg.nodes, dg.quadrature
    query_v, velocity_projection = velocity_data_projection(edges)
    velocity_projection = torch.as_tensor(velocity_projection, dtype=torch.float64, device=device)
    nqv = len(query_v)
    nx, nv, nt = NX, len(v), NT
    nqx = NX_QUERY
    x = xmin+(xmax-xmin)*np.arange(nqx)/nqx
    kx = 2*np.pi*np.fft.fftfreq(nx, (xmax-xmin)/nx)
    kx[nx//2] = 0.
    inverse_ik = np.zeros(nx, dtype=np.complex128)
    nonzero = kx != 0
    inverse_ik[nonzero] = 1/(1j*kx[nonzero])
    inverse_ik = torch.as_tensor(inverse_ik, device=device)
    quad_gpu = torch.as_tensor(quadrature, dtype=torch.complex128, device=device)
    weights = torch.as_tensor(np.stack([quadrature*v**p for p in range(4)]),
                              dtype=torch.complex128, device=device)
    def field(fhat):
        return (fhat@quad_gpu)*inverse_ik

    def project_samples(values):
        """Project fine velocity quadrature and retain spatial Fourier modes."""
        values = (values.reshape(nqx, NCELLS, NGAUSS_QUERY)
                  @velocity_projection.T).reshape(nqx, nv)
        fine = torch.fft.fft(values, dim=0)
        projected = torch.cat((fine[:nx//2], fine[-nx//2:]), dim=0)*(nx/nqx)
        projected[nx//2] = 0.
        return projected

    def macros(fhat):
        return torch.cat((weights@fhat.T/nx, field(fhat)[None, :]/nx), dim=0)

    def attach(prepared, effective_dt, count, source_nodes):
        prepared.velocity_cell_edges = edges
        prepared.velocity_nodes = v
        prepared.velocity_quadrature_weights = quadrature
        prepared.velocity_query_nodes = query_v
        prepared.effective_dt = effective_dt
        prepared.integration_steps = count
        prepared.effective_source_nodes = source_nodes
        prepared.grid_heuristic_used = guided_grid
        prepared.pilot_info = pilot_info
        return prepared

    xv = np.column_stack((np.repeat(x, nqv), np.tile(query_v, nqx)))
    f = project_samples(torch.as_tensor(query(xv, True).reshape(nqx, nqv), device=device))
    f[nx//2] = 0
    initial = macros(f).cpu().numpy()
    if not np.isfinite(initial).all():
        raise FloatingPointError("Initial DG moments are not finite")
    backup = attach(Prepared([tmin, tmax], np.stack((initial, initial)), xmin, xmax-xmin),
                    tmax-tmin, 0, 0)
    nodes = -np.cos(np.pi*np.arange(nt)/(nt-1))
    times = tmin+(tmax-tmin)*(1+nodes)/2
    if nt < 9 or (nt-1) & (nt-2):
        raise ValueError("NT must be nested 2**m+1 >= 9")
    source_samples = torch.empty((nt, nx, nv), dtype=torch.float64, device=device)
    points = np.column_stack((np.zeros(len(xv)), xv))
    initial_f = f
    del f
    evolution_records = []
    coarse_cost, coarse_nodes = None, 0
    coarse_attempted, coarse_attempt_seconds, coarse_time_limit = False, None, None
    evolution_reason = None

    def evolve(indices, deadline=None):
        """Build a complete trajectory, or report that its budget was exhausted."""
        nonlocal evolution_reason
        case_deadline = began+allowance-RETURN_RESERVE
        stop_at = case_deadline if deadline is None else min(case_deadline, deadline)
        time_reason = ('coarse_time_limit' if stop_at < case_deadline else 'case_time_limit')

        def stop(reason):
            nonlocal evolution_reason
            evolution_reason = reason
            return None

        if candidate_solve_clock() >= stop_at:
            return stop(time_reason)
        f = initial_f.clone()
        panel_samples = source_samples[torch.as_tensor(indices, device=device)]
        panel_times = times[indices]
        panel_nt = len(indices)
        spline = make_interp_spline(panel_times, np.eye(panel_nt), k=7, axis=0)
        transform = torch.as_tensor(spline.c, dtype=torch.float64, device=device)
        coefficients = torch.fft.fft((transform@panel_samples.reshape(panel_nt, -1)).reshape(panel_nt, nx, nv), dim=1)
        coefficients[:, nx//2] = 0
        rows = coefficients.reshape(panel_nt, -1)
        del panel_samples, transform
        nominal_h = min(DT, STREAMING_PHASE_LIMIT/max(1e-30, np.max(np.abs(kx))*np.max(np.abs(v))))
        base_steps = max(1, int(np.ceil((tmax-tmin)/nominal_h)))
        base_h = (tmax-tmin)/base_steps
        plans = {}

        def plan_for(level):
            if level not in plans:
                nstep = base_steps*2**level
                h = base_h/2**level
                stage_times = np.linspace(tmin, tmax, 2*nstep+1)
                design = BSpline.design_matrix(stage_times, spline.t, spline.k).tocsr()
                starts = design.indices.reshape(-1, 8)[:, 0].copy()
                local = torch.as_tensor(design.data.reshape(-1, 8), dtype=torch.complex128, device=device)
                z = -1j*h*kx[:, None]*v
                ph = torch.as_tensor(np.exp(z/2), device=device)
                phi1, phi2, phi3 = (_phi(z, order) for order in (1, 2, 3))
                q = torch.as_tensor(h/2*_phi(z/2, 1), device=device)
                f1 = torch.as_tensor(h*(phi1-3*phi2+4*phi3), device=device)
                f2 = torch.as_tensor(h*(phi2-2*phi3), device=device)
                f3 = torch.as_tensor(h*(-phi2+4*phi3), device=device)
                plans[level] = (nstep, h, starts, local, ph, ph*ph, q, f1, f2, f3)
            return plans[level]

        def nonlinear(fhat, sourcehat):
            values = torch.fft.ifft(fhat, dim=0).real
            electric = torch.fft.ifft(field(fhat)).real
            rhs = torch.fft.fft(dg.acceleration(values, electric), dim=0)+sourcehat
            rhs[nx//2] = 0
            return rhs

        history, out_times = [initial], [tmin]
        next_save = tmin+SAVE_DT
        level, tick, count = 0, 0, 0
        while True:
            if tick >= base_steps*2**level:
                break
            if candidate_solve_clock() >= stop_at:
                return stop(time_reason)
            electric_max = float(torch.fft.ifft(field(f)).real.abs().max().item())
            if not np.isfinite(electric_max):
                return stop('nonfinite_field')
            ratio = 1.5*base_h*electric_max/(dg.cfl*dg.min_width)
            needed = max(0, int(np.ceil(np.log2(max(1., ratio)))))
            if needed > level:
                tick *= 2**(needed-level)
                level = needed
            if count+(base_steps*2**level-tick) > MAX_STEPS:
                return stop('step_limit')
            nstep, h, starts, local, ph, full, q, f1, f2, f3 = plan_for(level)
            if tick >= nstep:
                break

            def source(index):
                start = starts[index]
                return (local[index]@rows[start:start+8]).reshape(nx, nv)

            s0, sm, s1 = source(2*tick), source(2*tick+1), source(2*tick+2)
            k1 = nonlinear(f, s0)
            a = ph*f+q*k1
            k2 = nonlinear(a, sm)
            b = ph*f+q*k2
            k3 = nonlinear(b, sm)
            c = ph*a+q*(2*k3-k1)
            k4 = nonlinear(c, s1)
            f = full*f+f1*k1+2*f2*(k2+k3)+f3*k4
            tick += 1
            count += 1
            current_time = tmax if tick == nstep else tmin+tick*h
            if current_time >= next_save or tick == nstep:
                history.append(macros(f).cpu().numpy())
                out_times.append(current_time)
                next_save = current_time+SAVE_DT
            if count % 20 == 0:
                torch.cuda.synchronize(device)
                if candidate_solve_clock() >= stop_at:
                    return stop(time_reason)
                if not bool(torch.isfinite(f).all().item()):
                    return stop('nonfinite_distribution')
        modes = np.stack(history)
        if not np.isfinite(modes).all():
            return stop('nonfinite_moments')
        evolution_reason = 'completed'
        return attach(Prepared(out_times, modes, xmin, xmax-xmin), base_h/2**level, count, panel_nt)

    def solve_panel(indices, deadline=None, phase='final'):
        nonlocal evolution_reason
        # This includes coefficient assembly, ETD plan creation and transfers.
        # The SDK clock excludes only execution-owned oracle queue waiting.
        torch.cuda.synchronize(device)
        started = candidate_solve_clock()
        evolution_reason = None
        result = evolve(indices, deadline=deadline)
        torch.cuda.synchronize(device)
        elapsed = max(1e-6, candidate_solve_clock()-started)
        evolution_records.append(dict(source_nodes=len(indices), seconds=elapsed,
                                      completed=result is not None,
                                      phase=phase, reason=evolution_reason))
        return result, elapsed

    def finish(result, used_backup):
        result.coarse_source_nodes = coarse_nodes
        result.coarse_evolution_seconds = coarse_cost
        result.coarse_attempted = coarse_attempted
        result.coarse_attempt_seconds = coarse_attempt_seconds
        result.coarse_time_limit_seconds = coarse_time_limit
        result.evolution_records = tuple(evolution_records)
        result.sampled_source_nodes = len(sampled)
        result.used_coarse_backup = bool(used_backup and coarse_nodes)
        return result

    complete, sampled, slice_costs, level = None, set(), [], 9
    while level <= nt:
        indices = np.arange(0, nt, (nt-1)//(level-1))
        new_indices = [int(i) for i in indices if i not in sampled]
        remaining = allowance-(candidate_solve_clock()-began)
        evolution_reserve = max(SOURCE_RESERVE,
                                RETURN_RESERVE+EVOLUTION_RESERVE_FACTOR*(coarse_cost or 0.))
        if complete is not None:
            estimate = max(np.mean(slice_costs), np.mean(slice_costs[-4:]))
            if estimate*len(new_indices) >= remaining-evolution_reserve:
                break
        try:
            for index in new_indices:
                started = candidate_solve_clock()
                predicted = max(np.mean(slice_costs), np.mean(slice_costs[-4:])) if slice_costs else 0.
                points[:, 0] = times[index]
                values = query(points, reserve=evolution_reserve+1.5*predicted).reshape(nqx, nqv)
                projected = project_samples(torch.as_tensor(values, device=device))
                source_samples[index].copy_(torch.fft.ifft(projected, dim=0).real)
                torch.cuda.synchronize(device)
                slice_costs.append(max(1e-6, candidate_solve_clock()-started))
                sampled.add(index)
        except _SamplingReserve:
            break
        complete = indices
        if not coarse_attempted and len(indices) >= min(COARSE_SOURCE_NODES, nt):
            coarse_attempted = True
            started = candidate_solve_clock()
            remaining = max(0., allowance-(started-began))
            coarse_time_limit = max(10., min(20., .1*remaining))
            deadline = min(began+allowance-RETURN_RESERVE, started+coarse_time_limit)
            prepared, elapsed = solve_panel(indices, deadline=deadline, phase='coarse')
            coarse_attempt_seconds = elapsed
            if prepared is not None:
                backup = prepared
                coarse_cost, coarse_nodes = elapsed, len(indices)
        level = 2*level-1

    if complete is None:
        return finish(backup, True)
    if coarse_cost is None:
        # A failed bounded coarse attempt must not prevent the fully sampled
        # source from receiving one ordinary solve with the remaining budget.
        prepared, _ = solve_panel(complete)
        return finish(backup if prepared is None else prepared, prepared is None)
    if len(complete) <= coarse_nodes:
        return finish(backup, True)
    remaining = allowance-(candidate_solve_clock()-began)
    if remaining < RETURN_RESERVE+EVOLUTION_RESERVE_FACTOR*coarse_cost:
        return finish(backup, True)
    prepared, _ = solve_panel(complete)
    if prepared is None:
        return finish(backup, True)
    return finish(prepared, False)
