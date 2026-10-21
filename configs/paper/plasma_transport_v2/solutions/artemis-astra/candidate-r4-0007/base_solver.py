"""Exact-pilot-refined Gauss DG velocity, Fourier x and ETDRK4."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import numpy as np
import torch
from scipy.interpolate import BSpline, make_interp_spline
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

FIELDS = ("M0", "M1", "M2", "M3", "E")
NX = 1024
NCELLS = 120
NGAUSS = 6
NT = 17
DT = 0.0025
GRID_UNIFORM_FRACTION = 0.25
MIN_CELL_WIDTH_FRACTION = 0.03
SOURCE_RESERVE = 100.0
STREAMING_PHASE_LIMIT = 3.0
MAX_STEPS = 100000
SPATIAL_MACRO_TOLERANCE = 1.e-4
SPATIAL_KINETIC_TOLERANCE = 1.e-4
SPATIAL_KINETIC_RATIO_TOLERANCE = 1.e-3
SPATIAL_INTERMEDIATE_NX = 1536
SPATIAL_INTERMEDIATE_MACRO_TOLERANCE = 1.e-7
SPATIAL_INTERMEDIATE_KINETIC_TOLERANCE = 1.e-6
SAVE_DT = 0.001
PILOT_BUDGET = 45.
PILOT_X_COUNT = 32
PILOT_FLOORS = np.array([1.0041253663782175, .4788345668393978,
                        2.930763214320111, 1.9801548942236534])

_sparse_spec = importlib.util.spec_from_file_location(
    '_transport_dense_source', Path(__file__).with_name('dense_source.py'))
_sparse_source = importlib.util.module_from_spec(_sparse_spec)
_sparse_spec.loader.exec_module(_sparse_source)

_initial_spec = importlib.util.spec_from_file_location(
    '_transport_initial_projection', Path(__file__).with_name('initial_projection.py'))
_initial_projection = importlib.util.module_from_spec(_initial_spec)
_initial_spec.loader.exec_module(_initial_projection)

_source_projection_spec = importlib.util.spec_from_file_location(
    '_transport_source_projection', Path(__file__).with_name('source_projection.py'))
_source_projection = importlib.util.module_from_spec(_source_projection_spec)
_source_projection_spec.loader.exec_module(_source_projection)

_compress_spec = importlib.util.spec_from_file_location(
    '_transport_compressed_source', Path(__file__).with_name('compressed_source.py'))
_compressed_source = importlib.util.module_from_spec(_compress_spec)
_compress_spec.loader.exec_module(_compressed_source)

_interaction_spec = importlib.util.spec_from_file_location(
    '_transport_interaction_source', Path(__file__).with_name('interaction_source.py'))
_interaction_source = importlib.util.module_from_spec(_interaction_spec)
_interaction_spec.loader.exec_module(_interaction_source)

_free_spec = importlib.util.spec_from_file_location(
    '_transport_free_source', Path(__file__).with_name('free_source.py'))
_free_source = importlib.util.module_from_spec(_free_spec)
_free_spec.loader.exec_module(_free_source)

_cell_spec = importlib.util.spec_from_file_location(
    '_transport_cell_source', Path(__file__).with_name('cell_source.py'))
_cell_source = importlib.util.module_from_spec(_cell_spec)
_cell_spec.loader.exec_module(_cell_source)

_integral_spec = importlib.util.spec_from_file_location(
    '_transport_source_integrals', Path(__file__).with_name('source_integral_plan.py'))
_source_integrals = importlib.util.module_from_spec(_integral_spec)
_integral_spec.loader.exec_module(_source_integrals)


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
        for time in (None, *np.linspace(tmin, tmax, 17)):
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
    maximum = int(case.public_data["equation"]["oracle_protocol"]["limits"]["max_points_per_request"])

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
    source_refinement_info = dict(attempted=False, completed=False, reason='not_reached')
    compression_info = dict(accepted=False, reason='not_reached')
    dg = DGVelocity(edges, device)
    v, quadrature = dg.nodes, dg.quadrature
    compression_plan, compression_info = _compressed_source.design(
        case, v, quadrature, query, began, allowance, _SamplingReserve)
    initial_nx = 2048
    initial_x = xmin+(xmax-xmin)*np.arange(initial_nx)/initial_nx
    try:
        initial_values = _initial_projection.sample_initial(initial_x, edges, NGAUSS, query)
    except _SamplingReserve:
        spec = importlib.util.spec_from_file_location(
            '_transport_startup_fallback', Path(__file__).with_name('startup_fallback.py'))
        fallback_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fallback_module)
        prepared = fallback_module.prepare(case, Prepared, xmin, xmax, tmin, tmax, NX)
        prepared.pilot_info = pilot_info
        prepared.source_compression_info = compression_info
        return prepared
    initial_hat = torch.fft.fft(torch.as_tensor(initial_values, device=device), dim=0)
    initial_weights = torch.as_tensor(np.stack([quadrature*v**p for p in range(4)]),
                                     dtype=torch.complex128, device=device)
    initial_moments = initial_weights@initial_hat.T/initial_nx
    high = np.abs(np.fft.fftfreq(initial_nx, 1./initial_nx)) >= NX//2
    scales = torch.as_tensor(PILOT_FLOORS, device=device)[:,None]
    high_indicator = float(torch.sqrt(torch.sum(
        torch.abs(initial_moments[:,high]/scales)**2)).item())
    kinetic_weights = torch.as_tensor(quadrature*(1.+v**6), device=device)
    kinetic_indicator = float(torch.sqrt(torch.sum(
        torch.abs(initial_hat[high]/initial_nx)**2*kinetic_weights)).item())
    kinetic_total = float(torch.sqrt(torch.sum(
        torch.abs(initial_hat/initial_nx)**2*kinetic_weights)).item())
    kinetic_ratio = kinetic_indicator/max(kinetic_total,1.e-300)
    use_fine = (high_indicator > SPATIAL_MACRO_TOLERANCE or
                (kinetic_indicator > SPATIAL_KINETIC_TOLERANCE and
                 kinetic_ratio > SPATIAL_KINETIC_RATIO_TOLERANCE))
    intermediate_macro_indicator = intermediate_kinetic_indicator = None
    use_intermediate = False
    if use_fine:
        intermediate_high = (np.abs(np.fft.fftfreq(initial_nx, 1./initial_nx))
                             >= SPATIAL_INTERMEDIATE_NX//2)
        intermediate_macro_indicator = float(torch.sqrt(torch.sum(
            torch.abs(initial_moments[:,intermediate_high]/scales)**2)).item())
        intermediate_kinetic_indicator = float(torch.sqrt(torch.sum(
            torch.abs(initial_hat[intermediate_high]/initial_nx)**2*kinetic_weights)).item())
        use_intermediate = (
            intermediate_macro_indicator <= SPATIAL_INTERMEDIATE_MACRO_TOLERANCE and
            intermediate_kinetic_indicator <= SPATIAL_INTERMEDIATE_KINETIC_TOLERANCE)
    nx = SPATIAL_INTERMEDIATE_NX if use_intermediate else 2048 if use_fine else NX
    nv, nt = len(v), NT
    if initial_nx == nx:
        f = initial_hat
    else:
        # Fourier projection avoids aliasing discarded initial modes.
        f = torch.cat((initial_hat[:nx//2], initial_hat[-nx//2:]), dim=0)*(nx/initial_nx)
    del initial_hat, initial_values
    spatial_info = dict(method='initial_high_macro_and_kinetic_energy', initial_spatial_nodes=initial_nx,
                        state_spatial_nodes=nx, eligible=True,
                        high_indicator=high_indicator, threshold=SPATIAL_MACRO_TOLERANCE,
                        kinetic_indicator=kinetic_indicator, kinetic_threshold=SPATIAL_KINETIC_TOLERANCE,
                        kinetic_total=kinetic_total, kinetic_ratio=kinetic_ratio,
                        kinetic_ratio_threshold=SPATIAL_KINETIC_RATIO_TOLERANCE,
                        intermediate_eligible=bool(use_fine),
                        intermediate_accepted=bool(use_intermediate),
                        intermediate_spatial_nodes=SPATIAL_INTERMEDIATE_NX,
                        intermediate_macro_indicator=intermediate_macro_indicator,
                        intermediate_macro_threshold=SPATIAL_INTERMEDIATE_MACRO_TOLERANCE,
                        intermediate_kinetic_indicator=intermediate_kinetic_indicator,
                        intermediate_kinetic_threshold=SPATIAL_INTERMEDIATE_KINETIC_TOLERANCE,
                        accepted=bool(use_fine))
    x = xmin+(xmax-xmin)*np.arange(nx)/nx
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

    def macros(fhat):
        return torch.cat((weights@fhat.T/nx, field(fhat)[None, :]/nx), dim=0)

    def attach(prepared, effective_dt, count, source_nodes):
        prepared.velocity_cell_edges = edges
        prepared.velocity_nodes = v
        prepared.velocity_quadrature_weights = quadrature
        prepared.effective_dt = effective_dt
        prepared.integration_steps = count
        prepared.effective_source_nodes = source_nodes
        prepared.grid_heuristic_used = guided_grid
        prepared.pilot_info = pilot_info
        prepared.source_refinement_info = source_refinement_info
        prepared.source_compression_info = compression_info
        prepared.spatial_resolution_info = spatial_info
        return prepared

    xv = np.column_stack((np.repeat(x, nv), np.tile(v, nx)))
    f[nx//2] = 0
    initial = macros(f).cpu().numpy()
    if not np.isfinite(initial).all():
        raise FloatingPointError("Initial DG moments are not finite")
    backup = attach(Prepared([tmin, tmax], np.stack((initial, initial)), xmin, xmax-xmin),
                    tmax-tmin, 0, 0)
    source_v = v if compression_plan is None else compression_plan['velocity']
    source_nv = len(source_v)
    validation_v = v
    validation_projection = None
    if compression_plan is None:
        def project_source(values):
            return torch.as_tensor(values, dtype=torch.float64, device=device)
    else:
        # The decision precedes main sampling: only compressed routes start at65.
        nt = 65
        # Integrate each P11 source element against the DG P5 basis exactly,
        # splitting at every source/state cell intersection before Gauss9.
        reconstruction = _source_projection.exact_element_projection(
            compression_plan['edges'], edges, NGAUSS, _compressed_source.SOURCE_ORDER)
        reconstruction = torch.as_tensor(reconstruction.T.copy(), dtype=torch.float64, device=device)
        high_nodes, validation_projection = _initial_projection.projection_matrix(NGAUSS)
        validation_v = ((edges[:-1, None]+edges[1:, None])/2
                        +np.diff(edges)[:, None]*high_nodes/2).ravel()
        compression_info['state_projection'] = 'exact_piecewise_p11_to_dg_p5'
        compression_info['validation_quadrature_order'] = len(high_nodes)
        def project_source(values):
            return torch.as_tensor(values, dtype=torch.float64, device=device) @ reconstruction
    validation_nv = len(validation_v)
    nodes = -np.cos(np.pi*np.arange(nt)/(nt-1))
    times = tmin+(tmax-tmin)*(1+nodes)/2
    if nt < 9 or (nt-1) & (nt-2):
        raise ValueError("NT must be nested 2**m+1 >= 9")
    source_samples = torch.empty((nt, nx, nv), dtype=torch.float64, device=device)
    points = np.column_stack((np.zeros(nx*source_nv), np.repeat(x,source_nv), np.tile(source_v,nx)))
    complete, sampled, slice_costs, level = None, set(), [], 9
    while level <= nt:
        indices = np.arange(0, nt, (nt-1)//(level-1))
        new_indices = [int(i) for i in indices if i not in sampled]
        remaining = allowance-(candidate_solve_clock()-began)
        if complete is not None:
            estimate = max(np.mean(slice_costs), np.mean(slice_costs[-4:]))
            if 1.5*estimate*len(new_indices) >= remaining-SOURCE_RESERVE:
                break
        try:
            for index in new_indices:
                started = candidate_solve_clock()
                predicted = max(np.mean(slice_costs), np.mean(slice_costs[-4:])) if slice_costs else 0.
                points[:, 0] = times[index]
                values = query(points, reserve=SOURCE_RESERVE+1.5*predicted).reshape(nx, source_nv)
                source_samples[index].copy_(project_source(values))
                torch.cuda.synchronize(device)
                slice_costs.append(max(1e-6, candidate_solve_clock()-started))
                sampled.add(index)
        except _SamplingReserve:
            if complete is None:
                return backup
            break
        complete = indices
        level = 2*level-1
    if complete is None:
        return backup
    source_samples = source_samples[torch.as_tensor(complete, device=device)]
    times = times[complete]
    initial_state = f

    def evolve(panel, panel_times, fallback, allow_interaction=True,
               validation_evolution_cost=None):
        """Evolve one complete source panel from the unchanged initial state."""
        optional_evolution = allow_interaction and validation_evolution_cost is not None
        finish_reserve = 10. if optional_evolution else 20.
        periodic_reserve = 8. if optional_evolution else 15.
        if candidate_solve_clock()-began >= allowance-finish_reserve:
            return fallback
        f = initial_state.clone()
        nt = len(panel_times)
        spline = make_interp_spline(panel_times, np.eye(nt), k=7, axis=0)
        transform = torch.as_tensor(spline.c, dtype=torch.float64, device=device)
        coefficients = torch.fft.fft((transform@panel.reshape(nt, -1)).reshape(nt, nx, nv), dim=1)
        coefficients[:, nx//2] = 0
        fit_started = candidate_solve_clock()
        carrier = np.zeros((nx, nv))
        free_mask = torch.zeros((nx, 1), dtype=torch.bool, device=device)
        # A completed final-panel trajectory protects optional source validation
        # and the subsequent interaction evolution. Measure its actual cost.
        fit_reserve = (max(20., 1.5*validation_evolution_cost+12.)+2.
                       if nt >= 33 and validation_evolution_cost is not None else 0.)
        if not allow_interaction or (nt >= 33 and allowance-(fit_started-began) < fit_reserve):
            interaction_mask = torch.zeros((nx, 1), dtype=torch.bool, device=device)
            interaction_info = dict(method='interaction_frame_row_gate', source_nodes=nt,
                                    accepted_rows=0, attempted=False,
                                    reason='raw_final_first' if not allow_interaction else 'fit_budget')
        else:
            validation_diagnostic = dict(validation_attempted=False,
                                         validation_completed=False,
                                         validation_slices=0)
            carrier_probe_cost = 0.
            validation_extra_reserve = 0.
            probe_diagnostic = dict(carrier_probe_attempted=False,
                                    carrier_probe_completed=False)

            def query_early_carrier_probe():
                """One exact early slice; preserve both later validation probes."""
                nonlocal carrier_probe_cost
                if compression_plan is not None:
                    probe_diagnostic['carrier_probe_reason'] = 'compressed_route_unchanged'
                    return None
                if nt < 33 or not slice_costs:
                    probe_diagnostic['carrier_probe_reason'] = 'panel_size'
                    return None
                estimate = max(float(np.mean(slice_costs)),float(np.mean(slice_costs[-4:])))
                measured = baseline_cost if validation_evolution_cost is None else validation_evolution_cost
                reserve = max(20.,1.5*measured+12.)+7.
                remaining = allowance-(candidate_solve_clock()-began)
                if 3.*estimate >= remaining-reserve:
                    probe_diagnostic['carrier_probe_reason'] = 'forecast_budget'
                    return None
                delta = min((tmax-tmin)*1.e-4,
                    np.pi/(4.*(np.max(np.abs(kx))*np.max(np.abs(v))+1.)))
                probe_time = tmin+delta
                if probe_time <= tmin:
                    probe_diagnostic['carrier_probe_reason'] = 'time_precision'
                    return None
                probe_points = np.column_stack((np.full(nx*nv,probe_time),
                                                np.repeat(x,nv),np.tile(v,nx)))
                probe_diagnostic['carrier_probe_attempted'] = True
                started = candidate_solve_clock()
                try:
                    values = query(probe_points,reserve=reserve+2.*estimate).reshape(nx,nv)
                except _SamplingReserve:
                    probe_diagnostic['carrier_probe_reason'] = 'query_budget'
                    return None
                carrier_probe_cost = max(1.e-6,candidate_solve_clock()-started)
                if not np.isfinite(values).all():
                    probe_diagnostic['carrier_probe_reason'] = 'nonfinite_probe'
                    return None
                probe_diagnostic.update(carrier_probe_completed=True,carrier_probe_reason='complete',
                    carrier_probe_time=float(probe_time),carrier_probe_seconds=carrier_probe_cost)
                return probe_time,values

            probe_cache = []
            def early_carrier_probe():
                if not probe_cache:
                    probe_cache.append(query_early_carrier_probe())
                return probe_cache[0]

            def validate_source_panel():
                """Optional two exact full slices; incomplete work never commits."""
                if nt < 33 or nt % 2 == 0 or not slice_costs:
                    validation_diagnostic['validation_reason'] = 'requires_complete_odd_panel'
                    return None
                estimate = max(float(np.mean(slice_costs)), float(np.mean(slice_costs[-4:])))
                # Compressed sources use exact L2 projection; validate against
                # independently queried G12 source projected to the same state.
                estimate *= float(validation_nv)/source_nv
                if compression_plan is None:
                    estimate = max(estimate,carrier_probe_cost)
                measured_evolution = (baseline_cost if validation_evolution_cost is None
                                      else validation_evolution_cost)
                reserve = max(20., 1.5*measured_evolution+12.)+2.+validation_extra_reserve
                remaining = allowance-(candidate_solve_clock()-began)
                validation_diagnostic['validation_reserve'] = reserve
                if estimate*2 >= remaining-reserve:
                    validation_diagnostic['validation_reason'] = 'forecast_budget'
                    return None
                validation_diagnostic['validation_attempted'] = True
                fractions = np.array([(3.-np.sqrt(5.))/2., .7179792746620228])
                validation_times = tmin+(tmax-tmin)*fractions
                validation_points = np.column_stack((np.zeros(nx*validation_nv),
                    np.repeat(x, validation_nv), np.tile(validation_v, nx)))
                validation_diagnostic['validation_velocity_nodes'] = validation_nv
                validation_diagnostic['validation_projection'] = (
                    'gauss12_to_dg_p5' if validation_projection is not None else 'identity')
                checked, costs = [], []
                validation_started = candidate_solve_clock()
                try:
                    for ordinal, validation_time in enumerate(validation_times):
                        predicted = max(estimate, max(costs) if costs else estimate)
                        remaining = allowance-(candidate_solve_clock()-began)
                        if predicted*(2-ordinal) >= remaining-reserve:
                            validation_diagnostic['validation_reason'] = 'measured_budget'
                            return None
                        started = candidate_solve_clock()
                        validation_points[:, 0] = validation_time
                        values = query(validation_points,
                            reserve=reserve+predicted*(1-ordinal)).reshape(nx, validation_nv)
                        if validation_projection is not None:
                            values = np.einsum('xcq,nq->xcn',
                                values.reshape(nx, len(edges)-1, len(high_nodes)),
                                validation_projection).reshape(nx, nv)
                        if not np.isfinite(values).all():
                            validation_diagnostic['validation_reason'] = 'nonfinite_validation'
                            return None
                        checked.append(values)
                        costs.append(max(1.e-6,candidate_solve_clock()-started))
                        validation_diagnostic['validation_slices'] = ordinal+1
                except _SamplingReserve:
                    validation_diagnostic['validation_reason'] = 'query_budget'
                    return None
                validation_diagnostic.update(validation_reason='complete', validation_completed=True,
                    validation_seconds=candidate_solve_clock()-validation_started)
                return validation_times, np.stack(checked), validation_diagnostic.copy()

            # Both source representations see the same independent exact
            # validation slices. Each retains its own row/cell safety gate.
            validation_cache = []
            def shared_validation():
                if not validation_cache:
                    validation_cache.append(validate_source_panel())
                return validation_cache[0]
            raw_coefficients = coefficients
            carrier_coefficients, carrier_mask, carrier_info, carrier = _interaction_source.improve_coefficients(
                panel, panel_times, kx, v, quadrature, coefficients.clone(), transform,
                validate=shared_validation if nt >= 33 else None,
                carrier_probe=early_carrier_probe if nt >= 33 and compression_plan is None else None)
            carrier = np.broadcast_to(np.asarray(carrier).reshape(nx, -1), (nx, nv)).copy()
            if compression_plan is None:
                carrier_info.update(probe_diagnostic)
            coefficients, free_mask, free_info = _free_source.improve_coefficients(
                panel, panel_times, kx, v, quadrature, coefficients.clone(), transform,
                validate=shared_validation if nt >= 33 else None)
            # Do not reopen rows accepted by either existing representation,
            # even if the combined exception cap later prunes that row.
            old_rejected = ~(carrier_mask[:,0] | free_mask[:,0])
            infinite = torch.full((nx,),float('inf'),device=device,dtype=torch.float64)
            carrier_error = carrier_info.pop('_validated_row_error',infinite)
            free_error = free_info.pop('_validated_row_error',infinite)
            carrier_exception = carrier_info.pop('_row_exception_bound',infinite)
            free_exception = free_info.pop('_row_exception_bound',infinite)
            carrier_wins = carrier_mask[:,0] & (~free_mask[:,0] | (carrier_error <= free_error))
            # Pair the choice itself as well as each representation's gate.
            partners = torch.as_tensor((-np.arange(nx)) % nx,device=device)
            carrier_wins &= carrier_wins[partners]
            free_mask &= ~carrier_wins[:,None]
            selected = carrier_wins | free_mask[:,0]
            exception = torch.where(carrier_wins,carrier_exception,
                                    torch.where(free_mask[:,0],free_exception,0.))
            selected_cpu = selected.cpu().numpy().copy()
            exception_cpu = exception.cpu().numpy()
            pair_costs = [(float(exception_cpu[j]**2+exception_cpu[-j]**2),j)
                          for j in range(1,nx//2) if selected_cpu[j]]
            squared = sum(cost for cost,j in pair_costs)
            before_pruning = np.sqrt(squared)
            for cost,j in sorted(pair_costs,reverse=True):
                if squared <= 1.e-8:
                    break
                selected_cpu[j] = selected_cpu[-j] = False
                squared = max(0.,squared-cost)
            selected = torch.as_tensor(selected_cpu,device=device)
            carrier_wins &= selected
            free_mask &= selected[:,None]
            carrier = np.where(carrier_wins.cpu().numpy()[:,None],carrier,0.)
            coefficients[:,carrier_wins] = carrier_coefficients[:,carrier_wins]
            coefficients[:,~selected] = raw_coefficients[:,~selected]
            interaction_mask = free_mask | carrier_wins[:,None]
            interaction_info = dict(method='dual_row_frame_gate',source_nodes=nt,
                accepted_rows=int(interaction_mask.sum().item()),
                carrier_rows=int(carrier_wins.sum().item()),
                free_streaming_rows=int(free_mask.sum().item()),
                combined_exception_rms_before_pruning=float(before_pruning),
                combined_exception_rms_after_pruning=float(np.sqrt(squared)),
                carrier_diagnostic=carrier_info,free_streaming_diagnostic=free_info)
            del raw_coefficients,carrier_coefficients
            del carrier_error,free_error,carrier_exception,free_exception,infinite,exception
            cell_info = dict(attempted=False,accepted_rows=0,reason='no_rejected_rows')
            if nt >= 33 and bool(old_rejected[1:nx//2].any().item()):
                cell_info['fit_extra_reserve'] = 10.
                remaining = allowance-(candidate_solve_clock()-began)
                cell_info['remaining_before_fit'] = float(remaining)
                if remaining <= fit_reserve+10.:
                    cell_info['reason'] = 'fit_budget'
                else:
                    free_bytes,_ = torch.cuda.mem_get_info(device)
                    available = (free_bytes+torch.cuda.memory_reserved(device)
                                 -torch.cuda.memory_allocated(device))
                    # The helper keeps full FFT/demodulated panels plus
                    # several held-out physical/ballistic arrays live.
                    required = 10*nt*nx*nv*16+(1<<30)
                    cell_info.update(available_bytes=int(available),required_bytes=int(required))
                    if available < required:
                        cell_info['reason'] = 'memory_forecast'
                    else:
                        def cell_validation():
                            nonlocal validation_extra_reserve
                            if validation_cache:
                                return validation_cache[0]
                            validation_extra_reserve = 10.
                            try:
                                return shared_validation()
                            finally:
                                validation_extra_reserve = 0.
                        cell_coefficients = None
                        cell_started = candidate_solve_clock()
                        try:
                            cell_coefficients,cell_mask,diagnostic,cell_carrier = _cell_source.improve_coefficients(
                                panel,panel_times,kx,v,quadrature,coefficients.clone(),transform,
                                validate=cell_validation,
                                carrier_probe=early_carrier_probe if compression_plan is None else None)
                            diagnostic.pop('_validated_row_error',None)
                            diagnostic.pop('_row_exception_bound',None)
                            cell_info.update(diagnostic,attempted=True)
                        except (torch.cuda.OutOfMemoryError,MemoryError):
                            cell_info.update(attempted=True,reason='fit_memory',accepted_rows=0)
                        if cell_coefficients is not None:
                            old_bound = max(np.max(np.abs(kx[:,None]*v[None,:]-carrier)),
                                            np.max(np.abs(carrier)))
                            row_bound = np.maximum(np.max(np.abs(kx[:,None]*v[None,:]-cell_carrier),axis=1),
                                                   np.max(np.abs(cell_carrier),axis=1))
                            phase_safe = torch.as_tensor(row_bound<=old_bound,device=device)
                            cell_selected = cell_mask[:,0] & old_rejected & phase_safe
                            cell_selected &= cell_selected[partners]
                            if allowance-(candidate_solve_clock()-began) <= fit_reserve:
                                cell_selected[:] = False
                                cell_info['reason'] = 'fit_deadline'
                            cell_info.update(accepted_before_parent_gate=int(cell_mask.sum().item()),
                                phase_veto_rows=int((cell_mask[:,0]&old_rejected&~phase_safe).sum().item()),
                                accepted_rows=int(cell_selected.sum().item()),
                                old_global_phase_bound=float(old_bound))
                            coefficients[:,cell_selected] = cell_coefficients[:,cell_selected]
                            selected_numpy = cell_selected.cpu().numpy()
                            carrier[selected_numpy] = cell_carrier[selected_numpy]
                            interaction_mask |= cell_selected[:,None]
                            del cell_coefficients,cell_mask,cell_carrier,cell_selected
                        cell_info['fit_seconds'] = candidate_solve_clock()-cell_started
            interaction_info.update(cell_carrier_diagnostic=cell_info,
                cell_carrier_rows=cell_info['accepted_rows'],
                accepted_rows=int(interaction_mask.sum().item()))
            interaction_info.update(validation_diagnostic)
        torch.cuda.synchronize(device)
        interaction_info['fit_seconds'] = candidate_solve_clock()-fit_started
        interaction_info['fit_reserve'] = fit_reserve
        interaction_info['source_integration'] = 'carrier_etdrk4_and_exact_free_source'
        use_interaction = bool(interaction_info['accepted_rows'])
        use_free = bool(free_mask.any().item())
        if nt >= 33 and allow_interaction and validation_evolution_cost is not None:
            fallback.interaction_source_info = interaction_info
        if nt >= 33 and allow_interaction and validation_evolution_cost is not None and not use_interaction:
            return fallback
        carrier_phase = torch.ones((nx, 1), dtype=torch.complex128, device=device)
        free_phase = torch.ones((nx,nv),dtype=torch.complex128,device=device) if use_free else None
        rows = coefficients.reshape(nt, -1)
        del transform
        linear_frequency = kx[:,None]*v[None,:]-carrier
        nominal_h = min(DT, STREAMING_PHASE_LIMIT/max(1e-30,
            np.max(np.abs(linear_frequency)),np.max(np.abs(carrier))))
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
                z = -1j*h*linear_frequency
                ph = torch.as_tensor(np.exp(z/2), device=device)
                phi1, phi2, phi3 = (_phi(z, order) for order in (1, 2, 3))
                q = torch.as_tensor(h/2*_phi(z/2, 1), device=device)
                f1 = torch.as_tensor(h*(phi1-3*phi2+4*phi3), device=device)
                f2 = torch.as_tensor(h*(phi2-2*phi3), device=device)
                f3 = torch.as_tensor(h*(-phi2+4*phi3), device=device)
                carrier_half = torch.as_tensor(np.exp(.5j*h*carrier),device=device)
                if use_free:
                    integral_starts, half_weights, full_weights = _source_integrals.integral_plan(
                        spline.t,spline.k,stage_times)
                    half_weights = torch.as_tensor(half_weights,dtype=torch.complex128,device=device)
                    full_weights = torch.as_tensor(full_weights,dtype=torch.complex128,device=device)
                    integrals = (integral_starts,half_weights,full_weights)
                else:
                    integrals = None
                plans[level] = (nstep, h, starts, local, ph, ph*ph, q, f1, f2, f3, carrier_half,integrals)
            return plans[level]

        def nonlinear(fhat, sourcehat, phase):
            physical = fhat*phase.conj() if use_interaction else fhat
            values = torch.fft.ifft(physical, dim=0).real
            electric = torch.fft.ifft(field(physical)).real
            rhs = torch.fft.fft(dg.acceleration(values, electric), dim=0)
            if use_interaction:
                rhs *= phase
            rhs += sourcehat
            rhs[nx//2] = 0
            return rhs

        history, out_times = [initial], [tmin]
        next_save = tmin+SAVE_DT
        level, tick, count = 0, 0, 0
        while True:
            if tick >= base_steps*2**level:
                break
            if candidate_solve_clock()-began >= allowance-finish_reserve:
                return fallback
            physical = f*carrier_phase.conj() if use_interaction else f
            electric_max = float(torch.fft.ifft(field(physical)).real.abs().max().item())
            if not np.isfinite(electric_max):
                return fallback
            ratio = 1.5*base_h*electric_max/(dg.cfl*dg.min_width)
            needed = max(0, int(np.ceil(np.log2(max(1., ratio)))))
            if needed > level:
                tick *= 2**(needed-level)
                level = needed
            if count+(base_steps*2**level-tick) > MAX_STEPS:
                return fallback
            nstep, h, starts, local, ph, full, q, f1, f2, f3, carrier_half,integrals = plan_for(level)
            if tick >= nstep:
                break

            def source(index, phase=None):
                start = starts[index]
                result = (local[index]@rows[start:start+8]).reshape(nx, nv)
                if use_free:
                    result = torch.where(free_mask,0.,result)
                return result

            s0, sm, s1 = source(2*tick), source(2*tick+1), source(2*tick+2)
            if use_free:
                integral_starts,half_weights,full_weights = integrals
                first = integral_starts[tick]
                ncoeff = half_weights.shape[1]
                integral_rows = rows[first:first+ncoeff]
                source_half = (half_weights[tick]@integral_rows).reshape(nx,nv)
                source_full = (full_weights[tick]@integral_rows).reshape(nx,nv)
                source_half = torch.where(free_mask,source_half*free_phase*ph,0.)
                source_full = torch.where(free_mask,source_full*free_phase*full,0.)
            else:
                source_half = source_full = 0.
            mid_phase = carrier_phase*carrier_half
            end_phase = mid_phase*carrier_half
            k1 = nonlinear(f, s0, carrier_phase)
            a = ph*f+q*k1+source_half
            k2 = nonlinear(a, sm, mid_phase)
            b = ph*f+q*k2+source_half
            k3 = nonlinear(b, sm, mid_phase)
            c = ph*(a-source_half)+q*(2*k3-k1)+source_full
            k4 = nonlinear(c, s1, end_phase)
            f = full*f+f1*k1+2*f2*(k2+k3)+f3*k4+source_full
            if use_interaction:
                carrier_phase = end_phase
            if use_free:
                free_phase *= full
            tick += 1
            count += 1
            current_time = tmax if tick == nstep else tmin+tick*h
            if current_time >= next_save or tick == nstep:
                physical = f*carrier_phase.conj() if use_interaction else f
                history.append(macros(physical).cpu().numpy())
                out_times.append(current_time)
                next_save = current_time+SAVE_DT
            if count % 20 == 0:
                torch.cuda.synchronize(device)
                if candidate_solve_clock()-began >= allowance-periodic_reserve or not bool(torch.isfinite(f).all().item()):
                    return fallback
        modes = np.stack(history)
        if not np.isfinite(modes).all():
            return fallback
        prepared = attach(Prepared(out_times, modes, xmin, xmax-xmin), base_h/2**level, count, nt)
        prepared.interaction_source_info = interaction_info
        return prepared

    # Complete the baseline trajectory before spending any optional budget.
    # Both timing boundaries synchronize CUDA so the reserve includes all
    # coefficient preparation, GPU evolution and host macro transfers.
    torch.cuda.synchronize(device)
    evolution_started = candidate_solve_clock()
    baseline = evolve(source_samples, times, backup, allow_interaction=False)
    torch.cuda.synchronize(device)
    baseline_cost = max(1e-6, candidate_solve_clock()-evolution_started)
    if baseline is backup:
        return backup
    if compression_plan is not None:
        refinement_reserve = max(30., 2.*baseline_cost+20.)
        current=baseline
        current_raw_cost=baseline_cost
        source_refinement_info = dict(method='compressed_common_panels',
                                     baseline_ready=True, stages=[], returned_source_nodes=len(times))
        current.source_refinement_info=source_refinement_info
        target_time_nodes=compression_plan['time_nodes']
        if nx == 1536 and target_time_nodes == 257 and len(times) == 65:
            estimate = max(float(np.mean(slice_costs)), float(np.mean(slice_costs[-4:])))
            remaining = allowance-(candidate_solve_clock()-began)
            validation_reserve = 2.*estimate*(float(validation_nv)/source_nv)*1.05+10.
            available = remaining-refinement_reserve-validation_reserve
            selected_target = next((target for target in (257, 193, 129)
                                    if (target-65)*estimate < available), 65)
            source_refinement_info['reserved_panel_decision'] = dict(
                original_target=257, selected_target=selected_target,
                measured_slice_seconds=estimate, remaining_seconds=remaining,
                refinement_reserve=refinement_reserve,
                optional_validation_and_fit_reserve=validation_reserve,
                reason='largest_complete_panel_preserving_validation')
            target_time_nodes = selected_target
        while len(times)<target_time_nodes:
            next_time_nodes=193 if target_time_nodes==193 else 2*len(times)-1
            refined_samples,refined_times,info = _compressed_source.refine(
                source_samples,times,x,source_v,project_source,query,began,allowance,
                slice_costs,_SamplingReserve,refinement_reserve,next_time_nodes)
            source_refinement_info['stages'].append(info)
            if not info.get('completed',False):
                break
            started=candidate_solve_clock()
            refined=evolve(refined_samples,refined_times,current,allow_interaction=False)
            torch.cuda.synchronize(device)
            if refined is current:
                info['reason']='refined_evolution_rollback'
                break
            source_samples,times,current=refined_samples,refined_times,refined
            current_raw_cost=max(1.e-6,candidate_solve_clock()-started)
            refinement_reserve=max(30.,2.*current_raw_cost+20.)
            source_refinement_info['returned_source_nodes']=len(times)
            source_refinement_info['refined_evolution_succeeded']=True
        # Preserve a complete raw checkpoint before optional carrier validation.
        improved=evolve(source_samples,times,current,allow_interaction=True,
                        validation_evolution_cost=current_raw_cost)
        torch.cuda.synchronize(device)
        source_refinement_info['raw_final_ready']=True
        source_refinement_info['raw_final_source_nodes']=len(times)
        source_refinement_info['raw_final_evolution_seconds']=current_raw_cost
        source_refinement_info['interaction_evolution_succeeded']=improved is not current
        return improved
    refinement_reserve = max(40., 2.*baseline_cost+20.)
    refined_samples, refined_times, source_refinement_info = _sparse_source.refine_source(
        source_samples, times, x, v, quadrature, query, began, allowance,
        slice_costs, _SamplingReserve, refinement_reserve=refinement_reserve,
        forecast_factor=1.05)
    source_refinement_info.update(baseline_ready=True, baseline_source_nodes=len(times),
                                  baseline_evolution_seconds=baseline_cost,
                                  returned_source_nodes=len(times),
                                  refined_evolution_succeeded=False)
    baseline.source_refinement_info = source_refinement_info
    if not source_refinement_info.get('completed', False):
        return baseline
    raw_refinement_started = candidate_solve_clock()
    refined = evolve(refined_samples, refined_times, baseline, allow_interaction=False)
    torch.cuda.synchronize(device)
    raw_refinement_cost = max(1.e-6,candidate_solve_clock()-raw_refinement_started)
    if refined is baseline:
        source_refinement_info['reason'] = 'refined_evolution_rollback'
        return baseline
    source_refinement_info['refined_evolution_succeeded'] = True
    source_refinement_info['returned_source_nodes'] = len(refined_times)
    source_refinement_info['raw_final_evolution_seconds'] = raw_refinement_cost
    source_refinement_info['raw_final_ready'] = True
    source_refinement_info['raw_final_source_nodes'] = len(refined_times)
    improved = evolve(refined_samples, refined_times, refined, allow_interaction=True,
                      validation_evolution_cost=raw_refinement_cost)
    torch.cuda.synchronize(device)
    source_refinement_info['interaction_evolution_succeeded'] = improved is not refined
    if improved is refined and not hasattr(refined,'interaction_source_info'):
        refined.interaction_source_info = dict(accepted_rows=0,attempted=False,reason='optional_budget')
    return improved
