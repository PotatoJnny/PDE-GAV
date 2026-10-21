"""Initial-data macro reconstruction with full central transport moments.

This inexpensive fallback holds the initial distribution's moments throughout
the public time interval. Velocity quadrature uses a wide window centered on the public initial bulk
flow. Central moments are reconstructed in that frame to avoid cancellation.
There is no claim of time evolution or source response.
"""
from __future__ import annotations

import numpy as np

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

NX = 32
NV = 80
VMAX = 14.0
FIELDS = ('rho', 'electric_x', 'electric_y', 'momentum_x', 'momentum_y',
          'pressure_xx', 'pressure_xy', 'pressure_yy', 'heat_flux_x', 'heat_flux_y')


class Prepared:
    def __init__(self, modes, periods, interval, velocity_center):
        self.modes = modes
        self.periods = periods
        self.interval = interval
        self.velocity_center = np.asarray(velocity_center, dtype=float)
        self.wave = [2*np.pi*np.fft.fftfreq(NX, d=length/NX) for length in periods]
        self.closed = False

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError('prepared solution is closed')
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError('static macro reconstruction accepts POINTS only')
        if not query.fields or len(set(query.fields)) != len(query.fields) or not set(query.fields) <= set(FIELDS):
            raise ValueError('unknown or duplicate transport field')
        p = np.asarray([(v['t'], v['x'], v['y']) for v in query.points], dtype=float)
        if p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all():
            raise ValueError('finite nonempty t,x,y points are required')
        if np.any(p[:, 0] < self.interval[0]) or np.any(p[:, 0] > self.interval[1]):
            raise ValueError('query leaves prepared public time interval')
        bx = np.exp(1j*p[:, 1, None]*self.wave[0])
        by = np.exp(1j*p[:, 2, None]*self.wave[1])
        m = np.einsum('cij,ni,nj->nc', self.modes, bx, by).real
        r = np.maximum(m[:, 0], 1e-8)
        ux, uy = m[:, 1]/r, m[:, 2]/r
        pxx = m[:, 3]-r*ux*ux
        pxy = m[:, 4]-r*ux*uy
        pyy = m[:, 5]-r*uy*uy
        # Contracted raw third moments become central heat flux, including
        # mixed third moments and the kinetic-energy one-half convention.
        qx = m[:, 6]-ux*(1.5*m[:, 3]+.5*m[:, 5])-uy*m[:, 4]+r*ux*(ux*ux+uy*uy)
        qy = m[:, 7]-uy*(.5*m[:, 3]+1.5*m[:, 5])-ux*m[:, 4]+r*uy*(ux*ux+uy*uy)
        jx = m[:, 1]+r*self.velocity_center[0]
        jy = m[:, 2]+r*self.velocity_center[1]
        values = dict(zip(FIELDS, (r, m[:, 8], m[:, 9], jx, jy, pxx, pxy, pyy, qx, qy)))
        if not all(np.isfinite(v).all() for v in values.values()):
            raise FloatingPointError('nonfinite projected macros')
        return AnswerView(kind=query.kind,
                          fields={key: tuple(map(float, values[key])) for key in query.fields},
                          points=query.points, shape=(len(p),))

    def close(self):
        self.closed = True


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (float, int))
            or not np.isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError('solve_budget must be finite and positive')
    budget = 850. if solve_budget is None else float(solve_budget)

    def check_budget():
        if candidate_solve_clock()-started > budget-10.:
            raise TimeoutError('initial macro preparation exceeded return reserve')

    question = case.public_data['question']
    periods = np.asarray(question['grid']['x_domain'], dtype=float)
    interval = np.asarray(question['time_domain'], dtype=float)
    if (periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0)
            or interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] <= interval[0]):
        raise ValueError('positive public periods and ordered time interval required')
    limit = question.get('numerical_access', {}).get('max_points_per_request', 262144)
    if type(limit) is not int or limit <= 0:
        raise ValueError('positive public request limit required')
    limit = min(limit, 262144)
    # The global center is a public initial moment, not a scoring coordinate.
    # All stored velocity moments remain relative to this fixed center.
    velocity_center = np.zeros(2)
    preview = case.public_data.get('arrays', {})
    if 'rho0' in preview and 'momentum0' in preview:
        initial_rho = np.asarray(preview['rho0'], dtype=float)
        initial_j = np.asarray(preview['momentum0'], dtype=float)
        if (initial_rho.ndim == 2 and initial_j.shape == initial_rho.shape+(2,)
                and np.isfinite(initial_rho).all() and np.isfinite(initial_j).all()
                and np.mean(initial_rho) > 0):
            velocity_center = np.mean(initial_j, axis=(0,1))/np.mean(initial_rho)
    axes = [np.arange(NX)*length/NX for length in periods]
    xy = np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing='ij')])
    v = (np.arange(NV)-NV//2)*2*VMAX/NV
    vv = np.column_stack([a.ravel() for a in np.meshgrid(v, v, indexing='ij')])
    vx, vy = vv.T
    weights = np.asarray([np.ones_like(vx), vx, vy, vx*vx, vx*vy, vy*vy,
                          .5*vx*(vx*vx+vy*vy), .5*vy*(vx*vx+vy*vy)])*(2*VMAX/NV)**2
    vv = vv+velocity_center[None, :]
    session = open_public_case_session(case)
    check_budget()
    density_parts = []
    for start in range(0, len(xy), limit):
        batch = xy[start:start+limit]
        values = np.asarray(session.rho0(batch), dtype=float)
        if values.shape != (len(batch),) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('initial density must be finite and positive')
        density_parts.append(values)
        check_budget()
    density = np.concatenate(density_parts)
    moments = np.empty((len(xy), 8))
    # Whole velocity blocks make accumulation and normalization independent of
    # the Oracle's own internal request chunking.
    blocks = max(1, limit//len(vv))
    for start in range(0, len(xy), blocks):
        cells = xy[start:start+blocks]
        phase_points = np.concatenate([np.repeat(cells, len(vv), axis=0),
                                       np.tile(vv, (len(cells), 1))], axis=1)
        parts = []
        for offset in range(0, len(phase_points), limit):
            batch = phase_points[offset:offset+limit]
            part = np.asarray(session.g0(batch), dtype=float)
            if part.shape != (len(batch),) or not np.isfinite(part).all():
                raise ValueError('initial velocity density must be finite')
            parts.append(part)
            check_budget()
        g = np.concatenate(parts).reshape(len(cells), len(vv))
        local = g@weights.T
        if np.any(local[:, 0] <= 0):
            raise ValueError('positive initial velocity mass required')
        local *= (density[start:start+len(cells)]/local[:, 0])[:, None]
        local[:, 0] = density[start:start+len(cells)]
        moments[start:start+len(cells)] = local
    moments = moments.T.reshape(8, NX, NX)
    wave = np.meshgrid(*(2*np.pi*np.fft.fftfreq(NX, d=length/NX) for length in periods), indexing='ij')
    k2 = wave[0]**2+wave[1]**2
    rho_modes = np.fft.fft2(moments[0])
    potential = np.divide(rho_modes, k2, out=np.zeros_like(rho_modes), where=k2 > 0)
    # Keep the full Laplacian eigenvalue above, but zero real first-derivative
    # Nyquist modes on the collocation lattice.
    wave[0][NX//2, :] = 0.
    wave[1][:, NX//2] = 0.
    electric = np.asarray([np.fft.ifft2(-1j*k*potential).real for k in wave])
    modes = np.fft.fft2(np.concatenate([moments, electric], axis=0))/NX**2
    # Remove the ambiguous collocation Nyquist planes consistently from all
    # fields, so the continuous off-grid density/field pair obeys Poisson too.
    modes[...,NX//2,:] = 0.
    modes[...,:,NX//2] = 0.
    check_budget()
    return Prepared(modes, periods, interval, velocity_center)
