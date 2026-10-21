"""Optional .25 hybrid fallback from already sampled triangular initial moments.

No Oracle calls or source evolution. Gaussian damping acts on rho/E/j/P raw
moments with exp(-.25*(t-t0)^2*|k|^2). Heat uses the undamped initial raw moments
and its own initial velocity. This matches hybrid Prepared given equal input
moments, although source and hybrid initial samplers can use different windows.
"""
from __future__ import annotations

import numpy as np
from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind

FIELDS = ('rho', 'electric_x', 'electric_y', 'momentum_x', 'momentum_y',
          'pressure_xx', 'pressure_xy', 'pressure_yy', 'heat_flux_x', 'heat_flux_y')


def packed_hybrid_modes(initial):
    """Pack degree>=3 triangular raw coefficients and their Poisson field."""
    modes = np.asarray(initial['modes'], dtype=complex)
    if modes.ndim != 3 or modes.shape[0] < 10 or not np.isfinite(modes).all():
        raise ValueError('finite triangular initial modes through degree three required')
    count = modes.shape[0]
    degree = int(round((np.sqrt(8*count+1)-3)/2))
    if count != (degree+1)*(degree+2)//2:
        raise ValueError('initial moment count is not triangular')
    if 'pairs' in initial:
        expected = tuple((a, n-a) for n in range(4) for a in range(n, -1, -1))
        if tuple(map(tuple, initial['pairs'][:10])) != expected:
            raise ValueError('initial moments use an unexpected order')
    periods = np.asarray(initial['periods'], dtype=float)
    if periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0):
        raise ValueError('positive initial periods required')
    packed = np.empty((10,)+modes.shape[-2:], dtype=complex)
    packed[:6] = modes[:6]
    packed[6] = .5*(modes[6]+modes[8])
    packed[7] = .5*(modes[7]+modes[9])
    kx, ky = np.meshgrid(*[2*np.pi*np.fft.fftfreq(n, d=L/n)
                           for n, L in zip(modes.shape[-2:], periods)], indexing='ij')
    square = kx*kx+ky*ky
    phi = np.divide(modes[0], square, out=np.zeros_like(modes[0]), where=square > 0)
    # This follows the full-Laplacian/zero-first-Nyquist convention used by
    # the source initial sampler and the independently tested hybrid program.
    nx, ny = modes.shape[-2:]
    if nx % 2 == 0:
        kx[nx//2, :] = 0.
    if ny % 2 == 0:
        ky[:, ny//2] = 0.
    packed[8] = -1j*kx*phi
    packed[9] = -1j*ky*phi
    if nx % 2 == 0:
        packed[..., nx//2, :] = 0.
    if ny % 2 == 0:
        packed[..., :, ny//2] = 0.
    return packed


class HybridInitialHandle:
    def __init__(self, initial, damping_rate=.25):
        self.modes = packed_hybrid_modes(initial)
        self.periods = np.asarray(initial['periods'], dtype=float)
        self.interval = np.asarray(initial['interval'], dtype=float)
        self.velocity_center = np.asarray(initial['center'], dtype=float)
        if (self.interval.shape != (2,) or not np.isfinite(self.interval).all()
                or self.interval[1] <= self.interval[0]
                or self.velocity_center.shape != (2,) or not np.isfinite(self.velocity_center).all()):
            raise ValueError('finite ordered interval and velocity center required')
        self.damping_rate = float(damping_rate)
        if not np.isfinite(self.damping_rate) or self.damping_rate < 0:
            raise ValueError('nonnegative finite damping rate required')
        self.wave = [2*np.pi*np.fft.fftfreq(n, d=L/n)
                     for n, L in zip(self.modes.shape[-2:], self.periods)]
        self.closed = False
        self.diagnostics = dict(fallback='initial hybrid', damping_rate=self.damping_rate,
                                heat='initial central', public_queries=0)

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError('prepared hybrid fallback is closed')
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError('hybrid fallback accepts POINTS only')
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError('unknown or duplicate transport field')
        p = np.asarray([(value['t'], value['x'], value['y']) for value in query.points], dtype=float)
        if p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all():
            raise ValueError('finite nonempty t,x,y points required')
        if np.any(p[:, 0] < self.interval[0]) or np.any(p[:, 0] > self.interval[1]):
            raise ValueError('query leaves prepared public interval')
        bx = np.exp(1j*p[:, 1, None]*self.wave[0])
        by = np.exp(1j*p[:, 2, None]*self.wave[1])
        bx_initial, by_initial = bx.copy(), by.copy()
        elapsed_squared = (p[:, 0, None]-self.interval[0])**2
        bx *= np.exp(-self.damping_rate*elapsed_squared*self.wave[0]**2)
        by *= np.exp(-self.damping_rate*elapsed_squared*self.wave[1]**2)
        m = np.einsum('cij,ni,nj->nc', self.modes, bx, by).real
        rho = np.maximum(m[:, 0], 1e-8)
        ux, uy = m[:, 1]/rho, m[:, 2]/rho
        xx = m[:, 3]-rho*ux*ux
        xy = m[:, 4]-rho*ux*uy
        yy = m[:, 5]-rho*uy*uy
        initial = np.einsum('cij,ni,nj->nc', self.modes[:8], bx_initial, by_initial).real
        r0 = np.maximum(initial[:, 0], 1e-8)
        u0, v0 = initial[:, 1]/r0, initial[:, 2]/r0
        qx = (initial[:, 6]-u0*(1.5*initial[:, 3]+.5*initial[:, 5])-v0*initial[:, 4]
              +r0*u0*(u0*u0+v0*v0))
        qy = (initial[:, 7]-v0*(.5*initial[:, 3]+1.5*initial[:, 5])-u0*initial[:, 4]
              +r0*v0*(u0*u0+v0*v0))
        jx, jy = m[:, 1]+rho*self.velocity_center[0], m[:, 2]+rho*self.velocity_center[1]
        values = dict(zip(FIELDS, (rho, m[:, 8], m[:, 9], jx, jy, xx, xy, yy, qx, qy)))
        if not all(np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError('nonfinite hybrid fallback macros')
        return AnswerView(kind=query.kind, points=query.points, shape=(len(p),),
                          fields={name: tuple(map(float, values[name])) for name in query.fields})

    def close(self):
        self.closed = True
