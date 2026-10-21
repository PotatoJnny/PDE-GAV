"""Fourier--Hermite VPFP macro solver with exact diagonal OU propagation."""
from __future__ import annotations

from math import factorial, isfinite
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

FIELDS = ("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
          "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y")
NX = 32
NH = 8
NQ = 24
VMAX = 8.0
NSOURCE = 257
NSTEPS = 128


def _hermite(z, n):
    h = np.empty((len(z), n), dtype=np.float64)
    h[:, 0] = 1.0
    if n > 1:
        h[:, 1] = z
    for k in range(1, n-1):
        h[:, k+1] = z*h[:, k] - k*h[:, k-1]
    return h


def _electric(rho, kx, ky):
    rh = np.fft.fft2(rho-rho.mean())
    k2 = kx*kx+ky*ky
    ph = np.zeros_like(rh)
    np.divide(rh, k2, out=ph, where=k2 > 0)
    return np.fft.ifft2(-1j*kx*ph).real, np.fft.ifft2(-1j*ky*ph).real


def _macros(a, theta, kx, ky):
    tiny = 1.e-8
    rho = a[0, 0]
    sr = np.sqrt(theta)
    jx = sr*a[1, 0]
    jy = sr*a[0, 1]
    mxx = theta*(rho+2.0*a[2, 0])
    mxy = theta*a[1, 1]
    myy = theta*(rho+2.0*a[0, 2])
    den = np.maximum(rho, tiny)
    ux, uy = jx/den, jy/den
    pxx = mxx-jx*ux
    pxy = mxy-jx*uy
    pyy = myy-jy*uy
    s3 = theta*sr
    rxxx = s3*(6.0*a[3, 0]+3.0*a[1, 0])
    rxxy = s3*(2.0*a[2, 1]+a[0, 1])
    rxyy = s3*(2.0*a[1, 2]+a[1, 0])
    ryyy = s3*(6.0*a[0, 3]+3.0*a[0, 1])
    cxxx = rxxx-3.0*ux*mxx+2.0*rho*ux**3
    cxxy = rxxy-2.0*ux*mxy-uy*mxx+2.0*rho*ux*ux*uy
    cxyy = rxyy-2.0*uy*mxy-ux*myy+2.0*rho*ux*uy*uy
    cyyy = ryyy-3.0*uy*myy+2.0*rho*uy**3
    ex, ey = _electric(rho, kx, ky)
    return np.stack((rho, ex, ey, jx, jy, pxx, pxy, pyy,
                     0.5*(cxxx+cxyy), 0.5*(cxxy+cyyy)))


class Prepared:
    def __init__(self, values, periods, interval):
        self.values = values
        self.periods = periods
        self.interval = interval
        self.closed = False

    def _interpolate(self, a, p):
        # a is time,x,y on a uniform inclusive time mesh and periodic space mesh.
        st = (p[:, 0]-self.interval[0])/(self.interval[1]-self.interval[0])*(len(a)-1)
        it = np.minimum(np.floor(st).astype(np.int64), len(a)-2)
        ft = st-it
        z = np.mod(p[:, 1:]/self.periods, 1.0)*NX
        lo = np.floor(z).astype(np.int64)
        f = z-lo
        out = np.zeros(len(p), dtype=np.float64)
        for dt in (0, 1):
            wt = ft if dt else 1.0-ft
            for i in (0, 1):
                for j in (0, 1):
                    w = wt*(f[:, 0] if i else 1.0-f[:, 0])*(f[:, 1] if j else 1.0-f[:, 1])
                    out += w*a[it+dt, (lo[:, 0]+i) % NX, (lo[:, 1]+j) % NX]
        return out

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS required")
        if not query.fields or len(set(query.fields)) != len(query.fields) or not set(query.fields) <= set(FIELDS):
            raise ValueError("invalid fields")
        p = np.asarray([(r["t"], r["x"], r["y"]) for r in query.points], dtype=np.float64)
        if p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all():
            raise ValueError("invalid points")
        if np.any(p[:, 0] < self.interval[0]) or np.any(p[:, 0] > self.interval[1]):
            raise ValueError("time outside interval")
        vals = {k: self._interpolate(self.values[:, FIELDS.index(k)], p) for k in query.fields}
        if any(not np.isfinite(x).all() for x in vals.values()):
            raise ValueError("nonfinite result")
        return AnswerView(kind=query.kind, fields={k: tuple(map(float, vals[k])) for k in query.fields},
                          points=query.points, shape=(len(p),))

    def close(self):
        self.closed = True


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
                                     or not isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")
    def check(reserve=5.0):
        if solve_budget is not None and candidate_solve_clock()-started > solve_budget-reserve:
            raise TimeoutError("Fourier-Hermite preparation exhausted budget")

    q = case.public_data["question"]
    periods = np.asarray(q["grid"]["x_domain"], dtype=np.float64)
    interval = np.asarray(q["time_domain"], dtype=np.float64)
    theta = float(q["temperature"])
    nu = float(q["collision_nu"])
    sr = np.sqrt(theta)
    x = np.arange(NX)*periods[0]/NX
    y = np.arange(NX)*periods[1]/NX
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    kx = (2*np.pi*np.fft.fftfreq(NX, d=periods[0]/NX))[:, None]
    ky = (2*np.pi*np.fft.fftfreq(NX, d=periods[1]/NX))[None, :]
    session = open_public_case_session(case)
    rho0 = np.asarray(session.rho0(xy), dtype=np.float64)

    z, w = np.polynomial.legendre.leggauss(NQ)
    v = VMAX*z
    w = VMAX*w
    vx, vy = np.meshgrid(v, v, indexing="ij")
    vv = np.column_stack((vx.ravel(), vy.ravel()))
    ww = (w[:, None]*w[None, :]).ravel()
    hx, hy = _hermite(vv[:, 0]/sr, NH), _hermite(vv[:, 1]/sr, NH)
    basis = np.empty((len(vv), NH, NH), dtype=np.float64)
    for i in range(NH):
        for j in range(NH):
            basis[:, i, j] = ww*hx[:, i]*hy[:, j]/(factorial(i)*factorial(j))
    bflat = basis.reshape(len(vv), -1)
    nphase = len(vv)
    maxq = int(q.get("numerical_access", {}).get("max_points_per_request", 262144))
    p0 = np.column_stack((np.repeat(xy[:, 0], nphase), np.repeat(xy[:, 1], nphase),
                          np.tile(vv[:, 0], len(xy)), np.tile(vv[:, 1], len(xy))))
    chunks = []
    for i in range(0, len(p0), maxq):
        check()
        chunks.append(np.asarray(session.g0(p0[i:i+maxq]), dtype=np.float64))
    g0 = np.concatenate(chunks).reshape(len(xy), nphase)
    a = ((rho0[:, None]*g0)@bflat).reshape(NX, NX, NH, NH).transpose(2, 3, 0, 1)

    # Project the public source onto the same finite Hermite space at a modest
    # time mesh. Its analytic time dependence is smooth; linear interpolation
    # is used inside the finer ODE integration.
    ts = np.linspace(interval[0], interval[1], NSOURCE)
    sources = np.empty((NSOURCE, NH, NH, NX, NX), dtype=np.float64)
    base = np.column_stack((np.repeat(xy[:, 0], nphase), np.repeat(xy[:, 1], nphase),
                            np.tile(vv[:, 0], len(xy)), np.tile(vv[:, 1], len(xy))))
    for it, t in enumerate(ts):
        check()
        ps = np.column_stack((np.full(len(base), t), base))
        vals = []
        for i in range(0, len(ps), maxq):
            vals.append(np.asarray(session.source(ps[i:i+maxq]), dtype=np.float64))
        sv = np.concatenate(vals).reshape(len(xy), nphase)
        sources[it] = (sv@bflat).reshape(NX, NX, NH, NH).transpose(2, 3, 0, 1)

    def source_at(t):
        zt = (t-interval[0])/(interval[1]-interval[0])*(NSOURCE-1)
        i = min(int(np.floor(zt)), NSOURCE-2)
        f = zt-i
        return (1.0-f)*sources[i]+f*sources[i+1]

    def rhs(u, t):
        bx = np.zeros_like(u)
        by = np.zeros_like(u)
        bx[1:] += u[:-1]
        by[:, 1:] += u[:, :-1]
        for i in range(NH-1):
            bx[i] += (i+1)*u[i+1]
            by[:, i] += (i+1)*u[:, i+1]
        dx = np.fft.ifftn(1j*kx*np.fft.fftn(bx, axes=(-2, -1)), axes=(-2, -1)).real
        dy = np.fft.ifftn(1j*ky*np.fft.fftn(by, axes=(-2, -1)), axes=(-2, -1)).real
        out = -sr*(dx+dy)+source_at(t)
        ex, ey = _electric(u[0, 0], kx, ky)
        out[1:] += (ex/sr)*u[:-1]
        out[:, 1:] += (ey/sr)*u[:, :-1]
        return out

    degree = np.add.outer(np.arange(NH), np.arange(NH))[:, :, None, None]
    h = (interval[1]-interval[0])/NSTEPS
    efull = np.exp(-nu*degree*h)
    ehalf = np.exp(-nu*degree*h/2)
    history = np.empty((NSTEPS+1, len(FIELDS), NX, NX), dtype=np.float64)
    history[0] = _macros(a, theta, kx, ky)
    t = interval[0]
    # Fourth-order Lawson method: exact OU attenuation, RK4 on transport,
    # electric acceleration, and the projected manufactured source.
    for step in range(NSTEPS):
        check()
        k1 = rhs(a, t)
        u2 = ehalf*(a+0.5*h*k1)
        k2 = rhs(u2, t+0.5*h)
        u3 = ehalf*a+0.5*h*k2
        k3 = rhs(u3, t+0.5*h)
        u4 = efull*a+h*ehalf*k3
        k4 = rhs(u4, t+h)
        a = efull*a+(h/6.0)*(efull*k1+2*ehalf*k2+2*ehalf*k3+k4)
        t = interval[0]+(step+1)*h
        history[step+1] = _macros(a, theta, kx, ky)
        if not np.isfinite(a).all() or np.max(np.abs(a)) > 1.e6:
            raise ValueError("unstable Fourier-Hermite solve")
    return Prepared(history, periods, interval)
