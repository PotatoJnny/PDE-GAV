"""Prepared macro realizability alternatives; no data access or fitting.

Moment axis is third from last for Fourier input and first for raw snapshots.
Raw moments use (rho, x, y, xx, xy, yy, xxx, xxy, xyy, yyy, ...).
Density and central pressure constraints define a convex set.  They impose no
restriction on degree-three moments without additional trustworthy degree-four
information; this is deliberate and is not a positivity theorem for f itself.
"""
from __future__ import annotations

import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView
from moments import central_values
from base import FIELDS
from realizability import gram


def evaluate_grid(modes, n):
    m = modes.shape[-1]
    if n < m:
        raise ValueError("oversampling must retain all Fourier modes")
    frequencies = np.rint(np.fft.fftfreq(m) * m).astype(int)
    out = np.zeros(modes.shape[:-2] + (n, n), complex)
    out[..., frequencies[:, None] % n, frequencies[None, :] % n] = modes
    return np.fft.ifft2(out, axes=(-2, -1)).real * n * n


def macro_valid(raw, density_floor=.15, pressure_floor=.01):
    """Return one Boolean per spatial point, using the central 2x2 stress.

P >= pressure_floor*I is a convex raw-moment constraint by its 3x3 Schur
complement, so every accepted convex interpolation also obeys the constraint.
"""
    r, x, y, xx, xy, yy = np.asarray(raw)[:6]
    den = np.maximum(r, np.finfo(float).tiny)
    with np.errstate(over='ignore', invalid='ignore'):
        px = xx-x*x/den-pressure_floor
        py = yy-y*y/den-pressure_floor
        pc = xy-x*y/den
        valid = ((r >= density_floor) & (px >= 0.) & (py >= 0.)
                 & (px*py >= pc*pc))
    return valid & np.isfinite(raw).all(axis=0)


def pointwise_weights(base, delta, *, degree4=False, density_floor=.15,
                      pressure_floor=.01, iterations=40, retreat=1e-8):
    """Maximize a local whole-state blend in a convex realizability domain."""
    base, delta = np.asarray(base, float), np.asarray(delta, float)
    if base.shape != delta.shape or base.ndim < 2 or base.shape[0] < 6:
        raise ValueError("matching finite raw moment arrays required")
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError("nonfinite raw state cannot be blended")
    def valid(raw):
        result = macro_valid(raw, density_floor, pressure_floor)
        if degree4:
            result &= np.linalg.eigvalsh(gram(raw, True))[..., 0] >= -1e-10
        return result
    if not valid(base).all():
        raise ValueError("prior is not a valid local convex anchor")
    accepted = valid(base+delta)
    if accepted.all():
        return np.ones(base.shape[1:])
    if not degree4:
        # Generalized eigenvalues solve the local semidefinite line search
        # exactly when the prior is interior: G(a)=L(I+a*H)L.T. The floor is
        # subtracted only from the two stress diagonals, making G affine in
        # every raw moment. This avoids 40 matrix scans on fine prepared grids.
        indices = np.asarray([[0,1,2],[1,3,4],[2,4,5]])
        g = np.moveaxis(base[indices], (0,1), (-2,-1)).copy()
        d = np.moveaxis(delta[indices], (0,1), (-2,-1)).copy()
        g[...,1,1] -= pressure_floor; g[...,2,2] -= pressure_floor
        try:
            scale = np.sqrt(np.diagonal(g, axis1=-2, axis2=-1))
            g /= scale[...,None,:]*scale[...,:,None]
            d /= scale[...,None,:]*scale[...,:,None]
            lower = np.linalg.cholesky(g)
            whitened = np.linalg.solve(lower, d)
            whitened = np.linalg.solve(lower, np.swapaxes(whitened,-1,-2))
            whitened = .5*(whitened+np.swapaxes(whitened,-1,-2))
            minimum = np.linalg.eigvalsh(whitened)[...,0]
            pressure_weight = np.divide(-1., minimum, out=np.ones_like(minimum), where=minimum < -1.)
            density_weight = np.divide(base[0]-density_floor, -delta[0],
                out=np.ones_like(base[0]), where=delta[0] < 0)
            weight = np.clip(np.minimum(pressure_weight,density_weight),0.,1.)
            weight = np.where(accepted,1.,weight*(1.-retreat))
            if valid(base+weight[None]*delta).all():
                return weight
        except np.linalg.LinAlgError:
            pass
        # A boundary prior or numerical conditioning can prevent Cholesky.
        # Convex bisection is the conservative fallback below.
    lo = np.zeros(base.shape[1:]); hi = np.ones_like(lo)
    lo[accepted] = 1.
    for _ in range(iterations):
        middle = (lo+hi)*.5
        good = valid(base+middle[None]*delta)
        lo = np.where(good, middle, lo)
        hi = np.where(good, hi, middle)
    return np.where(accepted, 1., lo*(1.-retreat))


def global_guard(prior_modes, correction_modes, check=lambda: None, *,
                 degree4=False, oversampling=2, iterations=40):
    """Drop-in global guard with continuous maximal rather than dyadic weights.

The spectral representation remains unchanged. Oversampling is a numerical
check, not a formal proof of between-node positivity.
"""
    prior_modes = np.asarray(prior_modes); correction_modes = np.asarray(correction_modes)
    out = prior_modes.copy(); weights = []
    for it in range(len(out)):
        check()
        base = evaluate_grid(prior_modes[it], oversampling*out.shape[-1])
        delta = evaluate_grid(correction_modes[it], oversampling*out.shape[-1])
        weights_local = pointwise_weights(base, delta, degree4=degree4, iterations=iterations)
        weight = float(weights_local.min())
        out[it] += weight*correction_modes[it]
        weights.append(weight)
    return out, weights


def local_guard_grid(prior_modes, correction_modes, check=lambda: None, *,
                     degree4=False, oversampling=2, iterations=40,
                     conserve_mass=True):
    """Prepare positive nodal raw moments for later multilinear projection.

This returns a grid, not Fourier coefficients. Do not Fourier-project it before
evaluation: a nonpositive reconstruction can defeat the local convex theorem.
Positive bilinear space and linear time interpolation preserve all guard
constraints at arbitrary coordinates, including off-grid points.
"""
    prior_modes = np.asarray(prior_modes); correction_modes = np.asarray(correction_modes)
    n = oversampling*prior_modes.shape[-1]
    out = np.empty(prior_modes.shape[:-2]+(n, n), float)
    weights = np.empty((len(out), n, n), float)
    for it in range(len(out)):
        check()
        base = evaluate_grid(prior_modes[it], n)
        delta = evaluate_grid(correction_modes[it], n)
        weights[it] = pointwise_weights(base, delta, degree4=degree4, iterations=iterations)
        out[it] = base+weights[it][None]*delta
        if conserve_mass:
            # Local mixing can change the zero Fourier mode. A common positive
            # scaling restores total particles while preserving PSD, velocity,
            # and moment relations. It scales the positive numerical floors by
            # the same known factor; no density-only clamp is introduced.
            out[it] *= base[0].mean()/out[it, 0].mean()
    return out, weights


def bilinear_values(grid, xy, periods):
    """Exact periodic convex interpolant of a prepared nodal raw state."""
    grid, xy, periods = np.asarray(grid), np.asarray(xy), np.asarray(periods)
    n = grid.shape[-1]
    z = (xy/periods*n) % n
    i = np.floor(z).astype(int); alpha = z-i
    j = (i+1) % n
    ax, ay = alpha.T
    return (grid[:, i[:, 0], i[:, 1]]*((1-ax)*(1-ay))[None]
            + grid[:, j[:, 0], i[:, 1]]*(ax*(1-ay))[None]
            + grid[:, i[:, 0], j[:, 1]]*((1-ax)*ay)[None]
            + grid[:, j[:, 0], j[:, 1]]*(ax*ay)[None])


def bilinear_density_electric_modes(grid, periods, cutoff=None):
    """Poisson field of the prepared bilinear density, spectrally truncated.

Periodic hat basis Fourier coefficients are nodal DFT coefficients times
sinc(kx/n)^2*sinc(ky/n)^2. Alias copies outside the nodal Nyquist cell are
included up to cutoff. This is consistent with the actual interpolated charge,
up to an explicitly selected field Fourier cutoff.
"""
    density = np.asarray(grid)[0]
    n = density.shape[-1]
    if cutoff is None:
        cutoff = n
    k = np.arange(-cutoff, cutoff+1)
    dft = np.fft.fft2(density)/n**2
    rho = dft[k[:, None] % n, k[None, :] % n]
    rho = rho*np.sinc(k/n)[:, None]**2*np.sinc(k/n)[None, :]**2
    kx = 2*np.pi*k[:, None]/periods[0]
    ky = 2*np.pi*k[None, :]/periods[1]
    k2 = kx*kx+ky*ky
    inv = np.divide(1., k2, out=np.zeros_like(k2), where=k2 > 0)
    return np.stack((-1j*kx*inv*rho, -1j*ky*inv*rho)), k


def field_values(modes, frequencies, xy, periods):
    xy, periods = np.asarray(xy), np.asarray(periods)
    bx = np.exp(2j*np.pi*xy[:, 0, None]*frequencies/periods[0])
    by = np.exp(2j*np.pi*xy[:, 1, None]*frequencies/periods[1])
    return np.einsum('cij,ni,nj->cn', modes, bx, by, optimize=True).real


class LocalPrepared:
    """Continuous positive raw-moment interpolant and its prepared Poisson field.

This can replace submission.Prepared after local_guard_grid. All evolution,
limiting, and Poisson preparation precede query disclosure; evaluate only
projects immutable prepared arrays and centralizes their own local velocity.
"""
    def __init__(self, baseline, times, raw_grid, diagnostics=None, check=lambda: None):
        self.baseline = baseline
        self.times = np.asarray(times, float)
        self.raw_grid = np.asarray(raw_grid, float)
        self.diagnostics = diagnostics or {}
        self.closed = False
        self.field_modes = []
        for grid in self.raw_grid:
            check()
            modes, self.field_frequencies = bilinear_density_electric_modes(grid, baseline.periods)
            self.field_modes.append(modes)
        self.field_modes = np.asarray(self.field_modes)

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError('prepared solution closed')
        # Reuse incumbent's validated public query contract.
        self.baseline.evaluate(query)
        p = np.asarray([(v['t'], v['x'], v['y']) for v in query.points], float)
        left = np.clip(np.searchsorted(self.times, p[:, 0], side='right')-1,
                       0, len(self.times)-2)
        alpha = (p[:, 0]-self.times[left])/(self.times[left+1]-self.times[left])
        raw = np.empty((10, len(p))); field = np.empty((2, len(p)))
        for i in np.unique(left):
            mask = left == i; xy = p[mask, 1:]; a = alpha[mask]
            raw[:, mask] = ((1-a)*bilinear_values(self.raw_grid[i, :10], xy, self.baseline.periods)
                            +a*bilinear_values(self.raw_grid[i+1, :10], xy, self.baseline.periods))
            field[:, mask] = ((1-a)*field_values(self.field_modes[i], self.field_frequencies, xy, self.baseline.periods)
                              +a*field_values(self.field_modes[i+1], self.field_frequencies, xy, self.baseline.periods))
        macros = central_values(raw, self.baseline.velocity_center)
        values = np.concatenate((macros[:1], field, macros[1:]))
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite local prepared projection')
        return AnswerView(kind=query.kind,
                          fields={key: tuple(map(float, values[FIELDS.index(key)])) for key in query.fields},
                          points=query.points, shape=(len(p),))

    def close(self):
        self.closed = True
        self.baseline.close()
