"""Source-informed, bounded macro forecasts; no Oracle access in this module."""
from __future__ import annotations
import numpy as np

FIELDS = ("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
          "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y")


def moment_pairs(degree):
    return tuple((a, total-a) for total in range(degree+1)
                 for a in range(total, -1, -1))


def wave_mesh(shape, periods):
    return np.meshgrid(*[2*np.pi*np.fft.fftfreq(n, d=length/n)
                         for n, length in zip(shape, periods)], indexing="ij")


def poisson_field(density, periods):
    kx, ky = wave_mesh(density.shape[-2:], periods)
    square = kx*kx+ky*ky
    phi = np.divide(np.fft.fft2(density), square,
                    out=np.zeros_like(density, dtype=complex), where=square > 0)
    if density.shape[-2] % 2 == 0:
        kx[density.shape[-2]//2, :] = 0.
    if density.shape[-1] % 2 == 0:
        ky[:, density.shape[-1]//2] = 0.
    return np.asarray([np.fft.ifft2(-1j*k*phi).real for k in (kx, ky)])


def spatial_gradients(moments, periods):
    kx, ky = wave_mesh(moments.shape[-2:], periods)
    if moments.shape[-2] % 2 == 0:
        kx[moments.shape[-2]//2, :] = 0.
    if moments.shape[-1] % 2 == 0:
        ky[:, moments.shape[-1]//2] = 0.
    coefficients = np.fft.fft2(moments)
    return tuple(np.fft.ifft2(1j*k*coefficients).real for k in (kx, ky))


def raw_moment_derivative(initial, source, periods, nu, theta,
                          center=(0., 0.), degree=3):
    """Hierarchy RHS through D, requiring initial D+1 in fixed w=v-center."""
    initial, source = np.asarray(initial, dtype=float), np.asarray(source, dtype=float)
    pairs = moment_pairs(degree+1)
    lookup = {pair: index for index, pair in enumerate(pairs)}
    count = len(moment_pairs(degree))
    if (initial.ndim != 3 or initial.shape[0] < len(pairs)
            or source.shape != (count,)+initial.shape[-2:]):
        raise ValueError("initial D+1 and source D on one spatial mesh required")
    if not np.isfinite(initial).all() or not np.isfinite(source).all():
        raise ValueError("finite moments required")
    cx, cy = np.asarray(center, dtype=float)
    dx, dy = spatial_gradients(initial, periods)
    ex, ey = poisson_field(initial[0], periods)
    derivative = source.copy()
    for i, (a, b) in enumerate(pairs[:count]):
        derivative[i] -= dx[lookup[(a+1, b)]]+cx*dx[i]
        derivative[i] -= dy[lookup[(a, b+1)]]+cy*dy[i]
        derivative[i] -= nu*(a+b)*initial[i]
        if a:
            derivative[i] += a*(ex-nu*cx)*initial[lookup[(a-1, b)]]
        if b:
            derivative[i] += b*(ey-nu*cy)*initial[lookup[(a, b-1)]]
        if a >= 2:
            derivative[i] += nu*theta*a*(a-1)*initial[lookup[(a-2, b)]]
        if b >= 2:
            derivative[i] += nu*theta*b*(b-1)*initial[lookup[(a, b-2)]]
    return derivative


def macros_and_derivative(raw, derivative, periods, center=(0., 0.)):
    """Central P/q and their exact own-velocity Jacobian actions."""
    raw, derivative = np.asarray(raw), np.asarray(derivative)
    if raw.shape[0] < 10 or derivative.shape != (10,)+raw.shape[1:]:
        raise ValueError("raw moments and derivatives through degree three required")
    rho, drho = raw[0], derivative[0]
    if np.any(rho <= 0):
        raise ValueError("positive density required")
    velocity = raw[1:3]/rho
    dvelocity = (derivative[1:3]-velocity*drho)/rho
    ux, uy = velocity
    dux, duy = dvelocity
    pressure = np.asarray([raw[3]-rho*ux*ux, raw[4]-rho*ux*uy, raw[5]-rho*uy*uy])
    dpressure = np.asarray([
        derivative[3]-drho*ux*ux-2*rho*ux*dux,
        derivative[4]-drho*ux*uy-rho*(dux*uy+ux*duy),
        derivative[5]-drho*uy*uy-2*rho*uy*duy])
    Q = .5*np.asarray([raw[6]+raw[8], raw[7]+raw[9]])
    dQ = .5*np.asarray([derivative[6]+derivative[8], derivative[7]+derivative[9]])
    trace, dtrace = raw[3]+raw[5], derivative[3]+derivative[5]
    Mu = np.asarray([raw[3]*ux+raw[4]*uy, raw[4]*ux+raw[5]*uy])
    Mdu = np.asarray([raw[3]*dux+raw[4]*duy, raw[4]*dux+raw[5]*duy])
    dMu = np.asarray([derivative[3]*ux+derivative[4]*uy,
                      derivative[4]*ux+derivative[5]*uy])
    speed2, u_dot_du = ux*ux+uy*uy, ux*dux+uy*duy
    heat = Q-Mu-.5*trace*velocity+rho*speed2*velocity
    dheat = (dQ-dMu-Mdu-.5*dtrace*velocity-.5*trace*dvelocity
             +drho*speed2*velocity+2*rho*u_dot_du*velocity+rho*speed2*dvelocity)
    center = np.asarray(center)[:, None, None]
    momentum, dmomentum = raw[1:3]+rho*center, derivative[1:3]+drho*center
    macros = np.concatenate([rho[None], poisson_field(rho, periods), momentum, pressure, heat])
    dmacros = np.concatenate([drho[None], poisson_field(drho, periods), dmomentum, dpressure, dheat])
    return macros, dmacros


def resize_modes(modes, shape):
    result = np.zeros(modes.shape[:-2]+tuple(shape), dtype=complex)
    mappings = []
    for n, target in zip(modes.shape[-2:], shape):
        source_frequency = np.rint(np.fft.fftfreq(n)*n).astype(int)
        target_frequency = np.rint(np.fft.fftfreq(target)*target).astype(int)
        lookup = {int(k): i for i, k in enumerate(source_frequency)
                  if n % 2 or i != n//2}
        pairs = [(lookup[int(k)], j) for j, k in enumerate(target_frequency)
                 if int(k) in lookup and (target % 2 or j != target//2)]
        mappings.append(tuple(np.asarray(a) for a in zip(*pairs)))
    (sx, tx), (sy, ty) = mappings
    result[..., tx[:, None], ty[None, :]] = modes[..., sx[:, None], sy[None, :]]
    return result


class GaussianDerivativeForecast:
    """Squared-exponential conditional mean anchored by value and derivative."""
    GROUPS = (("charge", (0,), (1.,), (0, 1, 2)),
              ("particle", (3, 4), (1., 1.), (3, 4)),
              ("pressure", (5, 6, 7), (1., 2., 1.), (5, 6, 7)),
              ("heat", (8, 9), (1., 1.), (8, 9)))

    def __init__(self, initial_macros, derivative_macros, periods, t0=0.,
                 tau_bounds=(.1, 1.), timescales=None):
        initial, derivative = np.asarray(initial_macros, dtype=float), np.asarray(derivative_macros, dtype=float)
        if initial.ndim != 3 or initial.shape[0] != 10 or derivative.ndim != 3 or derivative.shape[0] != 10:
            raise ValueError("ten initial/derivative macro fields on periodic meshes required")
        if not np.isfinite(initial).all() or not np.isfinite(derivative).all():
            raise ValueError("finite macro fields required")
        self.periods, self.t0 = np.asarray(periods, dtype=float), float(t0)
        self.shape = tuple(max(a, b) for a, b in zip(initial.shape[-2:], derivative.shape[-2:]))
        self.initial_modes = resize_modes(np.fft.fft2(initial)/np.prod(initial.shape[-2:]), self.shape)
        self.derivative_modes = resize_modes(np.fft.fft2(derivative)/np.prod(derivative.shape[-2:]), self.shape)
        self.mean = self.initial_modes[:, 0, 0].real
        self.fluctuation_modes = self.initial_modes.copy()
        self.fluctuation_modes[:, 0, 0] = 0.
        self.tau, self.diagnostics = np.empty(10), {}
        lower, upper = map(float, tau_bounds)
        if not 0 < lower <= upper < np.inf:
            raise ValueError("ordered positive finite tau bounds required")
        for name, fields, weights, assign in self.GROUPS:
            energy = np.sum(np.abs(self.fluctuation_modes[list(fields)])**2, axis=(-2, -1))
            derivative_energy = np.sum(np.abs(self.derivative_modes[list(fields)])**2, axis=(-2, -1))
            amplitude = float(np.sqrt(np.dot(weights, energy)))
            slope = float(np.sqrt(np.dot(weights, derivative_energy)))
            estimate = amplitude/slope if slope > 1e-30 else upper
            tau = float(np.clip(estimate, lower, upper))
            if timescales is not None and name in timescales:
                tau = float(timescales[name])
            if not np.isfinite(tau) or tau <= 0:
                raise ValueError("positive finite time scales required")
            self.tau[list(assign)] = tau
            self.diagnostics[name] = {"initial_fluctuation_rms": amplitude,
                                      "initial_derivative_rms": slope,
                                      "unclipped_tau": estimate, "tau": tau}
        self.wave = [2*np.pi*np.fft.fftfreq(n, d=length/n)
                     for n, length in zip(self.shape, self.periods)]

    def _project(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("finite (t,x,y) array required")
        if np.any(points[:, 0] < self.t0):
            raise ValueError("forecast requires times at or after its initial anchor")
        bx = np.exp(1j*points[:, 1, None]*self.wave[0])
        by = np.exp(1j*points[:, 2, None]*self.wave[1])
        fluctuation = np.einsum("cij,ni,nj->nc", self.fluctuation_modes, bx, by).real
        derivative = np.einsum("cij,ni,nj->nc", self.derivative_modes, bx, by).real
        elapsed = points[:, 0, None]-self.t0
        correlation = np.exp(-.5*(elapsed/self.tau)**2)
        return fluctuation, derivative, elapsed, correlation

    def evaluate(self, points):
        fluctuation, derivative, elapsed, correlation = self._project(points)
        return self.mean+correlation*(fluctuation+elapsed*derivative)

    def correction(self, points):
        fluctuation, derivative, elapsed, correlation = self._project(points)
        return (correlation-1)*fluctuation+correlation*elapsed*derivative
