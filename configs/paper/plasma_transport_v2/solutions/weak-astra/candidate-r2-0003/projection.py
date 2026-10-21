"""Continuous Fourier projection and conservative snapshot admissibility guard."""
from __future__ import annotations

import numpy as np
from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind

FIELDS = ("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
          "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y")


def moment_pairs(degree):
    return tuple((a, total-a) for total in range(degree+1)
                 for a in range(total, -1, -1))


def resize_modes(modes, size):
    """Crop/pad normalized Fourier coefficients without collocation aliasing."""
    modes = np.asarray(modes)
    target = (size, size) if np.isscalar(size) else tuple(size)
    if len(target) != 2 or any(type(n) is not int or n < 2 for n in target):
        raise ValueError("target Fourier shape must contain two positive integers")
    result = np.zeros(modes.shape[:-2]+target, dtype=complex)
    mappings = []
    for source_n, target_n in zip(modes.shape[-2:], target):
        source_freq = np.rint(np.fft.fftfreq(source_n)*source_n).astype(int)
        target_freq = np.rint(np.fft.fftfreq(target_n)*target_n).astype(int)
        lookup = {frequency: index for index, frequency in enumerate(source_freq)
                  if not (source_n % 2 == 0 and index == source_n//2)}
        pairs = [(lookup[frequency], index)
                 for index, frequency in enumerate(target_freq)
                 if frequency in lookup
                 and not (target_n % 2 == 0 and index == target_n//2)]
        mappings.append(tuple(np.asarray(v, dtype=int) for v in zip(*pairs)))
    (sx, tx), (sy, ty) = mappings
    result[..., tx[:, None], ty[None, :]] = modes[..., sx[:, None], sy[None, :]]
    return result


def modes_to_grid(modes, size=None):
    if size is not None:
        modes = resize_modes(modes, size)
    return np.fft.ifft2(modes).real*np.prod(modes.shape[-2:])


def low_initial(initial, nx):
    """Initial data for the low-grid evolution, with high modes removed."""
    return modes_to_grid(initial["modes"], nx)


def pressure_components(moments):
    rho = moments[..., 0]
    safe = np.maximum(rho, 1e-300)
    return (moments[..., 3]-moments[..., 1]**2/safe,
            moments[..., 4]-moments[..., 1]*moments[..., 2]/safe,
            moments[..., 5]-moments[..., 2]**2/safe)


def minimum_pressure_eigenvalue(moments):
    xx, xy, yy = pressure_components(moments)
    return .5*(xx+yy-np.hypot(xx-yy, 2*xy))


def electric_modes(rho_modes, periods):
    shape = rho_modes.shape[-2:]
    kx, ky = np.meshgrid(*[
        2*np.pi*np.fft.fftfreq(n, d=length/n)
        for n, length in zip(shape, periods)], indexing="ij")
    k2 = kx*kx+ky*ky
    phi = np.divide(rho_modes, k2, out=np.zeros_like(rho_modes), where=k2 > 0)
    return np.asarray([-1j*kx*phi, -1j*ky*phi])


class ProjectionHandle:
    """Prepared snapshot interpolation; evaluate only projects existing state."""
    def __init__(self, initial, times=None, snapshots=None, guard=True):
        self.periods = np.asarray(initial["periods"], dtype=float)
        self.interval = np.asarray(initial["interval"], dtype=float)
        self.center = np.asarray(initial["center"], dtype=float)
        initial_modes = np.asarray(initial["modes"], dtype=complex)
        count = initial_modes.shape[0]
        degree = int(round((np.sqrt(8*count+1)-3)/2))
        if count != (degree+1)*(degree+2)//2 or degree < 3:
            raise ValueError("triangular raw moments through degree at least three required")
        self.pairs = moment_pairs(degree)
        self.index = {pair: i for i, pair in enumerate(self.pairs)}
        if snapshots is None:
            times = np.asarray([self.interval[0]], dtype=float)
            prepared_modes = initial_modes[None].copy()
        else:
            times = np.asarray(times, dtype=float)
            snapshots = np.asarray(snapshots, dtype=float)
            if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                    or times[0] != self.interval[0]
                    or np.any(np.diff(times) <= 0) or times[-1] > self.interval[1]
                    or snapshots.ndim != 4 or snapshots.shape[:2] != (len(times), count)):
                raise ValueError("ordered snapshots must begin at the public initial time")
            low_shape = snapshots.shape[-2:]
            full_shape = tuple(max(a, b) for a, b in zip(initial_modes.shape[-2:], low_shape))
            initial_modes = resize_modes(initial_modes, full_shape)
            low_modes = np.fft.fft2(snapshots)/np.prod(low_shape)
            low_modes = resize_modes(low_modes, low_shape)
            low_initial_modes = resize_modes(initial_modes, low_shape)
            correction = initial_modes-resize_modes(low_initial_modes, full_shape)
            prepared_modes = resize_modes(low_modes, full_shape)+correction[None]
            # A solver may have rounded/resampled its initial low state. The
            # public initial reconstruction is the unambiguous exact anchor.
            prepared_modes[0] = initial_modes
        self.times = times
        self.initial_modes = initial_modes
        self.guard_fractions = np.ones(len(times))
        if guard:
            prepared_modes = self._guard_snapshots(prepared_modes)
        self.modes = prepared_modes
        self.electric = np.stack([electric_modes(m[0], self.periods)
                                  for m in self.modes])
        self.wave = [2*np.pi*np.fft.fftfreq(n, d=length/n)
                     for n, length in zip(self.modes.shape[-2:], self.periods)]
        self.closed = False

    def _guard_snapshots(self, modes):
        """One blend fraction per entire snapshot preserves the Poisson pair.

        The positive-density/PSD-pressure cone is convex in raw moments.
        Oversampling checks the continuous trigonometric interpolant at twice
        its original mesh resolution; a small physical margin limits off-grid
        undershoot. The initial state itself is kept exactly.
        """
        oversample = tuple(2*n for n in self.initial_modes.shape[-2:])
        initial_grid = np.moveaxis(modes_to_grid(self.initial_modes[:6], oversample), 0, -1)
        rho_min = float(np.min(initial_grid[..., 0]))
        pressure_min = float(np.min(minimum_pressure_eigenvalue(initial_grid)))
        density_floor = max(1e-10, .01*rho_min)
        pressure_floor = max(0., .01*pressure_min)

        def admissible(raw):
            return (np.isfinite(raw).all()
                    and np.min(raw[..., 0]) >= density_floor
                    and np.min(minimum_pressure_eigenvalue(raw)) >= pressure_floor)

        baseline_valid = admissible(initial_grid)
        for i in range(1, len(modes)):
            if not np.isfinite(modes[i]).all():
                self.guard_fractions[i] = 0.
                modes[i] = self.initial_modes
                continue
            evolved_grid = np.moveaxis(modes_to_grid(modes[i, :6], oversample), 0, -1)
            if admissible(evolved_grid):
                continue
            if not baseline_valid:
                self.guard_fractions[i] = 0.
                modes[i] = self.initial_modes
                continue
            lo, hi = 0., 1.
            increment = evolved_grid-initial_grid
            for _ in range(28):
                fraction = .5*(lo+hi)
                if admissible(initial_grid+fraction*increment):
                    lo = fraction
                else:
                    hi = fraction
            # Keep a modest distance inside the admissible cone.
            fraction = .98*lo
            self.guard_fractions[i] = fraction
            modes[i] = self.initial_modes+fraction*(modes[i]-self.initial_modes)
        return modes

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared solution is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("source macro reconstruction accepts POINTS only")
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError("unknown or duplicate transport field")
        points = np.asarray([(p["t"], p["x"], p["y"]) for p in query.points], dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
                or not np.isfinite(points).all()):
            raise ValueError("finite nonempty t,x,y points are required")
        if np.any(points[:, 0] < self.interval[0]) or np.any(points[:, 0] > self.interval[1]):
            raise ValueError("query leaves prepared public time interval")
        lower = np.maximum(np.searchsorted(self.times, points[:, 0], side="right")-1, 0)
        upper = np.minimum(lower+1, len(self.times)-1)
        duration = self.times[upper]-self.times[lower]
        fraction = np.divide(points[:, 0]-self.times[lower], duration,
                             out=np.zeros(len(points)), where=duration > 0)
        bx = np.exp(1j*points[:, 1, None]*self.wave[0])
        by = np.exp(1j*points[:, 2, None]*self.wave[1])
        raw = np.zeros((len(points), 10))
        field = np.zeros((len(points), 2))
        for indices, weight in ((lower, 1-fraction), (upper, fraction)):
            for frame in np.unique(indices):
                selected = np.flatnonzero((indices == frame) & (weight != 0))
                if not len(selected):
                    continue
                raw[selected] += weight[selected, None]*np.einsum(
                    "cij,ni,nj->nc", self.modes[frame, :10],
                    bx[selected], by[selected]).real
                field[selected] += weight[selected, None]*np.einsum(
                    "cij,ni,nj->nc", self.electric[frame],
                    bx[selected], by[selected]).real
        rho = np.maximum(raw[:, 0], 1e-8)
        ux, uy = raw[:, 1]/rho, raw[:, 2]/rho
        pxx = raw[:, 3]-rho*ux*ux
        pxy = raw[:, 4]-rho*ux*uy
        pyy = raw[:, 5]-rho*uy*uy
        raw_qx = .5*(raw[:, 6]+raw[:, 8])
        raw_qy = .5*(raw[:, 7]+raw[:, 9])
        heat_x = (raw_qx-ux*(1.5*raw[:, 3]+.5*raw[:, 5])-uy*raw[:, 4]
                  +rho*ux*(ux*ux+uy*uy))
        heat_y = (raw_qy-uy*(.5*raw[:, 3]+1.5*raw[:, 5])-ux*raw[:, 4]
                  +rho*uy*(ux*ux+uy*uy))
        momentum_x = raw[:, 1]+rho*self.center[0]
        momentum_y = raw[:, 2]+rho*self.center[1]
        values = dict(zip(FIELDS, (rho, field[:, 0], field[:, 1],
                                  momentum_x, momentum_y, pxx, pxy, pyy,
                                  heat_x, heat_y)))
        if not all(np.isfinite(v).all() for v in values.values()):
            raise FloatingPointError("nonfinite projected macros")
        return AnswerView(kind=query.kind,
                          fields={name: tuple(map(float, values[name])) for name in query.fields},
                          points=query.points, shape=(len(points),))

    def close(self):
        self.closed = True
