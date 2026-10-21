"""Per-velocity guarded spatial source sampling for fixed1024 evolution.

Every velocity receives exact even-x source values.  Only independently
validated columns are Fourier reconstructed at odd x.  Rejected columns
receive exact odd-x values.  The summed weighted diagnostics of accepted
columns obey the same budgets as the predecessor's whole-slice gate.
"""
import numpy as np


def budget_subset(contributions, capacities, eligible=None):
    """Greedily retain inexpensive columns within every additive budget.

    This is an allocation heuristic, not an unseen-coordinate certificate.
    The final inequalities are checked directly, including roundoff.
    """
    contributions = np.asarray(contributions, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    nv = contributions.shape[1]
    valid = np.isfinite(contributions).all(axis=0)
    valid &= (contributions >= 0.).all(axis=0)
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)
    indices = np.flatnonzero(valid)
    selected = np.zeros(nv, dtype=bool)
    if not len(indices):
        return selected
    safe = capacities*(1.-32.*np.finfo(float).eps)
    if np.all(contributions[:, indices].sum(axis=1) <= safe):
        selected[indices] = True
        return selected
    costs = np.max(contributions[:, indices]/capacities[:, None], axis=0)
    order = indices[np.argsort(costs, kind='stable')]
    cumulative = np.cumsum(contributions[:, order], axis=1)
    # All contributions are positive: once the prefix fails, longer prefixes
    # fail too.  Iterating remaining columns can fill the leftover capacity.
    prefix = np.flatnonzero(np.any(cumulative > safe[:, None], axis=0))
    count = int(prefix[0]) if len(prefix) else len(order)
    selected[order[:count]] = True
    used = contributions[:, order[:count]].sum(axis=1)
    for index in order[count:]:
        if np.all(used+contributions[:, index] <= safe):
            selected[index] = True
            used += contributions[:, index]
    # Verify with numpy's final summation order, rather than trusting the
    # incremental summation used to choose a subset.
    while selected.any() and np.any(contributions[:, selected].sum(axis=1) > capacities):
        active = np.flatnonzero(selected)
        cost = np.max(contributions[:, active]/capacities[:, None], axis=0)
        selected[active[np.argmax(cost)]] = False
    return selected


class SourceQueryGrid:
    def __init__(self, x, velocity, quadrature, tmin, tmax):
        self.x = np.asarray(x)
        self.v = np.asarray(velocity)
        self.nx, self.nv = len(x), len(velocity)
        if self.nx != 1024:
            raise ValueError('SourceQueryGrid requires fixed1024 evolution')
        self.duration = tmax-tmin
        self.weight = np.asarray(quadrature)*(1+np.abs(velocity)**3)
        self.length = (x[1]-x[0])*self.nx
        self.check_x = x[0]+self.length*(np.arange(17)+.3819660112501051)/17
        self.frequencies = np.fft.fftfreq(512, 1/512)
        self.interpolation = np.exp(2j*np.pi*(self.check_x-x[0])[:, None]
                                    /self.length*self.frequencies)
        self.info = dict(solver_nx=1024, source_query_nx=512, slices=0,
                         interpolated_slices=0, exact_slices=0,
                         mixed_slices=0, interpolated_columns=0,
                         exact_columns=0, probe_points=0,
                         tail_tolerance=1e-6, shifted_tolerance=1e-8,
                         nyquist_tolerance=1e-12, cutoff=128,
                         method='per_velocity_additive_spatial_budgets')

    def matches(self, points):
        if points.ndim != 2 or points.shape != (self.nx*self.nv, 3):
            return False
        if not np.all(points[:, 0] == points[0, 0]):
            return False
        shaped = points.reshape(self.nx, self.nv, 3)
        return (np.array_equal(shaped[:, 0, 1], self.x)
                and np.array_equal(shaped[0, :, 2], self.v))

    def sample(self, points, raw_query, reserve):
        shaped = points.reshape(self.nx, self.nv, 3)
        even = raw_query(shaped[::2].reshape(-1, 3), reserve=reserve).reshape(512, self.nv)
        coefficients = np.fft.fft(even, axis=0)/512
        finite = np.isfinite(even).all(axis=0)
        # An invalid column remains exact and makes the enclosing solver's
        # existing nonfinite-source check fail, with no manufactured values.
        scale = max(float(np.mean(np.abs(even[:, finite])@self.weight[finite]))
                    *self.duration, 1e-12) if finite.any() else 1e-12
        weighted = self.duration*self.weight
        tail = np.sum(np.abs(coefficients[np.abs(self.frequencies) >= 128]), axis=0)*weighted
        nyquist = np.abs(coefficients[256])*weighted
        capacities = scale*np.array([1e-6, 1e-12, 1e-8])
        spectral = np.stack((tail, nyquist))
        selected = budget_subset(spectral, capacities[:2], finite)
        coefficients[256] = 0.
        defect = np.zeros(self.nv)
        indices = np.flatnonzero(selected)
        if len(indices):
            # Even an exactly zero column is checked: mode512 can disappear
            # at every even-x point yet remain substantial at odd x.
            probe = np.column_stack((np.full(17*len(indices), points[0, 0]),
                                     np.repeat(self.check_x, len(indices)),
                                     np.tile(self.v[indices], 17)))
            exact = raw_query(probe, reserve=reserve).reshape(17, len(indices))
            predicted = (self.interpolation@coefficients[:, indices]).real
            defect[indices] = np.mean(np.abs(predicted-exact), axis=0)*weighted[indices]
            selected = budget_subset(np.vstack((spectral, defect)), capacities, selected)
            self.info['probe_points'] += len(probe)
        indices = np.flatnonzero(selected)
        rejected = np.flatnonzero(~selected)
        values = np.empty((1024, self.nv))
        values[::2] = even
        if len(indices):
            padded = np.zeros((1024, len(indices)), dtype=np.complex128)
            padded[:256] = coefficients[:256, indices]
            padded[-255:] = coefficients[-255:, indices]
            reconstructed = np.fft.ifft(padded, axis=0).real*1024
            values[1::2, indices] = reconstructed[1::2]
        if len(rejected):
            # Boolean/fancy indexing is explicit on both spatial and velocity
            # axes, so the flattened query keeps (x,v) order unambiguous.
            odd_points = shaped[1::2][:, rejected].reshape(-1, 3)
            values[1::2, rejected] = raw_query(odd_points, reserve=reserve).reshape(512, len(rejected))
        self.info['slices'] += 1
        self.info['interpolated_slices'] += int(not len(rejected))
        self.info['exact_slices'] += int(not len(indices))
        self.info['mixed_slices'] += int(bool(len(indices) and len(rejected)))
        self.info['interpolated_columns'] += len(indices)
        self.info['exact_columns'] += len(rejected)
        diagnostics = np.array([tail[selected].sum(), nyquist[selected].sum(), defect[selected].sum()])/scale
        self.info.update(last_time=float(points[0, 0]), last_accepted_columns=len(indices),
                         last_tail=float(diagnostics[0]), last_nyquist=float(diagnostics[1]),
                         last_shifted_defect=float(diagnostics[2]), last_scale=scale)
        return values.ravel()
