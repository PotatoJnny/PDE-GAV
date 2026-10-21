"""Conservative nested source quadrature on1024/2048-point Fourier states.

Every velocity column starts on the incumbent's512-point exact lattice and
independently passes spectral-tail, Nyquist and both shifted-grid checks.
Rejected columns acquire the next nested lattice, through the full state grid.
The inherited17
uniform probes and29 time-rotated stratified probes are checked separately.
No velocity interpolation is used.
"""
import numpy as np


class SourceQueryGrid:
    def __init__(self, x, velocity, quadrature, tmin, tmax):
        self.x = np.asarray(x)
        self.v = np.asarray(velocity)
        self.nx, self.nv = len(x), len(velocity)
        if self.nx not in (1024, 2048):
            raise ValueError('SourceQueryGrid requires1024 or2048 evolution')
        self.levels = tuple(512*2**level for level in range(1+int(np.log2(self.nx//512))))
        self.tmin = float(tmin)
        self.duration = float(tmax-tmin)
        self.weight = np.asarray(quadrature)*(1+np.abs(velocity)**3)
        self.total_weight = max(float(np.sum(self.weight)), 1e-300)
        self.length = (x[1]-x[0])*self.nx
        self.legacy_probe_count = 17
        self.stratified_probe_count = 29
        self.probe_count = self.legacy_probe_count+self.stratified_probe_count
        # One point in each stratum, with a deterministic irrational jitter.
        # A slice-dependent rigid rotation avoids a fixed temporal nullspace.
        index = np.arange(self.stratified_probe_count)
        jitter = .35*(2*np.mod((index+1)*.6180339887498949, 1.)-1.)
        self.probe_fractions = (index+.5+jitter)/self.stratified_probe_count
        self.legacy_fractions = (np.arange(self.legacy_probe_count)+.3819660112501051)/self.legacy_probe_count
        self.frequencies = {}
        for count in self.levels[:-1]:
            frequency = np.fft.fftfreq(count, 1/count)
            self.frequencies[count] = frequency
        self.info = dict(solver_nx=self.nx, source_query_nx='per_velocity', slices=0,
                         interpolated_slices=0, exact_slices=0, probe_points=0,
                         tail_tolerance=1e-6, shifted_tolerance=1e-8,
                         nyquist_tolerance=1e-12, cutoff_fraction=.25,
                         shifted_probe_count=self.probe_count,
                         shifted_probe_geometry='incumbent17_plus_stratified29',
                         source_query_levels=list(self.levels),
                         accepted_column_counts={level: 0 for level in self.levels})

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
        source_time = float(points[0, 0])
        fraction = ((source_time-self.tmin)/self.duration if self.duration else 0.)
        rotation = np.mod(.3819660112501051+np.sqrt(2.)*fraction, 1.)
        check_fraction = np.concatenate((self.legacy_fractions,
                                         np.mod(self.probe_fractions+rotation, 1.)))
        check_x = self.x[0]+self.length*check_fraction
        values = np.empty((self.nx, self.nv))
        exact_probe = None
        active = np.arange(self.nv)
        previous = 0
        accepted_counts = {}
        for count in self.levels:
            stride = self.nx//count
            index = np.arange(0 if previous == 0 else stride,
                              self.nx, stride if previous == 0 else 2*stride)
            query_points = shaped[index[:, None], active[None, :]].reshape(-1, 3)
            values[index[:, None], active[None, :]] = raw_query(
                query_points, reserve=reserve).reshape(len(index), len(active))
            previous = count
            if count == self.nx:
                accepted_counts[count] = len(active)
                break
            coarse = values[::stride, active]
            coefficients = np.fft.fft(coarse, axis=0)/count
            amplitude = np.mean(np.abs(coarse), axis=0)*self.duration
            total_amplitude = float(amplitude@self.weight[active])
            # Allocate the aggregate1e-12 floor only when that floor is active.
            # The weighted sum of these column scales equals the incumbent's
            # normalization, so passing every column implies its aggregate
            # tail/Nyquist/legacy-probe tests pass at the same tolerances.
            addition = max(0., 1e-12-total_amplitude)/self.total_weight
            scale = np.maximum(amplitude+addition, np.finfo(float).tiny)
            frequency = self.frequencies[count]
            tail = np.sum(np.abs(coefficients[np.abs(frequency) >= count//4]), axis=0)*self.duration/scale
            nyquist = np.abs(coefficients[count//2])*self.duration/scale
            accepted = (np.isfinite(coarse).all(axis=0) & np.isfinite(tail)
                        & (tail <= 1e-6) & (nyquist <= 1e-12))
            coefficients[count//2] = 0.
            candidate = np.flatnonzero(accepted)
            if len(candidate):
                if exact_probe is None:
                    probe = np.column_stack((np.full(self.probe_count*self.nv, source_time),
                                             np.repeat(check_x, self.nv),
                                             np.tile(self.v, self.probe_count)))
                    exact_probe = raw_query(probe, reserve=reserve).reshape(self.probe_count, self.nv)
                    self.info['probe_points'] += len(probe)
                interpolation = np.exp(2j*np.pi*check_fraction[:, None]*frequency)
                predicted = (interpolation@coefficients[:, candidate]).real
                residual = np.abs(predicted-exact_probe[:, active[candidate]])
                legacy_defect = np.mean(residual[:self.legacy_probe_count], axis=0)*self.duration/scale[candidate]
                stratified_defect = np.mean(residual[self.legacy_probe_count:], axis=0)*self.duration/scale[candidate]
                defect = np.maximum(legacy_defect, stratified_defect)
                accepted[candidate] &= np.isfinite(defect) & (defect <= 1e-8)
            accepted_local = np.flatnonzero(accepted)
            accepted_counts[count] = len(accepted_local)
            if len(accepted_local):
                padded = np.zeros((self.nx, len(accepted_local)), dtype=np.complex128)
                padded[:count//2] = coefficients[:count//2, accepted_local]
                padded[-(count//2-1):] = coefficients[-(count//2-1):, accepted_local]
                interpolated = np.fft.ifft(padded, axis=0).real*self.nx
                interpolated[::stride] = coarse[:, accepted_local]
                values[:, active[accepted_local]] = interpolated
            active = active[~accepted]
            if not len(active):
                break
        self.info['slices'] += 1
        for count, columns in accepted_counts.items():
            self.info['accepted_column_counts'][count] += columns
        self.info['last_accepted_column_counts'] = accepted_counts
        self.info['last_time'] = float(points[0, 0])
        exact_columns = accepted_counts.get(self.nx, 0)
        self.info['exact_slices'] += int(exact_columns == self.nv)
        self.info['interpolated_slices'] += int(exact_columns < self.nv)
        return values.ravel()
