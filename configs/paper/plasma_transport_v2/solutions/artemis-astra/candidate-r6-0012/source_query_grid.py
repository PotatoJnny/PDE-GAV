"""Nested source sampling with native-odd and representative-even guards.

Thirty-three stratified native odd-grid probes advance each source slice.
Their exact values survive reconstruction and are reused by exact fallback.
Accepted columns share additive weighted Nyquist and native-probe budgets.
The widened256 band additionally receives29 continuous validation probes.
The probe defect budget is shared over both stages and capped absolutely;
the unprobed old-band tail and new-band/512 tail retain separate bounds.
"""
import numpy as np

PROBE_ABSOLUTE_CAP = 1e-7


def budget_subset(contributions, capacities, eligible=None):
    contributions = np.asarray(contributions, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    valid = np.isfinite(contributions).all(axis=0)
    valid &= (contributions >= 0.).all(axis=0)
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)
    selected = np.zeros(contributions.shape[1], dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices) or np.any(capacities <= 0.):
        return selected
    safe = capacities*(1.-64.*np.finfo(float).eps)
    costs = np.max(contributions[:, indices]/capacities[:, None], axis=0)
    order = indices[np.argsort(costs, kind='stable')]
    used = np.zeros(len(capacities))
    for index in order:
        if np.all(used+contributions[:, index] <= safe):
            selected[index] = True
            used += contributions[:, index]
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
        self.check_x = x[0]+self.length*(np.arange(33)+.3819660112501051)/33
        self.frequencies = {n: np.fft.fftfreq(n, 1/n) for n in (256, 512)}
        self.interpolation = {n: np.exp(2j*np.pi*(self.check_x-x[0])[:, None]
                                       /self.length*self.frequencies[n])
                              for n in (256, 512)}
        self.extension_rng = np.random.default_rng(277803)
        self.native_rng = np.random.default_rng(901371)
        self.cells = [np.arange(start, min(start+6, self.nv))
                      for start in range(0, self.nv, 6)]
        self.representatives = np.array([cell[min(2+i%2, len(cell)-1)]
                                         for i, cell in enumerate(self.cells)])
        self.info = dict(solver_nx=1024, source_query_nx=256, slices=0,
                         accepted256_columns=0, accepted512_columns=0,
                         exact_columns=0, probe_points=0,
                         tail_tolerance=1e-6, shifted_tolerance=1e-8,
                         nyquist_tolerance=1e-12,
                         guard_points=0, guard_failed_cells=0,
                         guard_tolerance=1e-10,
                         extension_probe_points=0, extension_probed_columns=0,
                         extension_rejected_columns=0,
                         extension_probe_tolerance=1e-10,
                         native_probe_points=0, native_preserved_points=0,
                         native_fallback_reused_points=0,
                         native_probe_strategy='33_stratified_odd_points_per_slice',
                         probe_absolute_cap=PROBE_ABSOLUTE_CAP,
                         method='nested_representative_native_odd_checked_bandwidth')

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
        bounds = np.floor(np.arange(34)*512/33).astype(int)
        slots = np.array([self.native_rng.integers(a, b)
                          for a, b in zip(bounds[:-1], bounds[1:])])
        native_indices = 1+2*slots
        native_mask = np.zeros(self.nx, dtype=bool)
        native_mask[native_indices] = True
        self.check_x = self.x[native_indices]
        self.interpolation = {n: np.exp(2j*np.pi*(self.check_x-self.x[0])[:, None]
                                       /self.length*self.frequencies[n])
                              for n in (256, 512)}
        extension_x = self.x[0]+self.length*(np.arange(29)+self.extension_rng.random(29))/29
        quarter = raw_query(shaped[::4].reshape(-1, 3), reserve=reserve).reshape(256, self.nv)
        values = np.empty((1024, self.nv))
        values[::4] = quarter
        finite = np.isfinite(quarter).all(axis=0)
        scale = max(float(np.mean(np.abs(quarter[:, finite])@self.weight[finite]))
                    *self.duration, 1e-12) if finite.any() else 1e-12
        capacities = scale*np.array([1e-6, 1e-12, 1e-8])
        capacities[2] = min(capacities[2], PROBE_ABSOLUTE_CAP)
        used = np.zeros(3)
        available = np.arange(self.nv)
        even_cached = np.zeros(self.nv, dtype=bool)
        guard_eligible = np.zeros(self.nv, dtype=bool)
        # Each cell receives all512 exact even-x samples at one alternating
        # interior velocity. Quarter-grid values already queried are reused.
        representatives = self.representatives
        extra = shaped[2::4][:, representatives].reshape(-1, 3)
        exact_extra = raw_query(extra, reserve=reserve).reshape(256, len(representatives))
        values[2::4, representatives] = exact_extra
        even_cached[representatives] = True
        coarse_coeff = np.fft.fft(quarter[:, representatives], axis=0)/256
        coarse_coeff[128] = 0.
        half_coeff = np.zeros((512, len(representatives)), dtype=np.complex128)
        half_coeff[:128] = coarse_coeff[:128]
        half_coeff[-127:] = coarse_coeff[-127:]
        predicted_extra = (np.fft.ifft(half_coeff, axis=0).real*512)[1::2]
        local_scale = .5*(np.mean(np.abs(quarter[:, representatives]), axis=0)
                          +np.mean(np.abs(exact_extra), axis=0))
        local_defect = np.mean(np.abs(predicted_extra-exact_extra), axis=0)
        guard_passed = (np.isfinite(local_scale) & np.isfinite(local_defect)
                        & (local_defect <= 1e-10*local_scale))
        for cell, passed in zip(self.cells, guard_passed):
            guard_eligible[cell] = passed
        self.info['guard_points'] += len(extra)
        self.info['guard_failed_cells'] += int(np.count_nonzero(~guard_passed))
        self.info['last_guard_failed_cells'] = int(np.count_nonzero(~guard_passed))
        self.info['last_guard_max_relative_defect'] = float(np.max(
            np.divide(local_defect, local_scale, out=np.zeros_like(local_defect),
                      where=local_scale > 0.))) if len(local_scale) else 0.
        probe_cache = np.empty((len(self.check_x), self.nv))
        probed = np.zeros(self.nv, dtype=bool)
        accepted_counts = []
        for n in (256, 512):
            if not len(available):
                accepted_counts.append(0)
                continue
            if n == 512:
                missing = available[~even_cached[available]]
                if len(missing):
                    new = shaped[2::4][:, missing].reshape(-1, 3)
                    values[2::4, missing] = raw_query(new, reserve=reserve).reshape(256, len(missing))
                    even_cached[missing] = True
            samples = values[::1024//n, available]
            coefficients = np.fft.fft(samples, axis=0)/n
            weighted = self.duration*self.weight[available]
            cutoff = n//2 if n == 256 else n//4
            tail = np.sum(np.abs(coefficients[np.abs(self.frequencies[n]) >= cutoff]), axis=0)*weighted
            if n == 256:
                old_tail = np.sum(np.abs(coefficients[np.abs(self.frequencies[n]) >= 64]), axis=0)*weighted
            nyquist = np.abs(coefficients[n//2])*weighted
            spectral = np.stack((tail, nyquist))
            eligible = np.isfinite(samples).all(axis=0)
            if n == 256:
                eligible &= guard_eligible[available]
            selected = budget_subset(spectral, (capacities-used)[:2], eligible)
            coefficients[n//2] = 0.
            defect = np.zeros(len(available))
            indices = np.flatnonzero(selected)
            if len(indices):
                columns = available[indices]
                missing = columns[~probed[columns]]
                if len(missing):
                    probe = np.column_stack((np.full(len(self.check_x)*len(missing), points[0, 0]),
                                             np.repeat(self.check_x, len(missing)),
                                             np.tile(self.v[missing], len(self.check_x))))
                    probe_cache[:, missing] = raw_query(probe, reserve=reserve).reshape(len(self.check_x), len(missing))
                    values[np.ix_(native_indices, missing)] = probe_cache[:, missing]
                    probed[missing] = True
                    self.info['probe_points'] += len(probe)
                    self.info['native_probe_points'] += len(probe)
                    self.info['native_preserved_points'] += len(probe)
                predicted = (self.interpolation[n]@coefficients[:, indices]).real
                defect[indices] = np.mean(np.abs(predicted-probe_cache[:, columns]), axis=0)*weighted[indices]
                available_budget = capacities-used
                if n == 256:
                    available_budget = available_budget.copy()
                    available_budget[2] = min(available_budget[2], scale*1e-10)
                selected = budget_subset(np.vstack((spectral, defect)), available_budget, selected)
            if n == 256:
                # Columns below this additive threshold jointly fit the old
                # tail budget. Probe only columns using the enlarged band.
                extension = selected & (old_tail > scale*1e-6/self.nv
                                         *(1.-64.*np.finfo(float).eps))
                indices = np.flatnonzero(extension)
                if len(indices):
                    columns = available[indices]
                    check_x = extension_x
                    probe = np.column_stack((np.full(29*len(indices), points[0, 0]),
                                             np.repeat(check_x, len(indices)),
                                             np.tile(self.v[columns], 29)))
                    exact = raw_query(probe, reserve=reserve).reshape(29, len(indices))
                    interpolation = np.exp(2j*np.pi*(check_x-self.x[0])[:, None]
                                           /self.length*self.frequencies[n])
                    predicted = (interpolation@coefficients[:, indices]).real
                    extra_defect = np.mean(np.abs(predicted-exact), axis=0)*weighted[indices]
                    defect[indices] = np.maximum(defect[indices], extra_defect)
                    selected = budget_subset(np.vstack((spectral, defect)), available_budget, selected)
                    self.info['probe_points'] += len(probe)
                    self.info['extension_probe_points'] += len(probe)
                    self.info['extension_probed_columns'] += len(indices)
                    self.info['extension_rejected_columns'] += int(np.count_nonzero(extension & ~selected))
                self.info['last_low_band_tail'] = float(old_tail[selected & ~extension].sum()/scale)
            indices = np.flatnonzero(selected)
            accepted_counts.append(len(indices))
            if len(indices):
                padded = np.zeros((1024, len(indices)), dtype=np.complex128)
                padded[:n//2] = coefficients[:n//2, indices]
                padded[-n//2+1:] = coefficients[-n//2+1:, indices]
                reconstructed = np.fft.ifft(padded, axis=0).real*1024
                unknown = np.flatnonzero((np.arange(1024) % (1024//n) != 0)
                                         & ~native_mask)
                columns = available[indices]
                cached = even_cached[columns]
                if np.any(~cached):
                    values[np.ix_(unknown, columns[~cached])] = reconstructed[np.ix_(unknown, np.flatnonzero(~cached))]
                if np.any(cached):
                    odd = np.flatnonzero((np.arange(1024) % 2 == 1) & ~native_mask)
                    values[np.ix_(odd, columns[cached])] = reconstructed[np.ix_(odd, np.flatnonzero(cached))]
                used += np.vstack((spectral, defect))[:, selected].sum(axis=1)
            available = available[~selected]
        if len(available):
            # Preserve exact native probes and query each other odd point once.
            required = np.ones((512, len(available)), dtype=bool)
            required[(native_indices-1)//2] = ~probed[available]
            odd_points = shaped[1::2][:, available][required]
            locations = (np.arange(1, 1024, 2)[:, None]*self.nv+available[None, :])[required]
            if len(locations):
                values.ravel()[locations] = raw_query(odd_points, reserve=reserve)
            self.info['native_fallback_reused_points'] += int(33*np.count_nonzero(probed[available]))
        self.info['slices'] += 1
        self.info['accepted256_columns'] += accepted_counts[0]
        self.info['accepted512_columns'] += accepted_counts[1]
        self.info['exact_columns'] += len(available)
        self.info.update(last_time=float(points[0, 0]), last_accepted256=accepted_counts[0],
                         last_accepted512=accepted_counts[1], last_exact=len(available),
                         last_diagnostics=(used/scale).tolist(), last_scale=scale)
        return values.ravel()
