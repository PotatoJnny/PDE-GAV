"""Per-velocity guarded spatial source sampling for fixed1024 evolution.

Every velocity receives exact even-x source values.  Only independently
validated columns are Fourier reconstructed at odd x.  Rejected columns
receive exact odd-x values.  The summed weighted diagnostics of accepted
columns obey the same budgets as the predecessor's whole-slice gate.
"""
import numpy as np
from scipy.linalg import qr, solve_triangular, svdvals


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
        self.info.update(cur_attempts=0,cur_accepted=0,cur_fallbacks=0,
                         cur_interpolated_columns=0,cur_guard_columns=16,
                         cur_native_guard_rows=17,cur_condition_limit=1e14,
                         method='fourier_then_guarded_parity_cur')

    def parity_extension(self,shaped,even,rejected,raw_query,reserve,capacity):
        """Guarded column interpolation on remaining native odd-x values.

        The source-only fit sees even rows. Held-out full odd columns and
        independent native odd rows check its extension. These finite checks
        are diagnostics, not a uniform certificate for unseen coordinates.
        Every returned exact point is cached for rejection fallback.
        """
        count=len(rejected)
        odd_points=shaped[1::2][:,rejected]
        known=np.empty((512,count))
        seen=np.zeros((512,count),dtype=bool)
        diag=dict(rank=0,condition=0.,column_defect=0.,row_defect=0.,
                  guard_bound=0.,accepted=False,reason='not_attempted')

        def fetch(mask):
            rows,columns=np.nonzero(mask&~seen)
            if len(rows):
                # Keep query exceptions outside numerical fallback handling:
                # cooperative budget exhaustion must reach the caller.
                returned=raw_query(odd_points[rows,columns],reserve=reserve)
                known[rows,columns]=returned
                seen[rows,columns]=True

        def exact(reason):
            fetch(np.ones_like(seen))
            diag['reason']=reason
            if diag['rank']:
                self.info['cur_fallbacks']+=1
            return known,0,diag

        if count<96 or not np.isfinite(capacity) or capacity<=0:
            return exact('small_or_exhausted_capacity')
        training=even[:,rejected]
        weights=self.weight[rejected]
        if not np.isfinite(training).all() or not np.isfinite(weights).all() or np.any(weights<=0):
            return exact('nonfinite_training')
        self.info['cur_attempts']+=1
        matrix=training*weights
        try:
            _,triangular,pivots=qr(matrix,mode='economic',pivoting=True,check_finite=False)
            maximum=min(384,count-32,triangular.shape[0])
            levels=[r for r in (1,2,4,8,16,32,64,96,128,160,192,224,256,288,320,352,384) if r<=maximum]
            rank=levels[-1]
            # A conservative training-error bound chooses rank before any
            # validation observation. This is not used as a parity certificate.
            for proposal in levels:
                bound=np.sqrt(np.sum(triangular[proposal:]**2)*count/512)*self.duration
                if bound<=capacity/64:
                    rank=proposal
                    break
            diag['rank']=rank
            singular=svdvals(triangular[:rank,:rank],check_finite=False)
            condition=float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf
            diag['condition']=condition
            if not np.isfinite(condition) or condition>1e14:
                return exact('ill_conditioned')
            coefficients=np.empty((rank,count))
            coefficients[:,pivots]=solve_triangular(triangular[:rank,:rank],triangular[:rank],check_finite=False)
        except (np.linalg.LinAlgError,FloatingPointError):
            return exact('linear_algebra_failure')
        if not np.isfinite(coefficients).all() or np.max(abs(coefficients))>8.:
            return exact('unstable_coefficients')
        basis=pivots[:rank]
        heldout=pivots[rank:rank+16]
        mask=np.zeros_like(seen);mask[:,np.r_[basis,heldout]]=True
        fetch(mask)
        if not np.isfinite(known[mask]).all():
            return exact('nonfinite_basis_or_guard')
        prediction=((known[:,basis]*weights[basis])@coefficients)/weights
        if not np.isfinite(prediction).all():
            return exact('nonfinite_prediction')
        column_error=np.mean(abs(prediction[:,heldout]-known[:,heldout]),axis=0)*weights[heldout]*self.duration
        column_defect=float(np.mean(column_error)*(count-rank))
        # Deterministic spatial guards are separate from all even training
        # rows and sample every remaining velocity column.
        guard_rows=np.floor((np.arange(17)+.3819660112501051)*512/17).astype(int)
        mask=np.zeros_like(seen);mask[guard_rows]=True
        fetch(mask)
        if not np.isfinite(known[mask]).all():
            return exact('nonfinite_row_guard')
        row_error=prediction[guard_rows]-known[guard_rows]
        row_error[:,np.r_[basis,heldout]]=0.
        row_defect=float(np.mean(abs(row_error),axis=0)@(weights*self.duration))
        bound=max(column_defect,row_defect)
        diag.update(column_defect=column_defect,row_defect=row_defect,guard_bound=bound)
        if not np.isfinite(bound) or bound>capacity*(1.-64*np.finfo(float).eps):
            return exact('independent_guard_failure')
        prediction[seen]=known[seen]
        interpolated=count-rank-len(heldout)
        diag.update(accepted=True,reason='accepted')
        self.info['cur_accepted']+=1
        self.info['cur_interpolated_columns']+=interpolated
        return prediction,interpolated,diag

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
            available=max(0.,capacities[2]-defect[selected].sum())
            odd_values,cur_columns,cur_diagnostic=self.parity_extension(
                shaped,even,rejected,raw_query,reserve,available)
            values[1::2,rejected]=odd_values
        else:
            cur_columns=0
            cur_diagnostic=dict(rank=0,accepted=False,guard_bound=0.,reason='fourier_complete')
        self.info['slices'] += 1
        interpolated_count=len(indices)+cur_columns
        self.info['interpolated_slices'] += int(interpolated_count==self.nv)
        self.info['exact_slices'] += int(interpolated_count==0)
        self.info['mixed_slices'] += int(0<interpolated_count<self.nv)
        self.info['interpolated_columns'] += len(indices)+cur_columns
        self.info['exact_columns'] += len(rejected)-cur_columns
        diagnostics = np.array([tail[selected].sum(), nyquist[selected].sum(), defect[selected].sum()])/scale
        if cur_diagnostic.get('accepted',False):
            diagnostics[2]+=cur_diagnostic['guard_bound']/scale
        self.info.update(last_time=float(points[0, 0]), last_accepted_columns=interpolated_count,
                         last_fourier_accepted_columns=len(indices),
                         last_tail=float(diagnostics[0]), last_nyquist=float(diagnostics[1]),
                         last_shifted_defect=float(diagnostics[2]), last_scale=scale,
                         last_cur=cur_diagnostic,last_cur_columns=cur_columns)
        return values.ravel()
