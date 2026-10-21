"""Held-out public-preview gate for coherent initial g0 mass normalization.

No scoring query is used. The independent velocity quadrature choice remains
the caller's responsibility. This optional check falls back to the original
projection when public previews, remaining time, or evidence are inadequate.
"""
from __future__ import annotations
import math
import numpy as np

_SCOUT_SEED = 731924
_CURRENT_SCALE = .4788345668393978
_ENERGY_SCALE = 2.930763214320111


def scout_indices(nx, ny, count=128):
    """A nested deterministic sample from four interleaved spatial folds."""
    if count not in (128, 256) or nx < 8 or ny < 8 or nx * ny < 2*count:
        raise ValueError('insufficient public preview sites')
    ix, iy = np.divmod(np.arange(nx * ny), ny)
    fold = (ix + 2 * iy) % 4
    permutation = np.random.default_rng(_SCOUT_SEED).permutation(nx * ny)
    selected = []
    for group in range(4):
        sites = permutation[fold[permutation] == group]
        if len(sites) < count//4:
            raise ValueError('insufficient spatial fold')
        selected.extend(sites[:count//4])
    return np.asarray(selected), np.repeat(np.arange(4), count//4)


def _rms(value):
    value = np.asarray(value)
    return float(np.sqrt(np.mean(np.sum(value * value, axis=-1))))


def _rounding_margin(value):
    # Public float32 previews can be stored in a float64 container. Use an
    # explicit half-ULP bound, independent of the container's dtype.
    value = np.abs(np.asarray(value, dtype=np.float32))
    return .5 * np.spacing(value).astype(np.float64)


def assess_normalization(moments, rho, momentum, energy, variance, folds):
    """Assess conditional raw [mass,jx,jy,KE] against same-point previews."""
    moments = np.asarray(moments, float)
    rho = np.asarray(rho, float).reshape(-1)
    momentum = np.asarray(momentum, float)
    energy = np.asarray(energy, float).reshape(-1)
    folds = np.asarray(folds)
    diagnostics = {'enabled': False, 'folds': [], 'minimum_proxy_gain': .20,
                   'preview_sites': len(rho)}
    if (moments.shape != (len(rho), 4) or momentum.shape != (len(rho), 2)
            or energy.shape != rho.shape or folds.shape != rho.shape
            or len(rho) not in (128, 256) or not np.isfinite(variance) or variance <= 0):
        diagnostics['reason'] = 'invalid public preview or moment shapes'
        return False, diagnostics
    if (not all(np.isfinite(v).all() for v in (moments, rho, momentum, energy))
            or np.any(rho <= 0) or np.any(moments[:, 0] <= 0)
            or np.any(energy <= 0)):
        diagnostics['reason'] = 'nonfinite or nonpositive public initial data'
        return False, diagnostics
    raw = moments.copy()
    # Incumbent replaces H00 with exact rho while retaining the higher
    # Hermites. Its physical energy therefore includes this trace adjustment.
    raw[:, 3] += variance * (1. - raw[:, 0])
    raw *= rho[:, None]
    normalized = moments / moments[:, :1] * rho[:, None]
    current_margin = _rounding_margin(momentum)
    energy_margin = _rounding_margin(energy[:, None])
    good = True
    for group in range(4):
        mask = folds == group
        if mask.sum() != len(rho)//4:
            diagnostics['reason'] = 'incomplete spatial folds'
            return False, diagnostics
        rj = _rms(raw[mask, 1:3] - momentum[mask])
        nj = _rms(normalized[mask, 1:3] - momentum[mask])
        re = _rms(raw[mask, 3:4] - energy[mask, None])
        ne = _rms(normalized[mask, 3:4] - energy[mask, None])
        # An error-norm difference can change by twice the target rounding
        # radius. Require current gain beyond that conservative margin.
        jmargin = 2. * _rms(current_margin[mask])
        emargin = 2. * _rms(energy_margin[mask])
        rp = rj / _CURRENT_SCALE + re / _ENERGY_SCALE
        np_ = nj / _CURRENT_SCALE + ne / _ENERGY_SCALE
        passed = bool(np_ <= .8 * rp and rj - nj > max(jmargin, 1.e-12)
                      and ne <= re + emargin)
        diagnostics['folds'].append({'fold': group, 'raw_current_rms': rj,
            'normalized_current_rms': nj, 'raw_energy_rms': re,
            'normalized_energy_rms': ne, 'proxy_ratio': np_ / max(rp, 1.e-300),
            'current_gain_margin': jmargin, 'energy_rounding_margin': emargin,
            'passed': passed})
        good &= passed
    diagnostics['enabled'] = bool(good)
    diagnostics['reason'] = ('all spatial folds support normalization' if good
                             else 'insufficient held-out normalization gain')
    return bool(good), diagnostics


def choose_normalization(get, *, arrays, theta, variance, order, remaining_time):
    """Optional bounded g0 scouts using the existing initial quadrature rule."""
    fallback = {'enabled': False, 'reason': 'optional preview gate unavailable'}
    try:
        # Preparation still needs its entire initial projection and source
        # solve. Do not start an optional request close to the shared deadline.
        if remaining_time is None or remaining_time() < 180.:
            return False, {**fallback, 'reason': 'insufficient remaining case budget'}
        x = np.asarray(arrays['x'], float).reshape(-1)
        y = np.asarray(arrays['y'], float).reshape(-1)
        shape = (len(x), len(y))
        if (np.asarray(arrays['rho0']).shape != shape
                or np.asarray(arrays['momentum0']).shape != shape + (2,)
                or np.asarray(arrays['kinetic_energy0']).shape != shape):
            return False, {**fallback, 'reason': 'invalid public preview shapes'}
        selected, folds = scout_indices(*shape, count=256)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        xy = np.column_stack((xx.ravel(), yy.ravel()))[selected]
        rho = np.asarray(arrays['rho0'], float).ravel()[selected]
        momentum = np.asarray(arrays['momentum0'], float).reshape(-1, 2)[selected]
        energy = np.asarray(arrays['kinetic_energy0'], float).ravel()[selected]
        if not all(np.isfinite(v).all() for v in (xy, rho, momentum, energy)):
            return False, {**fallback, 'reason': 'nonfinite public preview'}
        # For very high quadrature orders, the optional check is needlessly
        # costly. Existing quadrature/projection semantics remain unchanged.
        order = int(order)
        if order < 4 or order > 48 or not np.isfinite(theta) or theta <= 0:
            return False, {**fallback, 'reason': 'optional scout order unsupported'}
        z, w = np.polynomial.hermite_e.hermegauss(order)
        v = z * math.sqrt(theta)
        weights = w * np.exp(z*z/2) * math.sqrt(theta)
        vx, vy = np.meshgrid(v, v, indexing='ij')
        basis = np.stack((np.ones(vx.size), vx.ravel(), vy.ravel(),
            .5 * (vx*vx + vy*vy).ravel())) * np.outer(weights, weights).ravel()
        # The caller splits by the actual public request cap and owns its
        # per-chunk SDK deadline. Q24/Q32 fit one request; Q48 needs three.
        if remaining_time() < 175.:
            return False, {**fallback, 'reason': 'optional scout budget exhausted'}
        points = np.empty((len(xy), order*order, 4))
        points[:, :, :2] = xy[:, None, :]
        points[:, :, 2] = vx.ravel(); points[:, :, 3] = vy.ravel()
        values = np.asarray(get('g0', points.reshape(-1, 4)), float)
        if values.shape != (len(xy) * order * order,):
            return False, {**fallback, 'reason': 'invalid optional g0 response'}
        moments = values.reshape(len(xy), -1) @ basis.T
        enabled, report = assess_normalization(moments, rho, momentum, energy,
                                               variance, folds)
        report['order'] = order
        return enabled, report
    except Exception as exc:
        # Optional scouts must also catch the caller's local deadline class.
        # The unchanged mandatory preparation remains responsible for errors.
        return False, {**fallback, 'reason': type(exc).__name__ + ': ' + str(exc)}
