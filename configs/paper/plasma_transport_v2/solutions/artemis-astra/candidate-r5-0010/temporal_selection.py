"""Select a temporal macro spline using held-out solver-step snapshots.

No oracle calls or scoring coordinates are used. Fourier macro arrays have
shape (time, 5, nx), ordered M0, M1, M2, M3, E and normalized by nx.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline


FLOORS = np.array([1.0041253663782175, .4788345668393978,
                   2.930763214320111, .9900774471118267,
                   .0698185715388132])
SCORE_WEIGHTS = np.array([.125, .25, .25, .25, .125])


def positive_control_certificate(control_modes, background=None):
    """Certify original and, if requested, charge-projected density positive.

    Each B-spline value is a convex combination of control coefficients.
    The Fourier Lipschitz bound covers space between the oversampled nodes.
    """
    modes = np.asarray(control_modes, dtype=np.complex128)
    if modes.ndim != 3 or modes.shape[1] != 5 or not np.isfinite(modes).all():
        return False
    density = modes[:, 0, :]
    means = density[:, 0].real
    if background is None:
        lower_mean = means
        floor = 0.
    else:
        background = float(background)
        if not np.isfinite(background) or background <= 0:
            return False
        lower_mean = np.minimum(means, background)
        floor = .05*background
    harmonics = density.copy()
    harmonics[:, 0] = 0.
    amplitudes = np.sum(np.abs(harmonics), axis=1)
    if np.all(lower_mean-amplitudes > floor):
        return True
    nx = density.shape[1]
    # The candidate uses an even spatial FFT grid. Reject other layouts
    # conservatively rather than silently interpreting a Nyquist mode wrong.
    if nx < 2 or nx % 2:
        return False
    panel_nx = max(256, 8*nx)
    frequencies = np.fft.fftfreq(nx, 1./nx)
    between_grid_bound = (np.pi/panel_nx
                          *np.sum(np.abs(harmonics)*np.abs(frequencies), axis=1))
    for start in range(0, len(density), 32):
        stop = min(start+32, len(density))
        padded = np.zeros((stop-start, panel_nx), dtype=np.complex128)
        padded[:, :nx//2] = harmonics[start:stop, :nx//2]
        padded[:, -nx//2:] = harmonics[start:stop, -nx//2:]
        minimum = (np.fft.ifft(padded, axis=1).real*panel_nx).min(axis=1)
        if np.any(lower_mean[start:stop]+minimum-between_grid_bound[start:stop] <= floor):
            return False
    return True


def centered_grid(modes):
    """Convert normalized Fourier raw moments to physical central moments."""
    modes = np.asarray(modes)
    raw = np.fft.ifft(modes, axis=-1).real*modes.shape[-1]
    n, j, m2, m3, electric = np.moveaxis(raw, 1, 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        velocity = j/n
        pressure = m2-j*velocity
        heat = .5*(m3-3*velocity*m2+2*n*velocity**3)
        central = np.stack((n, j, pressure, heat, electric), axis=1)
    return central


def holdout_defect(predicted_modes, reference_centered):
    predicted = centered_grid(predicted_modes)
    if not np.isfinite(predicted).all():
        return None
    axes = (0, 2)
    numerator = np.mean((predicted-reference_centered)**2, axis=axes)
    denominator = np.mean(reference_centered**2, axis=axes)+FLOORS**2
    errors = np.sqrt(numerator/denominator)
    score = float(errors@SCORE_WEIGHTS)
    if not np.isfinite(score):
        return None
    return dict(score=score, errors=errors.tolist())


def select_temporal_spline(times, modes, check_times, check_modes,
                           background=None):
    """Return (BSpline, diagnostics), retaining linear on insufficient data.

    The check frames must come from already completed, unsaved solver steps.
    A candidate degree replaces the current choice only if its held-out
    central-moment defect is below 0.8 of the current defect and its density
    control coefficients pass a global positivity certificate.
    """
    times = np.asarray(times, dtype=float)
    modes = np.asarray(modes, dtype=np.complex128)
    current = make_interp_spline(times, modes, k=1, axis=0)
    info = dict(chosen_degree=1, holdout_count=0, holdout_errors={},
                controlcert={}, reason='insufficient_history')
    if len(times) < 8:
        return current, info
    check_times = np.asarray(check_times, dtype=float)
    check_modes = np.asarray(check_modes, dtype=np.complex128)
    if (check_times.ndim != 1 or check_modes.shape != (len(check_times), *modes.shape[1:])):
        info['reason'] = 'invalid_holdout_shape'
        return current, info
    # Remove invalid/outside/duplicate/retained frames without changing the
    # already prepared history. Near-coincident frames cannot validate it.
    valid = np.isfinite(check_times) & np.isfinite(check_modes).all(axis=(1, 2))
    valid &= (check_times > times[0]) & (check_times < times[-1])
    left = np.clip(np.searchsorted(times, check_times)-1, 0, len(times)-2)
    distance = np.minimum(abs(check_times-times[left]), abs(check_times-times[left+1]))
    tolerance = 32*np.finfo(float).eps*max(1., abs(times[0]), abs(times[-1]))
    valid &= distance > tolerance
    indices = np.flatnonzero(valid)
    if len(indices):
        _, unique = np.unique(check_times[indices], return_index=True)
        indices = indices[np.sort(unique)]
    check_times, check_modes = check_times[indices], check_modes[indices]
    info['holdout_count'] = len(check_times)
    if len(check_times) < 8:
        info['reason'] = 'insufficient_holdout'
        return current, info
    reference = centered_grid(check_modes)
    if not np.isfinite(reference).all() or np.any(reference[:, 0] <= 0):
        info['reason'] = 'invalid_reference_density'
        return current, info
    current_defect = holdout_defect(current(check_times), reference)
    info['holdout_errors']['1'] = current_defect
    if current_defect is None:
        info['reason'] = 'invalid_linear_defect'
        return current, info
    info['reason'] = 'linear_preferred'
    for degree in (3, 7):
        try:
            candidate = make_interp_spline(times, modes, k=degree, axis=0)
            certified = positive_control_certificate(candidate.c, background)
            info['controlcert'][str(degree)] = bool(certified)
            defect = holdout_defect(candidate(check_times), reference)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            info['controlcert'][str(degree)] = False
            info['holdout_errors'][str(degree)] = None
            continue
        info['holdout_errors'][str(degree)] = defect
        if (certified and defect is not None
                and defect['score'] < .8*current_defect['score']
                and current_defect['score']-defect['score'] > 1e-12):
            current, current_defect = candidate, defect
            info['chosen_degree'] = degree
            info['reason'] = 'heldout_improvement'
    return current, info


def build_spline(times, modes, check_times, check_modes, background=None):
    """Portable integration entrypoint; see select_temporal_spline."""
    return select_temporal_spline(times, modes, check_times, check_modes, background)
