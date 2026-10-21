"""Prepared macroscopic projection onto periodic Poisson compatibility.

This module does not sample or evolve an equation. It corrects the known
density zero mode while preserving j, central P, q_h and E pointwise.
"""
from __future__ import annotations

import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView


def _positive_density_certificate(prepared, background):
    """Certify original and projected trigonometric densities stay positive."""
    modes = np.asarray(prepared.modes[:, 0, :], dtype=np.complex128)
    means = modes[:, 0].real
    harmonics = modes.copy()
    harmonics[:, 0] = 0.
    amplitudes = np.sum(np.abs(harmonics), axis=1)
    lower_mean = np.minimum(means, background)
    floor = .05*background
    if np.all(lower_mean-amplitudes > floor):
        return True
    # Grid minima minus a Fourier Lipschitz bound certify the intervening x.
    nx = modes.shape[1]
    panel_nx = max(256, 8*nx)
    frequencies = np.fft.fftfreq(nx, 1./nx)
    between_grid_bound = np.pi*np.sum(np.abs(harmonics)*np.abs(frequencies), axis=1)/panel_nx
    for start in range(0, len(modes), 32):
        stop = min(start+32, len(modes))
        padded = np.zeros((stop-start, panel_nx), dtype=np.complex128)
        padded[:, :nx//2] = harmonics[start:stop, :nx//2]
        padded[:, -nx//2:] = harmonics[start:stop, -nx//2:]
        minimum = (np.fft.ifft(padded, axis=1).real*panel_nx).min(axis=1)
        if np.any(lower_mean[start:stop]+minimum-between_grid_bound[start:stop] <= floor):
            return False
    # Linear interpolation between stored time slices preserves positivity.
    return True


def _well_conditioned_projection(prepared, background):
    """Conservative scale screen, not a rigorous floating-point error theorem."""
    modes = np.asarray(prepared.modes, dtype=np.complex128)
    floor = .05*background  # The preceding certificate proves this bound.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # Global time bounds also cover interpolation when different quantities
        # attain their largest magnitudes at different stored times.
        bounds = np.max(np.sum(np.abs(modes), axis=2), axis=0)
        means = modes[:, 0, 0].real
        harmonics = np.sum(np.abs(modes[:, 0, 1:]), axis=1)
        old_lower = np.min(np.maximum(means-harmonics, floor))
        new_lower = np.min(np.maximum(background-harmonics, floor))
        j, m2, m3 = bounds[1], bounds[2], bounds[3]
        old_velocity = j/old_lower
        new_velocity = j/new_lower
        shift = np.max(np.abs(background-means))
        velocity_change = shift*j/(old_lower*new_lower)
        pressure = m2+j*old_velocity
        new_m2 = m2+j*velocity_change
        new_m3 = m3+velocity_change*(3*pressure+j*(new_velocity+old_velocity))
        # Bound both reconstruction and the verifier's centering expressions.
        scales = np.stack((bounds[0]+shift+background,
                           pressure, new_m2+j*new_velocity,
                           m3+3*old_velocity*m2+2*j*old_velocity**2,
                           new_m3+3*new_velocity*new_m2+2*j*new_velocity**2))
        roundoff_scale = 100*np.finfo(float).eps*np.max(scales)
    return bool(np.isfinite(roundoff_scale) and roundoff_scale <= 1e-8)


class ChargeProjected:
    def __init__(self, prepared, background):
        self.prepared = prepared
        self.times = np.asarray(prepared.times, dtype=float)
        self.density_shifts = background-np.asarray(prepared.modes[:, 0, 0]).real

    def evaluate(self, query):
        answer = self.prepared.evaluate(query)
        times = np.asarray([point['t'] for point in query.points], dtype=float)
        shift = np.interp(times, self.times, self.density_shifts)
        n = np.asarray(answer.fields['M0'], dtype=float)
        j = np.asarray(answer.fields['M1'], dtype=float)
        m2 = np.asarray(answer.fields['M2'], dtype=float)
        m3 = np.asarray(answer.fields['M3'], dtype=float)
        corrected_n = n+shift
        velocity = j/n
        corrected_velocity = j/corrected_n
        velocity_change = corrected_velocity-velocity
        pressure = m2-j*velocity
        corrected_m2 = m2+j*velocity_change
        corrected_m3 = m3+velocity_change*(3*pressure+j*(corrected_velocity+velocity))
        fields = dict(answer.fields)
        fields['M0'] = tuple(corrected_n)
        fields['M2'] = tuple(corrected_m2)
        fields['M3'] = tuple(corrected_m3)
        return AnswerView(kind=answer.kind, fields=fields, axes=answer.axes,
                          points=answer.points, shape=answer.shape)


def project_prepared(prepared, background):
    """Retain the original representation if positivity cannot be certified."""
    background = float(background)
    if not np.isfinite(background) or background <= 0:
        return prepared
    if not np.isfinite(prepared.modes).all():
        return prepared
    if not _positive_density_certificate(prepared, background):
        return prepared
    if not _well_conditioned_projection(prepared, background):
        return prepared
    return ChargeProjected(prepared, background)
