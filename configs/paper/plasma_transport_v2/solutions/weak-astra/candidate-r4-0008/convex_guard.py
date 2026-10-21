"""Maximal admissible global raw-moment correction per prepared snapshot.

The physical inequalities match the incumbent guard. The feasible segment is
searched between its known valid anchor and first invalid endpoint, preserving
one scalar coefficient for every moment and spatial Fourier mode. No Oracle
calls or scoring-query access occurs here.
"""
from __future__ import annotations

import numpy as np
from moments import evaluate_grid
from realizability import gram


def guard_corrections(prior_modes, correction_modes, check, *, iterations=14,
                      safety_fraction=0.999):
    if prior_modes.shape != correction_modes.shape or prior_modes.ndim != 4:
        raise ValueError('matching (time,moment,x,y) mode arrays required')
    if iterations < 1 or not 0 < safety_fraction <= 1:
        raise ValueError('positive iterations and safety fraction in (0,1] required')
    modes = prior_modes + correction_modes
    weights = []
    n = modes.shape[-1]
    for it in range(len(modes)):
        check()
        base = evaluate_grid(prior_modes[it], 2*n)
        delta = evaluate_grid(correction_modes[it], 2*n)
        anchor_psd = np.min(np.linalg.eigvalsh(gram(base, True))) >= -1e-10
        if anchor_psd:
            gram_base = gram(base)
            gram_delta = gram(delta)
            diagonal_ids = np.arange(gram_base.shape[-1])

        def valid(amount):
            trial = base + amount*delta
            r = trial[0]
            safe = np.maximum(r, 1e-12)
            pxx = trial[3] - trial[1]**2/safe
            pxy = trial[4] - trial[1]*trial[2]/safe
            pyy = trial[5] - trial[2]**2/safe
            acceptable = (np.isfinite(trial).all() and np.min(r) > .15
                          and np.min(pxx) > .01 and np.min(pyy) > .01
                          and np.min(pxx*pyy-pxy*pxy) > 1e-5)
            if acceptable and anchor_psd:
                # G_normalized >= -tol*I iff G+tol*D^2 is PSD, where D
                # is precisely the diagonal scale used by gram(..., True).
                # Cholesky gives the same tolerance test up to boundary
                # roundoff, with substantially less work than eigvalsh.
                g = gram_base + amount*gram_delta
                diagonal = np.diagonal(g, axis1=-2, axis2=-1)
                floor = 1e-14*np.maximum(np.abs(r), 1.)[..., None]
                shift = 1e-10*np.maximum(np.abs(diagonal), floor)
                g[..., diagonal_ids, diagonal_ids] += shift
                try:
                    np.linalg.cholesky(g)
                except np.linalg.LinAlgError:
                    acceptable = False
            return acceptable

        if valid(1.):
            amount = 1.
        elif not valid(0.):
            # Preserve the incumbent fallback behavior when its prior itself
            # does not meet the non-Gram thresholds. No convex guarantee then.
            amount = 0.
            trial_amount = .5
            for _ in range(17):
                check()
                if valid(trial_amount):
                    amount = trial_amount
                    break
                trial_amount *= .5
        else:
            low, high = 0., 1.
            for _ in range(iterations):
                check()
                midpoint = .5*(low+high)
                if valid(midpoint):
                    low = midpoint
                else:
                    high = midpoint
            # A small margin avoids returning a state sitting exactly on the
            # sampled PSD boundary; it does not certify all off-grid sites.
            amount = safety_fraction*low
        weights.append(float(amount))
        modes[it] = prior_modes[it] + amount*correction_modes[it]
    return modes, weights
