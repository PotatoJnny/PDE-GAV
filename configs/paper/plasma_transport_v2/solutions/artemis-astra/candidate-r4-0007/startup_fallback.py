"""Emergency initial-state approximation after exact startup sampling expires.

Public f0 arrays are non-authoritative convenience data. They are used only
after a sampling reserve exception, never on a successful exact-query path.
"""
import numpy as np


def prepare(case, prepared_type, xmin, xmax, tmin, tmax, nx):
    length = xmax-xmin
    modes = np.zeros((5, nx), dtype=np.complex128)
    kind, detail_retained = 'positive_background_equilibrium', False
    background = case.public_data['equation'].get('poisson', {}).get('neutralizing_background', 1.)
    try:
        background = float(background)
    except (TypeError, ValueError, OverflowError):
        background = 1.
    if not np.isfinite(background) or background <= 0.:
        background = 1.
    modes[0, 0] = modes[2, 0] = background
    try:
        arrays = case.public_data.get('arrays', {})
        x = np.asarray(arrays.get('x', ()), dtype=float)
        v = np.asarray(arrays.get('v', ()), dtype=float)
        f0 = np.asarray(arrays.get('f0', ()), dtype=float)
        valid = (x.ndim == v.ndim == 1 and len(x) >= 2 and len(v) >= 2
                 and f0.shape == (len(x), len(v)) and np.isfinite(x).all()
                 and np.isfinite(v).all() and np.isfinite(f0).all()
                 and np.all(f0 >= 0.) and np.all(np.diff(v) > 0.))
        if valid:
            position = (x-xmin) % length
            order = np.argsort(position)
            position, f0 = position[order], f0[order]
            valid = bool(np.all(np.diff(position) > 0.))
        if valid:
            with np.errstate(over='ignore', invalid='ignore'):
                moments = np.stack([np.trapz(f0*v[None, :]**p, v, axis=1) for p in range(4)])
            valid = bool(np.isfinite(moments).all() and np.all(moments[0] > 0.))
        if valid:
            grid = length*np.arange(nx)/nx
            uniform = np.stack([np.interp(grid, position, row, period=length) for row in moments])
            candidate = np.zeros_like(modes)
            candidate[:4] = np.fft.fft(uniform, axis=1)/nx
            candidate[:, nx//2] = 0.
            # A sufficient density bound avoids zeros between interpolation
            # nodes. If it fails, retain the finite spatially averaged moments.
            mean = candidate[0, 0].real
            detail_retained = bool(np.sum(np.abs(candidate[0, 1:])) < .95*mean)
            if not detail_retained:
                candidate[:, 1:] = 0.
            kx = 2*np.pi*np.fft.fftfreq(nx, length/nx)
            nonzero = kx != 0.
            candidate[4, nonzero] = candidate[0, nonzero]/(1j*kx[nonzero])
            if np.isfinite(candidate).all():
                modes = candidate
                kind = 'non_authoritative_public_f0_moments'
    except (TypeError, ValueError, OverflowError):
        pass
    prepared = prepared_type([tmin, tmax], np.stack((modes, modes)), xmin, length)
    prepared.effective_dt = tmax-tmin
    prepared.integration_steps = 0
    prepared.effective_source_nodes = 0
    prepared.initial_fallback_info = dict(reason='initial_sampling_reserve', method=kind,
                                         approximate=True, spatial_detail_retained=detail_retained)
    prepared.source_refinement_info = dict(attempted=False, completed=False,
                                          reason='initial_sampling_reserve')
    prepared.spatial_resolution_info = dict(initial_spatial_nodes=2048, state_spatial_nodes=nx,
                                            method='startup_budget_fallback')
    return prepared
