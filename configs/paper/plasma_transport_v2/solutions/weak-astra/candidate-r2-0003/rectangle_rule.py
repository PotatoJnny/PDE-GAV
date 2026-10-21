"""Choose a deterministic source-moment quadrature from public pilot values.

The rule remains an ordinary uniform velocity quadrature inside a rectangle.
It does not fit source-specific weights. Diagnostics distinguish an accepted
pilot tolerance from the best feasible rule under a node cap.
"""
from __future__ import annotations

import numpy as np
from math import comb


def polynomial_powers(degree):
    return [(a, n-a) for n in range(degree+1) for a in range(n, -1, -1)]


def _polynomials(v, degree):
    x, y = np.meshgrid(v, v, indexing='ij')
    return np.asarray([x**a*y**b for a, b in polynomial_powers(degree)])


def adaptive_rectangle_rule(pilot, v, degree=4, max_nodes=1600,
                            tolerance=.003, stride=2, min_bound=8.,
                            train_rows=None, validation_rows=None,
                            group_norm=False, padding_steps=0):
    """Fit a symmetric, axis-specific rectangle using pilot moment errors.

    ``pilot`` has shape (nrow,nv,nv), ``v`` is a uniform, centered velocity
    axis. ``degree`` includes every monomial of total degree up to the limit.
    If train_rows is omitted all pilot rows select the rule. A caller may
    supply independent validation_rows, which never alter the selected rule.
    Return a serializable dictionary plus NumPy arrays for direct source use.
    The returned velocities are RELATIVE; add the public bulk center before
    sending them to the source Oracle. ``indices`` refer to the fine flattened
    pilot grid; returned weights already contain the velocity cell area.
    """
    pilot = np.asarray(pilot, dtype=float)
    v = np.asarray(v, dtype=float)
    if (pilot.ndim != 3 or pilot.shape[1:] != (len(v), len(v))
            or not np.isfinite(pilot).all() or len(v) < 4):
        raise ValueError('finite pilot rows on the square velocity grid required')
    dv = float(v[1]-v[0])
    if dv <= 0 or not np.allclose(np.diff(v), dv, rtol=1e-10, atol=1e-12):
        raise ValueError('uniform increasing velocity axis required')
    if degree < 0 or stride < 1 or max_nodes < 1 or tolerance <= 0:
        raise ValueError('invalid rectangle quadrature configuration')
    train = np.arange(len(pilot)) if train_rows is None else np.asarray(train_rows, dtype=int)
    validation = np.asarray([], dtype=int) if validation_rows is None else np.asarray(validation_rows, dtype=int)
    if not len(train):
        raise ValueError('at least one training row required')
    polys = _polynomials(v, degree)
    # The reference integrates the fine pilot mesh; it is a numerical reference,
    # not exact truth. The source pilot itself needs resolution/tail checks.
    reference = np.einsum('nxy,mxy->nm', pilot, polys)*dv**2
    rms = np.sqrt(np.mean(reference[train]**2, axis=0))
    denominator = np.maximum(rms, np.maximum(rms.max()*1e-12, 1e-30))
    powers = polynomial_powers(degree)

    def grouped_errors(approx, rows):
        squared_error = np.mean((approx[rows]-reference[rows])**2, axis=0)
        squared_reference = np.mean(reference[rows]**2, axis=0)
        errors = []
        for d in range(degree+1):
            indices = [j for j, (a, b) in enumerate(powers) if a+b == d]
            # Full symmetric moment-tensor Frobenius norm: off-diagonal
            # permutations count separately, making the metric rotationally
            # invariant within a homogeneous degree.
            multiplicity = np.asarray([comb(d, powers[j][0]) for j in indices])
            errors.append(float(np.sqrt(np.sum(multiplicity*squared_error[indices])
                          /max(np.sum(multiplicity*squared_reference[indices]), 1e-60))))
        return errors
    coarse = (pilot[:, None, ::stride, ::stride]
              *polys[None, :, ::stride, ::stride]*(dv*stride)**2)
    prefix = np.pad(coarse.cumsum(axis=-1).cumsum(axis=-2),
                    ((0, 0), (0, 0), (1, 0), (1, 0)))
    vc = v[::stride]
    bounds = np.unique(np.abs(vc)[np.abs(vc) >= min_bound-1e-10])
    records = []
    for bx in bounds:
        ix = np.flatnonzero(np.abs(vc) <= bx+1e-10)
        if not len(ix):
            continue
        for by in bounds:
            iy = np.flatnonzero(np.abs(vc) <= by+1e-10)
            count = len(ix)*len(iy)
            if not len(iy) or count > max_nodes:
                continue
            a, b, c, d = int(ix[0]), int(ix[-1]+1), int(iy[0]), int(iy[-1]+1)
            approx = prefix[:, :, b, d]-prefix[:, :, a, d]-prefix[:, :, b, c]+prefix[:, :, a, c]
            errors = np.sqrt(np.mean((approx[train]-reference[train])**2, axis=0))/denominator
            objective = max(grouped_errors(approx, train)) if group_norm else float(errors.max())
            records.append((objective, int(count), float(bx), float(by), approx, errors))
    if not records:
        raise ValueError('node cap cannot fit the minimum rectangle')
    accepted = [record for record in records if record[0] <= tolerance]
    if accepted:
        chosen = min(accepted, key=lambda row: (row[1], row[0], row[2], row[3]))
    else:
        chosen = min(records, key=lambda row: (row[0], row[1], row[2], row[3]))
    error, count, bx, by, approx, train_errors = chosen
    unexpanded_bounds = [bx, by]
    unexpanded_count = count
    unexpanded_error = error
    if padding_steps:
        bx = min(float(np.max(np.abs(vc))), bx+padding_steps*dv*stride)
        by = min(float(np.max(np.abs(vc))), by+padding_steps*dv*stride)
        ix = np.flatnonzero(np.abs(vc) <= bx+1e-10)
        iy = np.flatnonzero(np.abs(vc) <= by+1e-10)
        a, b, c, d = int(ix[0]), int(ix[-1]+1), int(iy[0]), int(iy[-1]+1)
        approx = prefix[:, :, b, d]-prefix[:, :, a, d]-prefix[:, :, b, c]+prefix[:, :, a, c]
        count = len(ix)*len(iy)
        train_errors = np.sqrt(np.mean((approx[train]-reference[train])**2, axis=0))/denominator
        error = max(grouped_errors(approx, train)) if group_norm else float(train_errors.max())
    fine_grid = np.arange(len(v)**2).reshape(len(v), len(v))
    coarse_indices = fine_grid[::stride, ::stride]
    x, y = np.meshgrid(vc, vc, indexing='ij')
    mask = (np.abs(x) <= bx+1e-10)&(np.abs(y) <= by+1e-10)
    indices = coarse_indices[mask]
    velocities = np.column_stack([x[mask], y[mask]])
    diagnostics = dict(accepted=bool(error <= tolerance), tolerance=float(tolerance),
                       selection_error=error, group_norm=bool(group_norm),
                       unexpanded_selection_error=unexpanded_error,
                       unexpanded_bounds=unexpanded_bounds,
                       unexpanded_node_count=int(unexpanded_count),
                       padding_steps=int(padding_steps),
                       max_train_relative_rms=float(train_errors.max()),
                       train_relative_rms=train_errors.tolist(),
                       train_degree_relative_rms=grouped_errors(approx, train),
                       bounds=[bx, by], node_count=count, degree=int(degree),
                       stride=int(stride), fine_node_count=int(len(v)**2),
                       train_row_count=int(len(train)),
                       validation_row_count=int(len(validation)))
    if len(validation):
        denom = np.maximum(np.sqrt(np.mean(reference[validation]**2, axis=0)),
                           np.maximum(rms.max()*1e-12, 1e-30))
        errors = np.sqrt(np.mean((approx[validation]-reference[validation])**2, axis=0))/denom
        diagnostics['validation_relative_rms'] = errors.tolist()
        diagnostics['max_validation_relative_rms'] = float(errors.max())
        group_errors = grouped_errors(approx, validation)
        diagnostics['validation_degree_relative_rms'] = group_errors
        diagnostics['validation_passed'] = bool((max(group_errors) if group_norm else errors.max()) <= tolerance)
    return dict(indices=indices, velocities=velocities,
                weights=np.full(len(indices), (dv*stride)**2),
                powers=polynomial_powers(degree), diagnostics=diagnostics)


def rectangle_rule(pilot, v, degree=4):
    """Source-sampler default: 0.5% per-degree tensor-RMS pilot tolerance.

    Permit the entire stride-two pilot box when truncation cannot meet that
    tolerance. A quadrature aliasing failure is reported, never silently marked
    accepted. Current source pilots still need independent time/space checks.
    """
    pilot = np.asarray(pilot, dtype=float)
    v = np.asarray(v, dtype=float)
    coarse = adaptive_rectangle_rule(pilot, v, degree=degree,
                                     max_nodes=len(v[::2])**2,
                                     tolerance=.005, group_norm=True,
                                     padding_steps=1)
    # Whole-box comparison isolates velocity-spacing effects from the chosen
    # rectangle's tail truncation. Both use the same finite public pilot box;
    # this is mesh evidence, not a certification of the fine grid or its tails.
    poly = _polynomials(v, degree)
    dv = float(v[1]-v[0])
    fine_moments = np.einsum('nxy,mxy->nm', pilot, poly)*dv**2
    coarse_moments = np.einsum('nxy,mxy->nm', pilot[:, ::2, ::2],
                               poly[:, ::2, ::2])*(2*dv)**2
    square_error = np.mean((coarse_moments-fine_moments)**2, axis=0)
    square_reference = np.mean(fine_moments**2, axis=0)
    powers = polynomial_powers(degree)
    spacing_errors = []
    for d in range(degree+1):
        inds = [i for i, (a, b) in enumerate(powers) if a+b == d]
        multiplicity = np.asarray([comb(d, powers[i][0]) for i in inds])
        spacing_errors.append(float(np.sqrt(np.sum(multiplicity*square_error[inds])
                              /max(np.sum(multiplicity*square_reference[inds]), 1e-60))))
    if coarse['diagnostics']['accepted']:
        result = coarse
        result['diagnostics']['fine_grid_fallback'] = False
    else:
        # Preserve the physical 2/3 margin when changing to fine spacing 1/3.
        # The node count may grow substantially. The caller must choose the
        # source time/space mesh using this final count and its budget meter.
        result = adaptive_rectangle_rule(pilot, v, degree=degree,
                                          max_nodes=len(v)**2, stride=1,
                                          tolerance=.005, group_norm=True,
                                          padding_steps=2)
        result['diagnostics']['fine_grid_fallback'] = True
        result['diagnostics']['rejected_coarse_rule'] = coarse['diagnostics']
    result['diagnostics']['coarse_fullbox_degree_relative_rms'] = spacing_errors
    result['diagnostics']['coarse_fullbox_passed'] = bool(max(spacing_errors) <= .005)
    result['diagnostics']['fine_grid_resolution_unverified'] = True
    result['diagnostics']['outside_pilot_box_tail_unverified'] = True
    return result
