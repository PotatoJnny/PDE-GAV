"""Pilot-validated removal of redundant brackets around trusted source joins.

The caller supplies source-only interpolation and sampling routines. This
module never queries the public source, solution, or eventual scoring points.
Validation is an empirical forcing diagnostic, not a VPFP error bound.
"""
from __future__ import annotations

import numpy as np


def collapse_trusted_brackets(times, corner_indices, trusted_times):
    """Keep a sole trusted interior join in an otherwise narrow bracket.

    Original trusted time values must be saved before corner refinement.
    Groups with zero or multiple trusted joins, noninterior anchors, and
    isolated joins are preserved. In particular, a refined off-node bracket
    without an original trusted time inside it is preserved.
    """
    times = np.asarray(times, dtype=float)
    corners = np.unique(np.asarray(corner_indices, dtype=np.int64))
    trusted = np.unique(np.asarray(trusted_times, dtype=float))
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or corners.ndim != 1 or np.any(corners < 0)
            or np.any(corners >= len(times))
            or trusted.ndim != 1 or not np.isfinite(trusted).all()):
        raise ValueError('ordered finite pilot times and valid source joins required')
    corners = corners[(corners > 0) & (corners < len(times) - 1)]
    spacing = float(np.median(np.diff(times)))
    groups = []
    for index in corners:
        if groups and times[index] - times[groups[-1][-1]] <= spacing / 4:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    selected = []
    changed = []
    for group in groups:
        retained = trusted[(trusted >= times[group[0]]) &
                           (trusted <= times[group[-1]])]
        # Exact equality is intentional: refinement retains original nodes.
        anchor = np.flatnonzero(times[group] == retained[0]) if len(retained) == 1 else []
        if len(group) >= 3 and len(anchor) == 1 and 0 < anchor[0] < len(group) - 1:
            selected.append(group[int(anchor[0])])
            changed.append({'removed': len(group) - 1,
                            'left': float(times[group[0]]),
                            'anchor': float(retained[0]),
                            'right': float(times[group[-1]])})
        else:
            selected.extend(group)
    return np.asarray(selected, dtype=np.int64), {
        'original_join_count': len(corners),
        'proposed_join_count': len(selected),
        'collapsed_groups': changed,
        'pilot_median_spacing': spacing,
    }


def select_guarded_corner_order(times, values, theta, nu, target_count,
                                corner_indices, trusted_times,
                                sample_order, cubic_builder,
                                response_diagnostic, keep_going=None,
                                validation_counts=(129, 193, 257, 385, 513, 769)):
    """Return ``(acquisition_order, joins, diagnostic)`` with a strict fallback.

    ``sample_order`` has the incumbent corner sampler signature;
    ``cubic_builder(times, values, break_indices)`` returns an interpolant;
    ``response_diagnostic(times, residual, theta, nu)`` returns the four
    incumbent source/isolated-OU diagnostic scalars. All validation is made
    before acquiring any full source frames using the proposed schedule.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    original = np.unique(np.asarray(corner_indices, dtype=np.int64))
    proposed, diagnostics = collapse_trusted_brackets(times, original, trusted_times)
    count = min(int(target_count), len(times))
    baseline = sample_order(times, values, theta, nu, count, corners=original)
    diagnostics.update(enabled=False, checks=[], empirical_gate=True)

    def reject(reason):
        diagnostics['reason'] = reason
        return baseline, original, diagnostics

    if np.array_equal(original, proposed):
        return reject('no unambiguous trusted bracket groups')
    if count < 2:
        return reject('fewer than two source frames')
    if keep_going is not None and not keep_going():
        return reject('guard deadline before alternative sampling')
    alternative = sample_order(times, values, theta, nu, count, corners=proposed)
    checkpoints = np.unique(np.minimum(np.asarray(validation_counts, dtype=int), count))
    checkpoints = checkpoints[checkpoints >= 2]
    # Source diagnostics should vanish continuously when the source is zero.
    # The absolute terms below only absorb arithmetic-level residuals.
    source_floor = 256 * np.finfo(float).eps * max(1., float(np.max(abs(values))))
    integrated_floor = source_floor * (times[-1] - times[0])
    limits = {'response_rms': 1., 'density_primitive_rms': 1.05,
              'density_final_rms': 1.3, 'max_defect': 1.25}
    meaningful_gain = False
    for checkpoint in checkpoints:
        if keep_going is not None and not keep_going():
            return reject('guard deadline during prefix validation')
        row = {'count': int(checkpoint)}
        try:
            for label, order, joins in [('baseline', baseline, original),
                                        ('proposed', alternative, proposed)]:
                selected = np.sort(order[:checkpoint])
                breaks = np.flatnonzero(np.isin(selected, joins))
                spline = cubic_builder(times[selected], values[selected], breaks)
                row[label] = response_diagnostic(times, spline(times) - values, theta, nu)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            diagnostics['checks'].append(row)
            return reject('prefix interpolation failure: ' + str(error))
        row['ratios'] = {}
        passed = True
        for key, limit in limits.items():
            a = float(row['baseline'][key])
            b = float(row['proposed'][key])
            floor = source_floor if key == 'max_defect' else integrated_floor
            row['ratios'][key] = b / max(a, floor)
            if not np.isfinite(a + b) or b > limit * a + floor:
                passed = False
        row['passed'] = passed
        diagnostics['checks'].append(row)
        if not passed:
            return reject('a source/primitive prefix gate rejected the proposal')
        meaningful_gain |= (row['baseline']['response_rms'] > integrated_floor and
                            row['proposed']['response_rms'] <= .95 * row['baseline']['response_rms'])
    if not meaningful_gain:
        return reject('no prefix has a five-percent response improvement')
    diagnostics['enabled'] = True
    diagnostics['reason'] = 'trusted bracket collapse passed all source/primitive prefix gates'
    return alternative, proposed, diagnostics
