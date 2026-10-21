"""Gate the original 30-to-60 Gauss12 pilot at39 and54 cells."""
from __future__ import annotations

import numpy as np

MAX_CELLS = 60
EARLY_CELLS = (39, 54)
ORDER = 12
MOMENT_TOLERANCE = 1e-6
SHAPE_TOLERANCE = 1e-3
REFINEMENT_FLOOR = 1e-12
FLOORS = np.array([1.0041253663782175, .4788345668393978,
                   2.930763214320111, 1.9801548942236534])


def refine_velocity_grid(case, query, began, allowance, select_grid,
                         reserve_error, clock, pilot_budget=45.):
    """Rejection preserves the existing 30,33,...,60 refinement path exactly."""
    fallback, guided = select_grid(case, MAX_CELLS)
    start, _ = select_grid(case, MAX_CELLS//2)
    domain = case.public_data['equation']['domain']
    xmin, xmax = map(float, domain['x_interval'])
    tmin, tmax = map(float, domain['time_interval'])
    horizon = tmax-tmin
    xs = xmin+(xmax-xmin)*(np.arange(32)+.173)/32
    sample_times = (None, *np.linspace(tmin, tmax, 17))
    pilot_started = clock()
    pilot_limit = min(pilot_budget, max(0., .1*allowance))
    count, trace = 0, []
    gate = dict(attempted=False, accepted=False, stages=[])
    low_nodes, _ = np.polynomial.legendre.leggauss(ORDER)
    high_nodes, _ = np.polynomial.legendre.leggauss(2*ORDER)
    interpolation = (np.polynomial.legendre.legvander(high_nodes, ORDER-1)
                     @np.linalg.inv(np.polynomial.legendre.legvander(low_nodes, ORDER-1)))

    def integrals(intervals, x=xs, times=sample_times):
        nonlocal count
        geometry = []
        for order in (ORDER, 2*ORDER):
            xi, w = np.polynomial.legendre.leggauss(order)
            half = (intervals[:, 1]-intervals[:, 0])/2
            velocity = intervals.mean(axis=1)[:, None]+half[:, None]*xi
            weight = half[:, None]*w
            geometry.append((velocity, weight))
        velocities = np.concatenate([item[0].ravel() for item in geometry])
        xx, vv = np.repeat(x, len(velocities)), np.tile(velocities, len(x))
        values = []
        for time in times:
            if clock()-pilot_started >= pilot_limit:
                raise reserve_error('Velocity pilot reached its budget')
            points = (np.column_stack((xx, vv)) if time is None else
                      np.column_stack((np.full(len(xx), time), xx, vv)))
            reserve = max(100., allowance-(pilot_started-began)-pilot_limit)
            value = query(points, initial=time is None, reserve=reserve).reshape(len(x), -1)
            count += len(points)
            values.append(value if time is None else horizon*value)
        values = np.concatenate(values, axis=0)
        moments, parts = [], []
        absolute, offset = None, 0
        for velocity, weight in geometry:
            size = velocity.size
            part = values[:, offset:offset+size].reshape(len(values), *velocity.shape)
            parts.append(part)
            weighted = weight[None, :, :]*velocity[None, :, :]**np.arange(4)[:, None, None]
            moments.append(np.einsum('scn,pcn->scp', part, weighted))
            absolute = np.einsum('scn,pcn->scp', np.abs(part), np.abs(weighted))
            offset += size
        shape_defect = np.sum(np.abs(parts[0]@interpolation.T-parts[1])
                              *geometry[1][1][None, :, :], axis=2)
        if not all(np.isfinite(a).all() for a in (*moments, absolute, shape_defect)):
            raise reserve_error('Nonfinite velocity pilot')
        return moments[0], moments[1], absolute, shape_defect

    def defects(low, high, absolute, shape):
        scales = np.maximum(np.max(np.sum(absolute, axis=1), axis=0), FLOORS)
        local = np.max(np.abs(high-low)/scales[None, None, :], axis=(0, 2))
        moment_error = float(np.sum(local))
        shape_error = float(np.max(np.sum(shape, axis=1)
                                  /np.maximum(np.sum(absolute[:, :, 0], axis=1), 1e-14)))
        return moment_error, shape_error

    def passed(error):
        return bool(np.isfinite(error).all() and error[0] <= MOMENT_TOLERANCE
                    and error[1] <= SHAPE_TOLERANCE)

    intervals = np.column_stack((start[:-1], start[1:]))
    try:
        low, high, absolute, shape = integrals(intervals)
        while len(intervals) < MAX_CELLS:
            if len(intervals) in EARLY_CELLS:
                stage = dict(cells=len(intervals), accepted=False)
                gate['stages'].append(stage)
                gate['attempted'] = True
                primary = defects(low, high, absolute, shape)
                stage.update(primary_moment=primary[0], primary_shape=primary[1])
                if passed(primary):
                    # Coprime shifted x count and nonmatching times test profiles
                    # that did not choose this grid. Query only a legal prefix.
                    check_x = xmin+(xmax-xmin)*(np.arange(17)+.619)/17
                    check_times = (None, *(tmin+horizon*np.array([.071, .283, .517, .793, .947])))
                    check = defects(*integrals(intervals, check_x, check_times))
                    stage.update(validation_moment=check[0], validation_shape=check[1])
                    if passed(check):
                        stage['accepted'] = True
                        gate.update(accepted=True, accepted_cells=len(intervals))
                        break
            scales = np.maximum(np.max(np.sum(absolute, axis=1), axis=0), FLOORS)
            errors = np.max(np.abs(high-low)/scales[None, None, :], axis=(0, 2))
            add = min(max(1, MAX_CELLS//16), MAX_CELLS-len(intervals))
            priority = errors if np.max(errors) >= REFINEMENT_FLOOR else np.diff(intervals, axis=1).ravel()
            selected = np.argpartition(priority, -add)[-add:]
            trace.append((len(intervals), float(np.max(errors)), float(np.sum(errors))))
            retained = np.ones(len(intervals), dtype=bool)
            retained[selected] = False
            chosen = intervals[selected]
            middle = chosen.mean(axis=1)
            children = np.concatenate((np.column_stack((chosen[:, 0], middle)),
                                       np.column_stack((middle, chosen[:, 1]))), axis=0)
            intervals = np.concatenate((intervals[retained], children), axis=0)
            if len(intervals) == MAX_CELLS:
                break
            child_low, child_high, child_absolute, child_shape = integrals(children)
            low = np.concatenate((low[:, retained], child_low), axis=1)
            high = np.concatenate((high[:, retained], child_high), axis=1)
            absolute = np.concatenate((absolute[:, retained], child_absolute), axis=1)
            shape = np.concatenate((shape[:, retained], child_shape), axis=1)
    except reserve_error:
        return fallback, guided, dict(completed=False, points=count, trace=trace,
                                      early_gate=gate, selected_cells=MAX_CELLS)
    edges = np.sort(np.r_[intervals[:, 0], np.max(intervals[:, 1])])
    return edges, guided, dict(completed=True, points=count, trace=trace,
                              early_gate=gate, selected_cells=len(intervals))
