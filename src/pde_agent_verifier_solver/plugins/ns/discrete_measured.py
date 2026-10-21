"""Package measured grids using the existing numerical NS finite-array format."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def write_discrete_measured_case(directory, *, case_id, velocity, point_x, point_y,
                                times, viscosity, provenance):
    """Keep measured values; use numerical D0's finite-difference MMS operator.

    Only the initial slice and perimeter velocities are public. The full velocity
    remains in NSFieldTruth's private file. Pressure zero is a manufactured
    convention, not an inferred experimental pressure. No continuous projection,
    smoothing, or Generator is used here.
    """
    from agent_verifier.neural_generator.triplane_dataset import (
        divergence, mms_forcing_from_velocity_pressure,
    )

    directory = Path(directory)
    px, py = np.asarray(point_x), np.asarray(point_y)
    t = np.asarray(times, dtype=np.float64)
    values = np.asarray(velocity)
    x, y = np.unique(px), np.unique(py)
    if (px.ndim != 1 or py.shape != px.shape or len(x) * len(y) != len(px)
            or len(x) < 3 or len(y) < 3 or t.ndim != 1 or len(t) < 3
            or not np.all(np.diff(t) > 0) or values.shape != (len(t), 2, len(px))
            or not all(np.isfinite(a).all() for a in (px, py, t, values))
            or not np.isfinite(viscosity) or viscosity <= 0):
        raise ValueError('expected finite complete Cartesian velocity samples and ordered time')
    ix, iy = np.searchsorted(x, px), np.searchsorted(y, py)
    if len(np.unique(ix * len(y) + iy)) != len(px):
        raise ValueError('duplicate Cartesian sample coordinates')
    # Index by coordinates, never assume point-cloud flattening order.
    vel = np.empty((len(t), 2, len(x), len(y)), dtype=values.dtype)
    vel[:, :, ix, iy] = values
    np.testing.assert_array_equal(vel[:, :, ix, iy], values)
    fluid = np.ones((len(x), len(y)), dtype=bool)
    outer = np.zeros_like(fluid)
    outer[[0, -1], :] = True
    outer[:, [0, -1]] = True
    geometry = SimpleNamespace(x=x, y=y, fluid_mask=fluid, outer_boundary_mask=outer)
    pressure = np.zeros((len(t), len(x), len(y)), dtype=vel.dtype)
    forcing = mms_forcing_from_velocity_pressure(
        velocity=vel, pressure=pressure, t=t, viscosity=viscosity, geometry=geometry,
    )
    indices = np.argwhere(outer)
    public = dict(x=x, y=y, t=t, fluid_mask=fluid, obstacle_mask=~fluid,
                  outer_boundary_mask=outer, initial_velocity=vel[0], forcing=forcing,
                  boundary_indices=indices,
                  boundary_points=np.column_stack((x[indices[:, 0]], y[indices[:, 1]])),
                  boundary_velocity=vel[:, :, outer])
    bounds = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    question = dict(
        case_id=case_id,
        task='Solve 2D incompressible NS from finite initial, forcing and boundary arrays.',
        case_source='measured_trajectory_derived_discrete_mms',
        forcing_derivation='numerical_D0_finite_difference_MMS_from_measured_velocity_and_zero_pressure',
        viscosity=float(viscosity), pressure_model='manufactured_zero_unscored',
        geometry=dict(domain=bounds, grid_sample='measured_grid_nodes',
                      outer_boundary='prescribed_velocity_arrays', obstacle_polygons=[]),
        time=dict(t0=float(t[0]), t1=float(t[-1]), nt=len(t),
                  dt_median=float(np.median(np.diff(t))), values=t.tolist()),
        arrays={k: 'public_arrays.npz:' + k for k in public}, target_shape=list(vel.shape),
        boundary=dict(kind='dirichlet_velocity', indices='boundary_indices[Nb,2]',
                      points='boundary_points[Nb,2]', values='boundary_velocity[T,2,Nb]',
                      note='Only the perimeter is supplied. Forcing on boundary nodes is unused and zero, as in numerical D0.'),
        solver_contract='prepare_solution(case, solve_budget) then evaluate(AnswerQuery POINTS)',
    )
    div = divergence(vel, x, y)
    rms = float(np.sqrt(np.mean(np.square(vel.astype(np.float64)))))
    diagnostic = dict(
        reference='unaltered discrete measured velocity; regular-grid interpolation for scoring',
        continuous_projection=False, pressure_model='manufactured_zero_unscored',
        velocity_rms=rms, divergence_rms=float(np.sqrt(np.mean(div ** 2))),
        note='Measured divergence/noise is retained, not projected away. Discrete MMS matches the momentum operator; it does not enforce zero divergence.',
    )
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'public_arrays.npz', **public)
    np.savez_compressed(directory / 'hidden_truth_cache.npz', u_true=vel, pressure=pressure, t=t)
    for name, data in (
        ('public_question.json', question),
        ('case_spec.json', dict(case_id=case_id, family='measured', parameters={})),
        ('source_provenance.json', provenance), ('diagnostics.json', diagnostic),
    ):
        (directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return diagnostic
