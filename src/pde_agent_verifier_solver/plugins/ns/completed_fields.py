"""Complete velocity data entering the numerical NS coefficient projector.

This module only adapts arrays and serializes the existing continuous field.
It contains no alternate fitter, PCA, Flow, guidance or perturbation algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


def numerical_symbols():
    # The product ships the numerical implementation alongside its NS plugin.
    root = Path(__file__).resolve().parents[4] / "legacy/ns/src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from agent_verifier.neural_generator import structured_ns_answer_first as module
    return module


@dataclass
class CompletedGridGeometry:
    x: np.ndarray
    y: np.ndarray
    fluid_mask: np.ndarray
    continuous_geometry: dict

    @property
    def config(self):
        return SimpleNamespace(nx=len(self.x), ny=len(self.y))

    def to_json_dict(self):
        return {"continuous_geometry": self.continuous_geometry}


def completed_grid(x, y, geometry):
    """Map paired lattice points to the original projector's grid contract.

    The mask records available reconstructed fluid sites; missing lattice sites
    are excluded from the fit, never silently invented as zero measurements.
    """
    numerical_symbols()
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 4 or not np.isfinite([x, y]).all():
        raise ValueError("completed velocity needs finite paired spatial coordinates")
    ax, ix = np.unique(x, return_inverse=True)
    ay, iy = np.unique(y, return_inverse=True)
    if min(len(ax), len(ay)) < 2 or len(ax) * len(ay) > 16 * len(x):
        raise ValueError("completed points must lie on a Cartesian lattice; explicitly resample point clouds first")
    indices = ix * len(ay) + iy
    if len(np.unique(indices)) != len(indices):
        raise ValueError("completed spatial coordinates contain duplicates")
    bounds = (float(ax[0]), float(ax[-1]), float(ay[0]), float(ay[-1]))
    declared = geometry.get("domain_bounds", bounds)
    if not np.allclose(declared, bounds, rtol=0, atol=1e-12):
        raise ValueError("declared domain differs from the completed lattice bounds")
    spec = ContinuousGeometrySpec(
        domain_bounds=bounds,
        obstacle_polygons=tuple(tuple(tuple(float(v) for v in point) for point in polygon)
                                for polygon in geometry["obstacle_polygons"]),
        outer_boundary=geometry.get("outer_boundary", "no_slip"),
    )
    spec.validate()
    mask = np.zeros((len(ax), len(ay)), dtype=bool)
    mask[ix, iy] = True
    return CompletedGridGeometry(ax, ay, mask, spec.to_dict()), (ix, iy)


def project_completed_velocity(velocity, x, y, time, *, geometry, projection_config,
                               pressure=None, pressure_model="zero", device="cpu"):
    """Call the same numerical projector on reconstructed or simulated arrays."""
    module = numerical_symbols()
    x, y = np.asarray(x), np.asarray(y)
    keep = np.ones(len(x), dtype=bool)
    if "domain_bounds" in geometry:
        left, right, bottom, top = geometry["domain_bounds"]
        keep = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)
    velocity = np.asarray(velocity)[..., keep]
    if pressure is not None:
        pressure = np.asarray(pressure)[..., keep]
    grid, (ix, iy) = completed_grid(x[keep], y[keep], geometry)
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim == 3:
        values = values[None]
    times = np.asarray(time, dtype=np.float64)
    if (values.ndim != 4 or values.shape[1:] != (len(times), 2, len(ix))
            or not np.isfinite(values).all() or times.ndim != 1 or len(times) < 2
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)):
        raise ValueError("completed velocity must be finite [N,T,2,P] with increasing time")
    full = np.zeros((*values.shape[:3], len(grid.x), len(grid.y)), dtype=np.float64)
    full[..., ix, iy] = values
    if pressure is None:
        if pressure_model != "zero":
            raise ValueError("missing pressure requires explicit manufactured pressure_model='zero'")
        full_pressure = np.zeros((len(values), len(times), len(grid.x), len(grid.y)))
    else:
        p = np.asarray(pressure, dtype=np.float64)
        if p.ndim == 2:
            p = p[None]
        if p.shape != (len(values), len(times), len(ix)) or not np.isfinite(p).all():
            raise ValueError("completed pressure must be finite [N,T,P]")
        full_pressure = np.zeros((len(values), len(times), len(grid.x), len(grid.y)))
        full_pressure[..., ix, iy] = p
    config = module.StructuredNSProjectionConfig.from_dict(dict(projection_config))
    projector_type = module.StructuredNSCoefficientProjector
    if getattr(config, "fitting_method", "separable_ridge") == "joint_smoothing":
        from agent_verifier.neural_generator.structured_ns_spline_projection import JointSmoothNSCoefficientProjector
        projector_type = JointSmoothNSCoefficientProjector
    projector = projector_type(geometry=grid, t=times, config=config, device=device)
    coefficients, report = projector.project(velocity=full, pressure=full_pressure)
    report["input_contract"] = {
        "grid_shape": [len(grid.x), len(grid.y)], "fitted_fluid_points": len(ix),
        "unobserved_lattice_sites": int(grid.fluid_mask.size - grid.fluid_mask.sum()),
        "input_point_count": int(len(x)), "fitted_point_indices": np.flatnonzero(keep).tolist(),
        "pressure_model": pressure_model if pressure is None else "provided",
        "projection_config": config.to_dict(), "geometry": grid.continuous_geometry,
    }
    return coefficients.detach().cpu().numpy(), report, grid.continuous_geometry


def field_from_coefficients(coefficients, *, config, geometry, time_interval, viscosity, device="cpu"):
    module = numerical_symbols()
    if viscosity is None or not np.isfinite(viscosity) or float(viscosity) <= 0:
        raise ValueError("manufacturing an NS Case requires an explicitly declared positive viscosity")
    cfg = module.StructuredNSProjectionConfig.from_dict(dict(config))
    values = np.asarray(coefficients)
    expected = (2, cfg.time_coefficient_count, cfg.spatial_mode_x + 1, cfg.spatial_mode_y + 1)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("continuous coefficients differ from the numerical projection contract")
    return module.ContinuousStructuredNSMMS(
        module.StructuredNSCaseCoefficients(values[0].reshape(cfg.time_coefficient_count, -1),
                                           values[1].reshape(cfg.time_coefficient_count, -1)),
        config=cfg, geometry_spec=geometry, domain_bounds=tuple(geometry["domain_bounds"]),
        viscosity=float(viscosity), t0=float(time_interval[0]), t1=float(time_interval[1]), device=device,
    )


def field_to_state(field):
    """Explicit JSON state of the original numerical continuous field."""
    field = getattr(field, "raw_answer", field)
    module = numerical_symbols()
    if not isinstance(field, module.ContinuousStructuredNSMMS) or field.geometry_spec is None:
        raise ValueError("expected the numerical ContinuousStructuredNSMMS representation")
    return {"schema": "ns_structured_completed_field_v1", "config": field.config.to_dict(),
            "geometry": field.geometry_spec.to_dict(), "abi": field.continuous_mms_abi,
            "time_interval": [field.t0, field.t1], "viscosity": field.viscosity,
            "psi": field.coefficients.psi.detach().cpu().tolist(),
            "pressure": field.coefficients.pressure.detach().cpu().tolist()}


def field_from_state(payload, device="cpu"):
    module = numerical_symbols()
    if payload.get("schema") != "ns_structured_completed_field_v1":
        raise ValueError("expected a numerical structured field; reprepare old experimental-only fields")
    config = module.StructuredNSProjectionConfig.from_dict(payload["config"])
    values = np.asarray([payload["psi"], payload["pressure"]], dtype=np.float64)
    shape = (2, config.time_coefficient_count, config.spatial_coefficient_count)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("saved numerical field has invalid coefficient arrays")
    field = field_from_coefficients(values.reshape(2, config.time_coefficient_count,
        config.spatial_mode_x + 1, config.spatial_mode_y + 1), config=config.to_dict(),
        geometry=payload["geometry"], time_interval=payload["time_interval"],
        viscosity=payload["viscosity"], device=device)
    if field.continuous_mms_abi != payload["abi"]:
        raise ValueError("saved numerical field ABI differs")
    return field
