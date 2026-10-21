from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.neural_generator.continuous_geometry import (
    ContinuousGeometrySpec,
    _prepare_continuous_polygons,
    continuous_geometry_spec_from_contract,
    continuous_no_slip_masks,
    interior_c3_no_slip_masks,
    legacy_c3_continuous_no_slip_masks,
)
from agent_verifier.neural_generator.triplane_decoder import (
    geometry_features_from_arrays,
)
from agent_verifier.neural_generator.triplane_mms import (
    REFERENCE_AUTOGRAD_BACKEND_ID,
    TriplaneDecodedFields,
    continuous_divergence_relative_max,
    continuous_mms_provenance,
    navier_stokes_quantities_from_psi_pressure,
)


LEGACY_STRUCTURED_NS_MMS_ABI_VERSION = "continuous_structured_ns_mms_v1"
LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION = "continuous_structured_ns_mms_v2"
LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION = "continuous_structured_ns_mms_v3"
LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION = (
    "continuous_structured_ns_mms_v4"
)
# Archived defaults remain v5. A v6 field/artifact must opt in explicitly.
STRUCTURED_NS_MMS_ABI_VERSION = "continuous_structured_ns_mms_v5"
INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION = "continuous_structured_ns_mms_v6"
SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS = {
    LEGACY_STRUCTURED_NS_MMS_ABI_VERSION,
    LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
    LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    STRUCTURED_NS_MMS_ABI_VERSION,
    INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
}
STRUCTURED_NS_GENERATOR_SCHEMA = "ns2d_coarse_seeded_structured_answer_first_v9"


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for structured NS answer-first fields") from exc
    return torch


@dataclass(frozen=True)
class StructuredNSProjectionConfig:
    spatial_mode_x: int = 8
    spatial_mode_y: int = 8
    time_order: int = 8
    geometry_mask_transition: float = 0.05
    ridge_relative: float = 1.0e-6
    evaluation_batch_points: int = 4096
    # This extends the cosine expansion domain, not the physical query domain.
    spatial_extension: float = 0.0
    spatial_basis: str = "cosine"
    temporal_basis: str = "chebyshev"
    spatial_degree: int = 5
    temporal_degree: int = 3
    spatial_knots_x: tuple[float, ...] = ()
    spatial_knots_y: tuple[float, ...] = ()
    temporal_knots: tuple[float, ...] = ()
    fitting_method: str = "separable_ridge"
    smoothing_spatial: float = 0.0
    smoothing_temporal: float = 0.0

    def __post_init__(self):
        from .structured_ns_basis import open_uniform_knots
        for axis, kind, count, degree in (
            ("x", self.spatial_basis, int(self.spatial_mode_x)+1, self.spatial_degree),
            ("y", self.spatial_basis, int(self.spatial_mode_y)+1, self.spatial_degree),
            ("t", self.temporal_basis, int(self.time_order)+1, self.temporal_degree),
        ):
            name = "temporal_knots" if axis == "t" else f"spatial_knots_{axis}"
            knots = tuple(float(x) for x in getattr(self, name))
            if kind == "bspline" and not knots:
                knots = open_uniform_knots(count, degree)
            object.__setattr__(self, name, knots)

    @property
    def is_legacy_basis(self) -> bool:
        return self.spatial_basis == "cosine" and self.temporal_basis == "chebyshev"

    def validate(self) -> None:
        if (isinstance(self.spatial_extension, bool)
                or not np.isfinite(float(self.spatial_extension)) or float(self.spatial_extension) < 0.0):
            raise ValueError("spatial_extension must be finite and nonnegative")
        if min(
            int(self.spatial_mode_x),
            int(self.spatial_mode_y),
            int(self.time_order),
            int(self.evaluation_batch_points),
        ) < 1:
            raise ValueError("structured NS projection dimensions must be positive")
        if float(self.geometry_mask_transition) <= 0.0:
            raise ValueError("geometry mask transition must be positive")
        if float(self.ridge_relative) < 0.0:
            raise ValueError("ridge regularization must be non-negative")
        from .structured_ns_basis import validate_bspline_spec
        if self.spatial_basis not in ("cosine", "bspline") or self.temporal_basis not in ("chebyshev", "bspline"):
            raise ValueError("unsupported spatial or temporal NS basis")
        if self.spatial_basis == "bspline":
            if self.spatial_degree != 5 or isinstance(self.spatial_degree, bool):
                raise ValueError("the spatial NS B-spline basis requires quintic degree=5")
            if self.spatial_extension != 0.0:
                raise ValueError("spatial_extension belongs to cosine; B-spline knots use the physical normalized domain")
            validate_bspline_spec(self.spatial_knots_x, n_basis=self.spatial_mode_x+1, degree=self.spatial_degree)
            validate_bspline_spec(self.spatial_knots_y, n_basis=self.spatial_mode_y+1, degree=self.spatial_degree)
        elif self.spatial_knots_x or self.spatial_knots_y:
            raise ValueError("cosine spatial bases do not use B-spline knots")
        if self.temporal_basis == "bspline":
            if self.temporal_degree != 3 or isinstance(self.temporal_degree, bool):
                raise ValueError("the temporal NS B-spline basis requires cubic degree=3")
            validate_bspline_spec(self.temporal_knots, n_basis=self.time_order+1, degree=self.temporal_degree)
        elif self.temporal_knots:
            raise ValueError("Chebyshev temporal bases do not use B-spline knots")
        if self.fitting_method not in ("separable_ridge", "joint_smoothing"):
            raise ValueError("unsupported NS coefficient fitting method")
        if any(isinstance(value, bool) or not np.isfinite(value) or value < 0
               for value in (self.smoothing_spatial, self.smoothing_temporal)):
            raise ValueError("function-space smoothing strengths must be finite and nonnegative")
        if self.fitting_method == "separable_ridge" and (self.smoothing_spatial or self.smoothing_temporal):
            raise ValueError("function-space smoothing requires fitting_method='joint_smoothing'")

    @property
    def spatial_coefficient_count(self) -> int:
        return (int(self.spatial_mode_x) + 1) * (int(self.spatial_mode_y) + 1)

    @property
    def time_coefficient_count(self) -> int:
        return int(self.time_order) + 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Old numerical checkpoints retain their exact configuration signature.
        if self.spatial_extension == 0.0:
            payload.pop("spatial_extension")
        if self.spatial_basis == "cosine":
            for key in ("spatial_basis", "spatial_degree", "spatial_knots_x", "spatial_knots_y"):
                payload.pop(key)
        else:
            payload["spatial_knots_x"] = list(self.spatial_knots_x)
            payload["spatial_knots_y"] = list(self.spatial_knots_y)
        if self.temporal_basis == "chebyshev":
            for key in ("temporal_basis", "temporal_degree", "temporal_knots"):
                payload.pop(key)
        else:
            payload["temporal_knots"] = list(self.temporal_knots)
        if self.fitting_method == "separable_ridge":
            for key in ("fitting_method", "smoothing_spatial", "smoothing_temporal"):
                payload.pop(key)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructuredNSProjectionConfig":
        return cls(**dict(payload))


@dataclass(frozen=True)
class StructuredNSCaseCoefficients:
    psi: Any
    pressure: Any


class StructuredNSCoefficientProjector:
    """Reusable linear projector for large fixed-grid trajectory datasets."""

    def __init__(
        self,
        *,
        geometry,
        t,
        config: StructuredNSProjectionConfig | None = None,
        device: str | Any = "cpu",
        continuous_mms_abi: str = STRUCTURED_NS_MMS_ABI_VERSION,
    ) -> None:
        torch = _require_torch()
        cfg = config or StructuredNSProjectionConfig()
        cfg.validate()
        self.torch = torch
        self.config = cfg
        self.continuous_mms_abi = continuous_mms_abi
        self.device = torch.device(device)
        self.geometry = geometry
        self.geometry_spec = continuous_geometry_spec_from_contract(geometry)
        self.times = torch.as_tensor(t, dtype=torch.float64, device=self.device)
        if self.times.ndim != 1:
            raise ValueError("structured projection time coordinates must be one-dimensional")
        if cfg.temporal_basis == "chebyshev" and int(cfg.time_order) >= int(self.times.numel()):
            raise ValueError("time_order must be smaller than the number of coarse frames")
        if cfg.temporal_basis == "bspline" and self.times.numel() < 2:
            raise ValueError("temporal B-spline projection requires at least two frames")
        self.fluid = torch.as_tensor(
            np.asarray(geometry.fluid_mask, dtype=bool),
            device=self.device,
        )
        self.spatial_operators = _projection_spatial_operators(
            torch,
            geometry=geometry,
            geometry_spec=self.geometry_spec,
            config=cfg,
            device=self.device,
            continuous_mms_abi=continuous_mms_abi,
        )
        tau = (self.times - self.times[0]) / (self.times[-1] - self.times[0])
        if cfg.temporal_basis == "chebyshev":
            self.temporal_design = chebyshev_basis(tau, order=int(cfg.time_order))
        else:
            from .structured_ns_basis import axis_basis_derivatives
            self.temporal_design = axis_basis_derivatives(tau, config=cfg, axis="t")[0]
        self.velocity_projection, velocity_diagnostics = (
            _build_tikhonov_projection_operator(
                self.spatial_operators["velocity_design"],
                ridge_relative=float(cfg.ridge_relative),
            )
        )
        self.pressure_projection, pressure_diagnostics = (
            _build_tikhonov_projection_operator(
                self.spatial_operators["pressure_design_centered"],
                ridge_relative=float(cfg.ridge_relative),
            )
        )
        self.temporal_projection, temporal_diagnostics = (
            _build_tikhonov_projection_operator(
                self.temporal_design,
                ridge_relative=float(cfg.ridge_relative),
            )
        )
        self.operator_diagnostics = {
            "velocity_spatial": velocity_diagnostics,
            "pressure_spatial": pressure_diagnostics,
            "temporal": temporal_diagnostics,
        }

    def project(self, *, velocity, pressure):
        """Return `[B,2,R,Kx,Ky]` coefficients and native-grid errors."""

        torch = self.torch
        if self.config.fitting_method != "separable_ridge":
            raise ValueError("joint_smoothing must be fitted by the explicit joint NS projector")
        target_velocity = torch.as_tensor(
            velocity,
            dtype=torch.float64,
            device=self.device,
        )
        target_pressure = torch.as_tensor(
            pressure,
            dtype=torch.float64,
            device=self.device,
        )
        if target_velocity.ndim == 4:
            target_velocity = target_velocity[None]
        if target_pressure.ndim == 3:
            target_pressure = target_pressure[None]
        if target_velocity.ndim != 5 or int(target_velocity.shape[2]) != 2:
            raise ValueError("velocity must be [B,T,2,X,Y] or [T,2,X,Y]")
        expected_pressure = (
            int(target_velocity.shape[0]),
            int(target_velocity.shape[1]),
            int(target_velocity.shape[3]),
            int(target_velocity.shape[4]),
        )
        if tuple(target_pressure.shape) != expected_pressure:
            raise ValueError(
                f"pressure shape {tuple(target_pressure.shape)} does not match "
                f"{expected_pressure}"
            )
        if int(target_velocity.shape[1]) != int(self.times.numel()):
            raise ValueError("trajectory time dimension does not match projector")
        if tuple(target_velocity.shape[-2:]) != tuple(self.geometry.fluid_mask.shape):
            raise ValueError("trajectory grid does not match projector geometry")

        batch_size, nt = int(target_velocity.shape[0]), int(target_velocity.shape[1])
        spatial_count = int(self.config.spatial_coefficient_count)
        velocity_target = torch.cat(
            (
                target_velocity[:, :, 0, self.fluid],
                target_velocity[:, :, 1, self.fluid],
            ),
            dim=-1,
        )
        pressure_target = target_pressure[:, :, self.fluid]
        pressure_target = pressure_target - pressure_target.mean(dim=-1, keepdim=True)

        psi_per_time = (
            self.velocity_projection
            @ velocity_target.reshape(batch_size * nt, -1).T
        ).T.reshape(batch_size, nt, spatial_count)
        pressure_per_time = (
            self.pressure_projection
            @ pressure_target.reshape(batch_size * nt, -1).T
        ).T.reshape(batch_size, nt, spatial_count)
        spatial_only_velocity = torch.einsum(
            "fs,bts->btf",
            self.spatial_operators["velocity_design"],
            psi_per_time,
        )
        spatial_only_pressure = torch.einsum(
            "fs,bts->btf",
            self.spatial_operators["pressure_design_centered"],
            pressure_per_time,
        )
        psi_coefficients = torch.einsum(
            "rt,bts->brs",
            self.temporal_projection,
            psi_per_time,
        )
        pressure_coefficients = torch.einsum(
            "rt,bts->brs",
            self.temporal_projection,
            pressure_per_time,
        )

        psi_at_times = torch.einsum(
            "tr,brs->bts",
            self.temporal_design,
            psi_coefficients,
        )
        pressure_at_times = torch.einsum(
            "tr,brs->bts",
            self.temporal_design,
            pressure_coefficients,
        )
        predicted_velocity = torch.einsum(
            "fs,bts->btf",
            self.spatial_operators["velocity_design"],
            psi_at_times,
        )
        predicted_pressure = torch.einsum(
            "fs,bts->btf",
            self.spatial_operators["pressure_design_centered"],
            pressure_at_times,
        )
        velocity_relative = _torch_relative_l2_rows(
            predicted_velocity - velocity_target,
            velocity_target,
        )
        initial_velocity_relative = _torch_relative_l2_rows(
            predicted_velocity[:, 0] - velocity_target[:, 0],
            velocity_target[:, 0],
        )
        pressure_relative = _torch_relative_l2_rows(
            predicted_pressure - pressure_target,
            pressure_target,
        )
        spatial_velocity_relative = _torch_relative_l2_rows(
            spatial_only_velocity - velocity_target,
            velocity_target,
        )
        spatial_pressure_relative = _torch_relative_l2_rows(
            spatial_only_pressure - pressure_target,
            pressure_target,
        )
        rows = [
            {
                "velocity_relative_l2": float(velocity_relative[index].detach().cpu()),
                "initial_velocity_relative_l2": float(
                    initial_velocity_relative[index].detach().cpu()
                ),
                "pressure_gauge_relative_l2": float(
                    pressure_relative[index].detach().cpu()
                ),
                "velocity_spatial_only_relative_l2": float(
                    spatial_velocity_relative[index].detach().cpu()
                ),
                "pressure_spatial_only_gauge_relative_l2": float(
                    spatial_pressure_relative[index].detach().cpu()
                ),
            }
            for index in range(batch_size)
        ]
        coefficient_tensor = torch.stack(
            (psi_coefficients, pressure_coefficients),
            dim=1,
        ).reshape(
            batch_size,
            2,
            int(self.config.time_coefficient_count),
            int(self.config.spatial_mode_x) + 1,
            int(self.config.spatial_mode_y) + 1,
        )
        report = {
            "schema": "ns2d_structured_coefficient_projection_v2",
            "generator_schema": STRUCTURED_NS_GENERATOR_SCHEMA,
            "continuous_mms_abi": self.continuous_mms_abi,
            "coarse_trajectory_role": "shape_seed_only",
            "projection": "cached_linear_tikhonov_low_band_cosine_chebyshev",
            "config": self.config.to_dict(),
            "operator_diagnostics": self.operator_diagnostics,
            "cases": rows,
            "summary": {
                key: _summary([float(row[key]) for row in rows])
                for key in (
                    "velocity_relative_l2",
                    "initial_velocity_relative_l2",
                    "pressure_gauge_relative_l2",
                    "velocity_spatial_only_relative_l2",
                    "pressure_spatial_only_gauge_relative_l2",
                )
            },
        }
        return coefficient_tensor, report


class ContinuousStructuredNSMMS:
    """Low-band continuous streamfunction/pressure answer with exact NS source."""

    def __init__(
        self,
        coefficients: StructuredNSCaseCoefficients,
        *,
        config: StructuredNSProjectionConfig,
        geometry_spec: ContinuousGeometrySpec | dict[str, Any] | None = None,
        canonical_geometry=None,
        continuous_mms_abi: str | None = None,
        domain_bounds: tuple[float, float, float, float],
        viscosity: float,
        t0: float,
        t1: float,
        device: str | Any = "cpu",
    ) -> None:
        torch = _require_torch()
        config.validate()
        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.coefficients = StructuredNSCaseCoefficients(
            psi=torch.as_tensor(coefficients.psi, dtype=self.dtype, device=self.device),
            pressure=torch.as_tensor(
                coefficients.pressure,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        expected = (
            int(config.time_coefficient_count),
            int(config.spatial_coefficient_count),
        )
        if tuple(self.coefficients.psi.shape) != expected:
            raise ValueError(
                f"psi coefficients must have shape {expected}, got "
                f"{tuple(self.coefficients.psi.shape)}"
            )
        if tuple(self.coefficients.pressure.shape) != expected:
            raise ValueError(
                f"pressure coefficients must have shape {expected}, got "
                f"{tuple(self.coefficients.pressure.shape)}"
            )
        self.domain_bounds = tuple(float(value) for value in domain_bounds)
        if (geometry_spec is None) == (canonical_geometry is None):
            raise ValueError(
                "structured NS field requires exactly one continuous geometry spec "
                "or legacy sampled canonical geometry"
            )
        if geometry_spec is not None:
            self.geometry_spec = (
                geometry_spec
                if isinstance(geometry_spec, ContinuousGeometrySpec)
                else ContinuousGeometrySpec.from_dict(dict(geometry_spec))
            )
            self.geometry_spec.validate()
            if not np.allclose(
                np.asarray(self.geometry_spec.domain_bounds),
                np.asarray(self.domain_bounds),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("continuous geometry bounds do not match field domain")
            self.canonical_geometry = None
            requested_abi = continuous_mms_abi or STRUCTURED_NS_MMS_ABI_VERSION
            if requested_abi not in {
                LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
                LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
                LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
                STRUCTURED_NS_MMS_ABI_VERSION,
                INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
            }:
                raise ValueError(
                    f"continuous polygon geometry is incompatible with {requested_abi}"
                )
            if (self.geometry_spec.outer_boundary == "manufactured_trace"
                    and requested_abi not in {STRUCTURED_NS_MMS_ABI_VERSION, INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION}):
                raise ValueError("manufactured outer traces require the current structured NS MMS ABI")
            self.continuous_mms_abi = requested_abi
        else:
            self.geometry_spec = None
            self.canonical_geometry = torch.as_tensor(
                canonical_geometry,
                dtype=self.dtype,
                device=self.device,
            )
            if (
                self.canonical_geometry.ndim != 3
                or int(self.canonical_geometry.shape[0]) != 2
            ):
                raise ValueError(
                    "canonical geometry must have shape [2,X,Y], got "
                    f"{tuple(self.canonical_geometry.shape)}"
                )
            requested_abi = continuous_mms_abi or LEGACY_STRUCTURED_NS_MMS_ABI_VERSION
            if requested_abi != LEGACY_STRUCTURED_NS_MMS_ABI_VERSION:
                raise ValueError(
                    f"sampled canonical geometry is incompatible with {requested_abi}"
                )
            self.continuous_mms_abi = requested_abi
        # Query-independent tensors never retain coordinate graphs. Eager
        # construction avoids shared lazy cache mutation during Oracle calls.
        self._prepared_polygons = (
            _prepare_continuous_polygons(self.geometry_spec, dtype=self.dtype, device=self.device)
            if self.geometry_spec is not None else None
        )
        with torch.inference_mode(False), torch.no_grad():
            self._spatial_frequencies = (
                torch.arange(int(config.spatial_mode_x) + 1, dtype=self.dtype, device=self.device),
                torch.arange(int(config.spatial_mode_y) + 1, dtype=self.dtype, device=self.device),
            )
        self.viscosity = float(viscosity)
        self.t0 = float(t0)
        self.t1 = float(t1)
        if self.t1 <= self.t0:
            raise ValueError("continuous structured NS field requires t1 > t0")

    @property
    def boundary_condition(self):
        if self.geometry_spec is not None and self.geometry_spec.outer_boundary == "manufactured_trace":
            return {"kind": "manufactured_velocity_trace", "access": "query_boundary_velocity",
                    "derivation": "trace_of_hidden_continuous_answer"}
        return {"kind": "no_slip_velocity", "velocity": [0.0, 0.0]}

    def psi_pressure(self, physical_coords):
        psi, pressure, coords, _fluid_indicator = self._psi_pressure_with_masks(
            physical_coords
        )
        return psi, pressure, coords

    def _psi_pressure_with_masks(self, physical_coords):
        torch = self.torch
        coords = torch.as_tensor(
            physical_coords,
            dtype=self.dtype,
            device=self.device,
        )
        if coords.ndim == 3 and int(coords.shape[0]) == 1:
            coords = coords[0]
        if coords.ndim != 2 or int(coords.shape[-1]) != 3:
            raise ValueError(f"physical_coords must be [N,3], got {tuple(coords.shape)}")
        if not coords.requires_grad:
            coords = coords.clone().detach().requires_grad_(True)
        normalized = self._normalize_coordinates(coords)
        streamfunction_mask, pressure_mask = self.geometry_masks(normalized)
        if self.config.spatial_basis == "cosine":
            spatial = spatial_cosine_basis(
                normalized,
                mode_x=int(self.config.spatial_mode_x),
                mode_y=int(self.config.spatial_mode_y),
                _frequencies=self._spatial_frequencies,
                spatial_extension=float(self.config.spatial_extension),
            )
        else:
            from .structured_ns_basis import spatial_basis_with_derivatives
            spatial = spatial_basis_with_derivatives(normalized, config=self.config)[0]
        if self.config.temporal_basis == "chebyshev":
            temporal = chebyshev_basis(
                normalized[:, 2],
                order=int(self.config.time_order),
            )
        else:
            from .structured_ns_basis import axis_basis_derivatives
            temporal = axis_basis_derivatives(normalized[:, 2], config=self.config, axis="t")[0]
        psi_core = torch.einsum(
            "nr,rs,ns->n",
            temporal,
            self.coefficients.psi,
            spatial,
        )
        pressure_core = torch.einsum(
            "nr,rs,ns->n",
            temporal,
            self.coefficients.pressure,
            spatial,
        )
        pressure = (
            pressure_core
            if self.continuous_mms_abi
            in {
                LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
                STRUCTURED_NS_MMS_ABI_VERSION,
                INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
            }
            else pressure_mask * pressure_core
        )
        return streamfunction_mask * psi_core, pressure, coords, pressure_mask

    def geometry_masks(self, normalized_coords):
        if self.geometry_spec is not None:
            mask_function = (
                legacy_c3_continuous_no_slip_masks
                if self.continuous_mms_abi
                == LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION
                else interior_c3_no_slip_masks
                if self.continuous_mms_abi == INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION
                else continuous_no_slip_masks
            )
            return mask_function(
                normalized_coords,
                geometry=self.geometry_spec,
                transition=float(self.config.geometry_mask_transition),
                _prepared_polygons=(
                    self._prepared_polygons
                    if normalized_coords.dtype == self.dtype
                    and normalized_coords.device == self.coefficients.psi.device
                    else None
                ),
            )
        geometry_features = geometry_features_from_arrays(
            normalized_coords[None],
            canonical_geometry=self.canonical_geometry,
            interpolation="c3_smoothstep",
        )[0]
        return structured_geometry_masks(
            normalized_coords,
            geometry_features,
            transition=float(self.config.geometry_mask_transition),
        )

    def velocity_pressure(self, physical_coords):
        psi, pressure, coords = self.psi_pressure(physical_coords)
        gradient = self.torch.autograd.grad(
            psi.sum(),
            coords,
            create_graph=False,
            retain_graph=False,
        )[0]
        return gradient[:, 1], -gradient[:, 0], pressure

    def quantities(self, physical_coords, *, include_laplacian_components: bool = False):
        coords = self.torch.as_tensor(
            physical_coords,
            dtype=self.dtype,
            device=self.device,
        )
        if coords.ndim == 3 and int(coords.shape[0]) == 1:
            coords = coords[0]
        coords = coords.clone().detach().requires_grad_(True)
        psi, pressure, coords, fluid_indicator = self._psi_pressure_with_masks(coords)
        quantities = navier_stokes_quantities_from_psi_pressure(
            psi,
            pressure,
            coords,
            viscosity=self.viscosity,
            include_laplacian_components=bool(include_laplacian_components),
        )
        if self.continuous_mms_abi in {
            LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
            STRUCTURED_NS_MMS_ABI_VERSION,
            INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
        }:
            # Reuse this query's indicator; no graph or coordinates survive
            # across calls, and exact coordinate differentiation is unchanged.
            for name in ("forcing_u", "forcing_v", "residual_u", "residual_v"):
                quantities[name] = quantities[name] * fluid_indicator
            quantities["fluid_indicator"] = fluid_indicator
        return quantities

    def sample_grid(
        self,
        *,
        x,
        y,
        t,
        batch_points: int = 2048,
        derivative_backend: str = REFERENCE_AUTOGRAD_BACKEND_ID,
    ) -> TriplaneDecodedFields:
        if str(derivative_backend) != REFERENCE_AUTOGRAD_BACKEND_ID:
            raise ValueError(
                "structured NS answer-first currently supports exact torch.autograd derivatives"
            )
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        t_values = np.asarray(t, dtype=np.float64)
        tt, xx, yy = np.meshgrid(t_values, x_values, y_values, indexing="ij")
        coordinates = np.stack((xx, yy, tt), axis=-1).reshape(-1, 3)
        names = (
            "u",
            "v",
            "d_x_u",
            "d_y_v",
            "pressure",
            "psi",
            "forcing_u",
            "forcing_v",
            "divergence",
            "residual_u",
            "residual_v",
            "d_t_u",
            "d_t_v",
            "d_y_u",
            "d_x_v",
            "laplacian_u",
            "laplacian_v",
            "d_x_p",
            "d_y_p",
        )
        if self.continuous_mms_abi in {
            LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
            STRUCTURED_NS_MMS_ABI_VERSION,
            INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
        }:
            names = (*names, "fluid_indicator")
        parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
        chunk = max(1, int(batch_points))
        for start in range(0, int(coordinates.shape[0]), chunk):
            quantities = self.quantities(coordinates[start : start + chunk])
            for name in names:
                parts[name].append(
                    quantities[name].detach().cpu().numpy().astype(np.float64)
                )
            del quantities
        shape = (len(t_values), len(x_values), len(y_values))
        arrays = {
            name: np.concatenate(values).reshape(shape)
            for name, values in parts.items()
        }
        velocity = np.stack((arrays["u"], arrays["v"]), axis=1)
        forcing = np.stack((arrays["forcing_u"], arrays["forcing_v"]), axis=1)
        residual = np.stack((arrays["residual_u"], arrays["residual_v"]), axis=1)
        provenance = structured_ns_mms_provenance(
            self.config,
            continuous_mms_abi=self.continuous_mms_abi,
            geometry_spec=self.geometry_spec,
        )
        continuous_statistics = (
            _continuous_exact_physical_statistics(arrays)
            if self.continuous_mms_abi
            in {
                LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
                STRUCTURED_NS_MMS_ABI_VERSION,
                INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
            }
            else None
        )
        return TriplaneDecodedFields(
            velocity=velocity,
            pressure=arrays["pressure"],
            forcing=forcing,
            divergence=arrays["divergence"],
            residual=residual,
            psi=arrays["psi"],
            t=t_values,
            viscosity=self.viscosity,
            metadata={
                "truth_type": "continuous_structured_streamfunction",
                "forcing_type": "continuous_exact_coordinate_derivatives_from_hidden_solution",
                "continuous_mms_abi": self.continuous_mms_abi,
                "time_interval": [self.t0, self.t1],
                "domain_bounds": list(self.domain_bounds),
                "canonical_state": True,
                "grid_arrays_are_materialized_views": True,
                "source_materialization": {
                    "backend_id": REFERENCE_AUTOGRAD_BACKEND_ID,
                    "batch_points": int(batch_points),
                },
                "continuous_divergence_abs_max": float(
                    np.max(np.abs(arrays["divergence"]))
                ),
                "continuous_divergence_rel_max": continuous_divergence_relative_max(
                    arrays["divergence"],
                    arrays["d_x_u"],
                    arrays["d_y_v"],
                ),
                "continuous_exact_physical_statistics": continuous_statistics,
                **provenance,
            },
        )

    def _normalize_coordinates(self, coordinates):
        x_min, x_max, y_min, y_max = self.domain_bounds
        return self.torch.stack(
            (
                (coordinates[:, 0] - x_min) / (x_max - x_min),
                (coordinates[:, 1] - y_min) / (y_max - y_min),
                (coordinates[:, 2] - self.t0) / (self.t1 - self.t0),
            ),
            dim=-1,
        )


def _continuous_exact_physical_statistics(arrays: dict[str, np.ndarray]):
    fluid = np.asarray(arrays["fluid_indicator"] > 0.5, dtype=bool)
    velocity = np.stack((arrays["u"], arrays["v"]), axis=1)
    forcing = np.stack((arrays["forcing_u"], arrays["forcing_v"]), axis=1)
    time_derivative = np.stack((arrays["d_t_u"], arrays["d_t_v"]), axis=1)
    laplacian = np.stack((arrays["laplacian_u"], arrays["laplacian_v"]), axis=1)
    pressure_gradient = np.stack((arrays["d_x_p"], arrays["d_y_p"]), axis=1)
    vorticity = arrays["d_x_v"] - arrays["d_y_u"]
    fluid_spatial = fluid[0]
    repeated_initial = np.repeat(velocity[:1], velocity.shape[0], axis=0)
    return {
        "velocity_rms": _masked_vector_rms(velocity, fluid),
        "forcing_rms": _masked_vector_rms(forcing, fluid),
        "time_derivative_rms": _masked_vector_rms(time_derivative, fluid),
        "vorticity_rms": float(np.sqrt(np.mean(vorticity[fluid] ** 2))),
        "velocity_laplacian_rms": _masked_vector_rms(laplacian, fluid),
        "pressure_gradient_rms": _masked_vector_rms(pressure_gradient, fluid),
        "pressure_rms": float(
            np.sqrt(np.mean(np.asarray(arrays["pressure"])[fluid] ** 2))
        ),
        "repeat_initial_difficulty": float(
            np.linalg.norm(
                repeated_initial[:, :, fluid_spatial]
                - velocity[:, :, fluid_spatial]
            )
            / max(np.linalg.norm(velocity[:, :, fluid_spatial]), 1.0e-12)
        ),
    }


def _masked_vector_rms(values: np.ndarray, fluid: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array.transpose(1, 0, 2, 3)[:, fluid] ** 2)))


def fit_coarse_trajectory_to_structured_fields(
    *,
    velocity,
    pressure,
    geometry,
    t,
    viscosity: float,
    config: StructuredNSProjectionConfig | None = None,
    device: str | Any = "cpu",
    continuous_mms_abi: str = STRUCTURED_NS_MMS_ABI_VERSION,
) -> tuple[list[ContinuousStructuredNSMMS], dict[str, Any]]:
    """Deterministically project coarse NS values into a low-band exact field."""

    torch = _require_torch()
    cfg = config or StructuredNSProjectionConfig()
    cfg.validate()
    if cfg.fitting_method != "separable_ridge":
        raise ValueError("joint_smoothing must be fitted by the explicit joint NS projector")
    device = torch.device(device)
    target_velocity = torch.as_tensor(velocity, dtype=torch.float64, device=device)
    target_pressure = torch.as_tensor(pressure, dtype=torch.float64, device=device)
    if target_velocity.ndim == 4:
        target_velocity = target_velocity[None]
    if target_pressure.ndim == 3:
        target_pressure = target_pressure[None]
    if target_velocity.ndim != 5 or int(target_velocity.shape[2]) != 2:
        raise ValueError("velocity must be [B,T,2,X,Y] or [T,2,X,Y]")
    expected_pressure = (
        int(target_velocity.shape[0]),
        int(target_velocity.shape[1]),
        int(target_velocity.shape[3]),
        int(target_velocity.shape[4]),
    )
    if tuple(target_pressure.shape) != expected_pressure:
        raise ValueError(
            f"pressure shape {tuple(target_pressure.shape)} does not match {expected_pressure}"
        )
    times = torch.as_tensor(t, dtype=torch.float64, device=device)
    if times.ndim != 1 or int(times.numel()) != int(target_velocity.shape[1]):
        raise ValueError("time coordinates do not match the trajectory")
    if cfg.temporal_basis == "chebyshev" and int(cfg.time_order) >= int(times.numel()):
        raise ValueError("time_order must be smaller than the number of coarse frames")
    if cfg.temporal_basis == "bspline" and times.numel() < 2:
        raise ValueError("temporal B-spline projection requires at least two frames")
    if tuple(target_velocity.shape[-2:]) != tuple(geometry.fluid_mask.shape):
        raise ValueError("coarse trajectory grid does not match geometry")

    geometry_spec = continuous_geometry_spec_from_contract(geometry)
    spatial_operators = _projection_spatial_operators(
        torch,
        geometry=geometry,
        geometry_spec=geometry_spec,
        config=cfg,
        device=device,
        continuous_mms_abi=continuous_mms_abi,
    )
    fluid = torch.as_tensor(
        np.asarray(geometry.fluid_mask, dtype=bool),
        device=device,
    )
    velocity_target = torch.cat(
        (
            target_velocity[:, :, 0, fluid],
            target_velocity[:, :, 1, fluid],
        ),
        dim=-1,
    )
    pressure_target = target_pressure[:, :, fluid]
    pressure_target = pressure_target - pressure_target.mean(dim=-1, keepdim=True)
    batch_size, nt = int(target_velocity.shape[0]), int(target_velocity.shape[1])
    spatial_count = int(cfg.spatial_coefficient_count)
    psi_per_time, velocity_solve = _solve_tikhonov(
        spatial_operators["velocity_design"],
        velocity_target.reshape(batch_size * nt, -1).T,
        ridge_relative=float(cfg.ridge_relative),
    )
    pressure_per_time, pressure_solve = _solve_tikhonov(
        spatial_operators["pressure_design_centered"],
        pressure_target.reshape(batch_size * nt, -1).T,
        ridge_relative=float(cfg.ridge_relative),
    )
    psi_per_time = psi_per_time.T.reshape(batch_size, nt, spatial_count)
    pressure_per_time = pressure_per_time.T.reshape(batch_size, nt, spatial_count)
    tau = (times - times[0]) / (times[-1] - times[0])
    if cfg.temporal_basis == "chebyshev":
        temporal_design = chebyshev_basis(tau, order=int(cfg.time_order))
    else:
        from .structured_ns_basis import axis_basis_derivatives
        temporal_design = axis_basis_derivatives(tau, config=cfg, axis="t")[0]
    psi_coefficients, psi_time_solve = _fit_temporal_coefficients(
        temporal_design,
        psi_per_time,
        ridge_relative=float(cfg.ridge_relative),
    )
    pressure_coefficients, pressure_time_solve = _fit_temporal_coefficients(
        temporal_design,
        pressure_per_time,
        ridge_relative=float(cfg.ridge_relative),
    )
    fields = [
        ContinuousStructuredNSMMS(
            StructuredNSCaseCoefficients(
                psi=psi_coefficients[index],
                pressure=pressure_coefficients[index],
            ),
            config=cfg,
            geometry_spec=geometry_spec,
            continuous_mms_abi=continuous_mms_abi,
            domain_bounds=geometry_spec.domain_bounds,
            viscosity=float(viscosity),
            t0=float(times[0]),
            t1=float(times[-1]),
            device=device,
        )
        for index in range(batch_size)
    ]
    evaluation_rows = []
    for index, field in enumerate(fields):
        evaluation_rows.append(
            evaluate_structured_projection(
                field,
                target_velocity=target_velocity[index],
                target_pressure=target_pressure[index],
                geometry=geometry,
                t=times,
                batch_points=int(cfg.evaluation_batch_points),
            )
        )
    report = {
        "schema": "ns2d_structured_answer_first_projection_v7",
        "generator_schema": STRUCTURED_NS_GENERATOR_SCHEMA,
        "continuous_mms_abi": continuous_mms_abi,
        "coarse_trajectory_role": "shape_seed_only",
        "learned_decoder_used": False,
        "optimized_neural_parameters": False,
        "projection": "deterministic_tikhonov_low_band_cosine_chebyshev",
        "config": cfg.to_dict(),
        "operator_diagnostics": {
            "velocity_spatial": velocity_solve,
            "pressure_spatial": pressure_solve,
            "psi_temporal": psi_time_solve,
            "pressure_temporal": pressure_time_solve,
        },
        "cases": evaluation_rows,
        "summary": {
            key: _summary([float(row[key]) for row in evaluation_rows])
            for key in (
                "velocity_relative_l2",
                "initial_velocity_relative_l2",
                "pressure_gauge_relative_l2",
            )
        },
    }
    return fields, report


def evaluate_structured_projection(
    field: ContinuousStructuredNSMMS,
    *,
    target_velocity,
    target_pressure,
    geometry,
    t,
    batch_points: int,
) -> dict[str, float]:
    torch = _require_torch()
    times = torch.as_tensor(t, dtype=torch.float64).detach().cpu().numpy()
    tt, xx, yy = np.meshgrid(times, geometry.x, geometry.y, indexing="ij")
    coordinates = np.stack((xx, yy, tt), axis=-1).reshape(-1, 3)
    u_parts = []
    v_parts = []
    pressure_parts = []
    for start in range(0, len(coordinates), max(1, int(batch_points))):
        u, v, pressure = field.velocity_pressure(
            coordinates[start : start + max(1, int(batch_points))]
        )
        u_parts.append(u.detach().cpu().numpy())
        v_parts.append(v.detach().cpu().numpy())
        pressure_parts.append(pressure.detach().cpu().numpy())
    shape = (len(times), geometry.config.nx, geometry.config.ny)
    predicted_velocity = np.stack(
        (
            np.concatenate(u_parts).reshape(shape),
            np.concatenate(v_parts).reshape(shape),
        ),
        axis=1,
    )
    predicted_pressure = np.concatenate(pressure_parts).reshape(shape)
    truth_velocity = np.asarray(target_velocity.detach().cpu(), dtype=np.float64)
    truth_pressure = np.asarray(target_pressure.detach().cpu(), dtype=np.float64)
    fluid = np.asarray(geometry.fluid_mask, dtype=bool)
    velocity_error = predicted_velocity[:, :, fluid] - truth_velocity[:, :, fluid]
    pressure_pred = predicted_pressure[:, fluid]
    pressure_truth = truth_pressure[:, fluid]
    pressure_pred -= pressure_pred.mean(axis=1, keepdims=True)
    pressure_truth -= pressure_truth.mean(axis=1, keepdims=True)
    return {
        "velocity_relative_l2": _relative_l2(
            velocity_error,
            truth_velocity[:, :, fluid],
        ),
        "initial_velocity_relative_l2": _relative_l2(
            velocity_error[0],
            truth_velocity[0, :, fluid],
        ),
        "pressure_gauge_relative_l2": _relative_l2(
            pressure_pred - pressure_truth,
            pressure_truth,
        ),
    }


def structured_geometry_masks(normalized_coords, geometry_features, *, transition: float):
    obstacle = c3_unit_ramp(geometry_features[..., 0] / float(transition))
    wall = c3_unit_ramp(geometry_features[..., 1] / float(transition))
    x = normalized_coords[..., 0]
    y = normalized_coords[..., 1]
    rectangular = (
        c3_unit_ramp(x / float(transition))
        * c3_unit_ramp((1.0 - x) / float(transition))
        * c3_unit_ramp(y / float(transition))
        * c3_unit_ramp((1.0 - y) / float(transition))
    )
    return obstacle * wall * rectangular, obstacle


def c3_unit_ramp(value):
    clipped = value.clamp(min=0.0, max=1.0)
    return clipped**4 * (
        35.0 - 84.0 * clipped + 70.0 * clipped**2 - 20.0 * clipped**3
    )


def spatial_cosine_basis(normalized_coords, *, mode_x: int, mode_y: int, _frequencies=None,
                         spatial_extension: float = 0.0):
    torch = _require_torch()
    x = normalized_coords[..., 0:1]
    y = normalized_coords[..., 1:2]
    if spatial_extension:
        width = 1.0 + 2.0 * float(spatial_extension)
        x, y = (x + spatial_extension) / width, (y + spatial_extension) / width
    if _frequencies is None:
        kx = torch.arange(
            int(mode_x) + 1,
            dtype=normalized_coords.dtype,
            device=normalized_coords.device,
        )
        ky = torch.arange(
            int(mode_y) + 1,
            dtype=normalized_coords.dtype,
            device=normalized_coords.device,
        )
    else:
        kx, ky = _frequencies
    x_values = torch.cos(torch.pi * x * kx)
    y_values = torch.cos(torch.pi * y * ky)
    return (x_values[..., :, None] * y_values[..., None, :]).reshape(
        *normalized_coords.shape[:-1],
        (int(mode_x) + 1) * (int(mode_y) + 1),
    )


def spatial_cosine_basis_with_derivatives(
    normalized_coords,
    *,
    mode_x: int,
    mode_y: int,
    spatial_extension: float = 0.0,
):
    torch = _require_torch()
    x = normalized_coords[..., 0:1]
    y = normalized_coords[..., 1:2]
    if spatial_extension:
        width = 1.0 + 2.0 * float(spatial_extension)
        x, y = (x + spatial_extension) / width, (y + spatial_extension) / width
    kx = torch.arange(
        int(mode_x) + 1,
        dtype=normalized_coords.dtype,
        device=normalized_coords.device,
    )
    ky = torch.arange(
        int(mode_y) + 1,
        dtype=normalized_coords.dtype,
        device=normalized_coords.device,
    )
    cos_x = torch.cos(torch.pi * x * kx)
    cos_y = torch.cos(torch.pi * y * ky)
    d_x = -torch.pi * kx * torch.sin(torch.pi * x * kx)
    d_y = -torch.pi * ky * torch.sin(torch.pi * y * ky)
    if spatial_extension:
        d_x, d_y = d_x / width, d_y / width
    shape = normalized_coords.shape[:-1] + (
        (int(mode_x) + 1) * (int(mode_y) + 1),
    )
    basis = (cos_x[..., :, None] * cos_y[..., None, :]).reshape(shape)
    basis_dx = (d_x[..., :, None] * cos_y[..., None, :]).reshape(shape)
    basis_dy = (cos_x[..., :, None] * d_y[..., None, :]).reshape(shape)
    return basis, basis_dx, basis_dy


def chebyshev_basis(tau, *, order: int):
    torch = _require_torch()
    values = torch.as_tensor(tau)
    z = 2.0 * values - 1.0
    rows = [torch.ones_like(z)]
    if int(order) >= 1:
        rows.append(z)
    for _degree in range(2, int(order) + 1):
        rows.append(2.0 * z * rows[-1] - rows[-2])
    return torch.stack(rows, dim=-1)


def structured_ns_mms_provenance(
    config: StructuredNSProjectionConfig,
    *,
    continuous_mms_abi: str = STRUCTURED_NS_MMS_ABI_VERSION,
    geometry_spec: ContinuousGeometrySpec | None = None,
) -> dict[str, Any]:
    if continuous_mms_abi not in SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS:
        raise ValueError(f"unsupported structured NS continuous ABI: {continuous_mms_abi}")
    provenance = continuous_mms_provenance(
        source_derivative_backend=REFERENCE_AUTOGRAD_BACKEND_ID
    )
    spatial_basis = (
        f"tensor_product_cosine_modes_0_to_{config.spatial_mode_x}_"
        f"and_0_to_{config.spatial_mode_y}"
    )
    provenance.update(
        {
            "hidden_representation": "direct_low_band_streamfunction_pressure_coefficients",
            "spatial_interpolation_or_basis": spatial_basis,
            "temporal_interpolation_or_basis": (
                f"global_chebyshev_order_{config.time_order}"
            ),
            "pressure_representation_or_method": "low_band_cosine_chebyshev_with_C3_obstacle_mask",
            "generator_projection": "deterministic_tikhonov_no_learned_decoder",
            "structured_projection_config": config.to_dict(),
        }
    )
    if continuous_mms_abi == LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION:
        provenance.update(
            {
                "boundary_contract_version": "continuous_no_slip_c3_polygon_v2",
                "geometry_contract_version": "continuous_geometry_convex_polygons_v1",
                "boundary_representation": (
                    "C3 rectangular wall ramps and generic convex polygon edge ramps"
                ),
                "geometry_representation": (
                    "serialized anonymous convex polygon components evaluated continuously"
                ),
                "pressure_representation_or_method": (
                    "low_band_cosine_chebyshev_with_continuous_C3_polygon_mask"
                ),
            }
        )
    elif continuous_mms_abi in {
        LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
        LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
        STRUCTURED_NS_MMS_ABI_VERSION,
        INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
    }:
        provenance.update(
            {
                "boundary_contract_version": "continuous_no_slip_fluid_side_polygon_v2",
                "geometry_contract_version": (
                    "continuous_geometry_convex_polygons_v2"
                    if continuous_mms_abi in {STRUCTURED_NS_MMS_ABI_VERSION, INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION}
                    else "continuous_geometry_convex_polygons_v1"
                ),
                "boundary_representation": (
                    "one-sided quadratic no-slip wall ramps with C3 saturation"
                ),
                "geometry_representation": (
                    "serialized anonymous convex polygon components evaluated continuously"
                ),
            }
        )
        if continuous_mms_abi == LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION:
            provenance["pressure_representation_or_method"] = (
                "low-band cosine Chebyshev in fluid with constant solid extension"
            )
        else:
            provenance.update(
                {
                    "pressure_representation_or_method": (
                        "low-band cosine Chebyshev with smooth nonphysical solid extension"
                    ),
                    "source_domain_restriction": (
                        "exact_fluid_operator_then_zero_outside_continuous_domain"
                    ),
                }
            )
            if continuous_mms_abi in {STRUCTURED_NS_MMS_ABI_VERSION, INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION}:
                provenance.update(
                    {
                        "boundary_contract_version": (
                            "continuous_no_slip_public_query_domain_v3"
                        ),
                        "continuous_query_domain": (
                            "first_to_last_public_cell_center_with_exact_endpoint_walls"
                        ),
                    }
                )
    if continuous_mms_abi == INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION:
        provenance.update({
            "boundary_contract_version": "continuous_no_slip_interior_c3_polygon_v4",
            "boundary_representation": (
                "quadratic outer/flat-face wall ramps; r(L4_positive_edge_distances) "
                "at polygon corners; C3 across internal supporting-line seams"
            ),
            "polygon_clearance": "r((sum(max(d_i/transition,0)^4))^(1/4))",
            "regularity_scope": "C3 in the open fluid; no C3 claim across physical walls or vertices",
        })
    if config.spatial_extension:
        provenance["cosine_expansion_extension"] = float(config.spatial_extension)
        provenance["cosine_expansion_domain_role"] = "basis_only_physical_measurement_domain_unchanged"
    if geometry_spec is not None and geometry_spec.outer_boundary == "manufactured_trace":
        provenance.update(
            boundary_contract_version=("continuous_manufactured_outer_trace_with_interior_c3_obstacles_v2"
                                       if continuous_mms_abi == INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION
                                       else "continuous_manufactured_outer_trace_with_no_slip_obstacles_v1"),
            boundary_representation=("manufactured outer velocity trace; interior C3 no-slip obstacle ramps"
                                     if continuous_mms_abi == INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION
                                     else "manufactured outer velocity trace; unchanged no-slip obstacle ramps"),
            continuous_query_domain="declared physical measurement domain including outer traces",
        )
    if not config.is_legacy_basis:
        provenance["spatial_interpolation_or_basis"] = (
            f"tensor_product_bspline_degree_{config.spatial_degree}"
            if config.spatial_basis == "bspline" else spatial_basis
        )
        provenance["temporal_interpolation_or_basis"] = (
            f"clamped_bspline_degree_{config.temporal_degree}_count_{config.time_coefficient_count}"
            if config.temporal_basis == "bspline" else f"global_chebyshev_order_{config.time_order}"
        )
        provenance["pressure_representation_or_method"] = "declared_tensor_product_basis_with_existing_pressure_gauge_and_geometry_contract"
        provenance["generator_projection"] = config.fitting_method
    if config.fitting_method == "joint_smoothing":
        provenance["generator_projection"] = config.fitting_method
    return provenance


def structured_answer_first_provenance(
    config: StructuredNSProjectionConfig,
    *,
    continuous_mms_abi: str = STRUCTURED_NS_MMS_ABI_VERSION,
) -> dict[str, Any]:
    return {
        "generator_schema": STRUCTURED_NS_GENERATOR_SCHEMA,
        "generator_backend_and_version": "coarse_seeded_structured_answer_first_continuous_mms_v9",
        "continuous_mms_abi": continuous_mms_abi,
        "truth_primary": "continuous_hidden_streamfunction_pressure",
        "low_fidelity_equation_solve_role": "shape_seed_only",
        "low_fidelity_equation_solve_is_truth": False,
        "coarse_equation_source_is_discarded": True,
        "public_source_is_recomputed_from_hidden_answer": True,
        "diffusion_model_used": False,
        "learned_decoder_used": False,
        "diagnostic_field_stack_allowed": False,
        "velocity_projection_enabled": False,
        "D0_distribution_gate_enabled": False,
        **structured_ns_mms_provenance(
            config,
            continuous_mms_abi=continuous_mms_abi,
        ),
    }


def write_structured_continuous_case_state(
    case_dir: str | Path,
    *,
    field: ContinuousStructuredNSMMS,
) -> Path:
    torch = _require_torch()
    if field.continuous_mms_abi not in {STRUCTURED_NS_MMS_ABI_VERSION, INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION}:
        raise ValueError("new structured cases must use the continuous geometry ABI")
    if field.geometry_spec is None:
        raise ValueError("new structured cases are missing a continuous geometry spec")
    path = Path(case_dir)
    path.mkdir(parents=True, exist_ok=True)
    state_path = path / "hidden_continuous_case.pt"
    provenance = structured_ns_mms_provenance(
        field.config,
        continuous_mms_abi=field.continuous_mms_abi,
        geometry_spec=field.geometry_spec,
    )
    torch.save(
        {
            "continuous_mms_abi": field.continuous_mms_abi,
            "config": field.config.to_dict(),
            "coefficients": {
                "psi": field.coefficients.psi.detach().cpu(),
                "pressure": field.coefficients.pressure.detach().cpu(),
            },
            "continuous_geometry": field.geometry_spec.to_dict(),
            "domain_bounds": list(field.domain_bounds),
            "time_interval": [field.t0, field.t1],
            "viscosity": field.viscosity,
            "source_provenance": provenance,
        },
        state_path,
    )
    manifest = {
        "continuous_mms_abi": field.continuous_mms_abi,
        "canonical_truth": state_path.name,
        "solution": "continuous low-band streamfunction/pressure queried at arbitrary physical coordinates",
        "forcing": "continuous Navier-Stokes forcing derived by exact coordinate differentiation from the hidden solution",
        "sampled_grid_cache": "hidden_truth_cache.npz is a materialized view, not the truth definition",
        **provenance,
    }
    (path / "hidden_continuous_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    hidden_manifest_path = path / "hidden_truth_manifest.json"
    if hidden_manifest_path.exists():
        hidden_manifest = json.loads(hidden_manifest_path.read_text())
        hidden_manifest.update(
            {
                "truth_type": "continuous_structured_streamfunction",
                "continuous_mms_abi": field.continuous_mms_abi,
                "canonical_truth": state_path.name,
                "hidden_arrays_role": "materialized_cache_only",
                **provenance,
            }
        )
        hidden_manifest_path.write_text(
            json.dumps(hidden_manifest, indent=2, sort_keys=True) + "\n"
        )
    public_question_path = path / "public_question.json"
    if public_question_path.exists():
        public_question = json.loads(public_question_path.read_text())
        public_question["case_source"] = "fixed_geometry_incompressible_flow"
        public_question["forcing_derivation"] = "provided_public_arrays_forcing"
        public_question.pop("truth_representation", None)
        public_question_path.write_text(
            json.dumps(public_question, indent=2, sort_keys=True) + "\n"
        )
    return state_path


def load_structured_continuous_case(
    case_dir: str | Path,
    *,
    device: str = "cpu",
) -> ContinuousStructuredNSMMS:
    torch = _require_torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(
        Path(case_dir) / "hidden_continuous_case.pt",
        map_location=device,
        weights_only=False,
    )
    continuous_mms_abi = state.get("continuous_mms_abi")
    if continuous_mms_abi not in SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS:
        raise ValueError(
            f"unsupported structured NS continuous ABI: {continuous_mms_abi}"
        )
    geometry_argument = (
        {
            "geometry_spec": state["continuous_geometry"],
            "continuous_mms_abi": continuous_mms_abi,
        }
        if continuous_mms_abi
        in {
            LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
            LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
            LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
            STRUCTURED_NS_MMS_ABI_VERSION,
            INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
        }
        else {"canonical_geometry": state["canonical_geometry"]}
    )
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(
            psi=state["coefficients"]["psi"],
            pressure=state["coefficients"]["pressure"],
        ),
        config=StructuredNSProjectionConfig.from_dict(state["config"]),
        **geometry_argument,
        domain_bounds=tuple(state["domain_bounds"]),
        viscosity=float(state["viscosity"]),
        t0=float(state["time_interval"][0]),
        t1=float(state["time_interval"][1]),
        device=device,
    )


def _projection_spatial_operators(
    torch,
    *,
    geometry,
    geometry_spec: ContinuousGeometrySpec,
    config: StructuredNSProjectionConfig,
    device,
    continuous_mms_abi: str = STRUCTURED_NS_MMS_ABI_VERSION,
):
    if continuous_mms_abi not in {STRUCTURED_NS_MMS_ABI_VERSION, INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION}:
        raise ValueError("structured projection requires an explicit v5 or v6 mask contract")
    mask_function = (interior_c3_no_slip_masks
                     if continuous_mms_abi == INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION
                     else continuous_no_slip_masks)
    nx, ny = int(geometry.config.nx), int(geometry.config.ny)
    x_min, x_max, y_min, y_max = geometry_spec.domain_bounds
    x = torch.as_tensor(
        (np.asarray(geometry.x) - x_min) / (x_max - x_min),
        dtype=torch.float64,
        device=device,
    )
    y = torch.as_tensor(
        (np.asarray(geometry.y) - y_min) / (y_max - y_min),
        dtype=torch.float64,
        device=device,
    )
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    normalized = torch.stack(
        (xx, yy, torch.zeros_like(xx)),
        dim=-1,
    ).reshape(-1, 3).requires_grad_(True)
    streamfunction_mask, _fluid_indicator = mask_function(
        normalized,
        geometry=geometry_spec,
        transition=float(config.geometry_mask_transition),
    )
    mask_gradient = (torch.autograd.grad(
        streamfunction_mask.sum(),
        normalized,
        create_graph=False,
        retain_graph=False,
    )[0] if streamfunction_mask.requires_grad else torch.zeros_like(normalized))
    if config.spatial_basis == "cosine":
        basis, basis_dx, basis_dy = spatial_cosine_basis_with_derivatives(
            normalized.detach(),
            mode_x=int(config.spatial_mode_x),
            mode_y=int(config.spatial_mode_y),
            spatial_extension=float(config.spatial_extension),
        )
    else:
        from .structured_ns_basis import spatial_basis_with_derivatives
        basis, basis_dx, basis_dy = spatial_basis_with_derivatives(normalized.detach(), config=config)
    psi_dx_normalized = (
        mask_gradient[:, 0:1] * basis
        + streamfunction_mask.detach()[:, None] * basis_dx
    )
    psi_dy_normalized = (
        mask_gradient[:, 1:2] * basis
        + streamfunction_mask.detach()[:, None] * basis_dy
    )
    fluid = torch.as_tensor(
        np.asarray(geometry.fluid_mask, dtype=bool).reshape(-1),
        device=device,
    )
    u_design = psi_dy_normalized[fluid] / float(y_max - y_min)
    v_design = -psi_dx_normalized[fluid] / float(x_max - x_min)
    # The current continuous ABI evaluates pressure as a smooth, unmasked
    # cosine field. Fit that same field definition on fluid points.
    pressure_design = basis[fluid]
    return {
        "velocity_design": torch.cat((u_design, v_design), dim=0),
        "pressure_design_centered": pressure_design
        - pressure_design.mean(dim=0, keepdim=True),
        "grid_shape": (nx, ny),
    }


def _fit_temporal_coefficients(design, per_time, *, ridge_relative: float):
    batch_size, nt, spatial_count = per_time.shape
    targets = per_time.permute(1, 0, 2).reshape(nt, batch_size * spatial_count)
    solution, diagnostics = _solve_tikhonov(
        design,
        targets,
        ridge_relative=ridge_relative,
    )
    coefficients = solution.reshape(
        int(design.shape[1]),
        batch_size,
        spatial_count,
    ).permute(1, 0, 2)
    return coefficients, diagnostics


def _build_tikhonov_projection_operator(design, *, ridge_relative: float):
    """Precompute the linear map used by repeated fixed-design projections."""

    torch = _require_torch()
    full_matrix = torch.as_tensor(design, dtype=torch.float64)
    if full_matrix.ndim != 2:
        raise ValueError("Tikhonov projection design must be a matrix")
    raw_column_scale = torch.sqrt(torch.mean(full_matrix.square(), dim=0))
    maximum_scale = raw_column_scale.max().clamp_min(1.0e-15)
    active = raw_column_scale > maximum_scale * 1.0e-10
    if not bool(torch.any(active)):
        raise ValueError("least-squares design has no observable columns")
    matrix = full_matrix[:, active]
    column_scale = raw_column_scale[active].clamp_min(1.0e-15)
    normalized = matrix / column_scale
    if float(ridge_relative) > 0.0:
        regularizer_scale = torch.sqrt(
            torch.as_tensor(
                float(ridge_relative) * float(matrix.shape[0]),
                dtype=normalized.dtype,
                device=normalized.device,
            )
        )
        reference_column_scale = torch.median(column_scale)
        physical_coefficient_weights = reference_column_scale / column_scale
        diagonal = (regularizer_scale * physical_coefficient_weights).square()
        normal_matrix = normalized.T @ normalized + torch.diag(diagonal)
        normalized_projection = torch.linalg.solve(
            normal_matrix,
            normalized.T,
        )
        solver = "regularized_normal_equations"
    else:
        normalized_projection = torch.linalg.pinv(normalized)
        physical_coefficient_weights = torch.ones_like(column_scale)
        solver = "moore_penrose_pseudoinverse"
    active_projection = normalized_projection / column_scale[:, None]
    projection = torch.zeros(
        (int(full_matrix.shape[1]), int(full_matrix.shape[0])),
        dtype=active_projection.dtype,
        device=active_projection.device,
    )
    projection[active] = active_projection
    singular_values = torch.linalg.svdvals(matrix)
    condition = float(
        (singular_values[0] / singular_values[-1].clamp_min(1.0e-15))
        .detach()
        .cpu()
    )
    return projection, {
        "rows": int(full_matrix.shape[0]),
        "columns": int(full_matrix.shape[1]),
        "active_columns": int(active.sum().detach().cpu()),
        "inactive_columns": int((~active).sum().detach().cpu()),
        "condition_number": condition,
        "ridge_relative": float(ridge_relative),
        "solver": solver,
        "ridge_parameterization": (
            "physical_coefficient_l2_relative_to_median_design_column_energy"
        ),
        "ridge_weight_min": float(physical_coefficient_weights.min().detach().cpu()),
        "ridge_weight_max": float(physical_coefficient_weights.max().detach().cpu()),
    }


def _solve_tikhonov(
    design,
    targets,
    *,
    ridge_relative: float,
):
    torch = _require_torch()
    full_matrix = torch.as_tensor(design, dtype=torch.float64)
    original_rhs = torch.as_tensor(
        targets,
        dtype=torch.float64,
        device=full_matrix.device,
    )
    if (
        full_matrix.ndim != 2
        or original_rhs.ndim != 2
        or int(full_matrix.shape[0]) != int(original_rhs.shape[0])
    ):
        raise ValueError("least-squares design and targets have incompatible shapes")
    raw_column_scale = torch.sqrt(torch.mean(full_matrix.square(), dim=0))
    maximum_scale = raw_column_scale.max().clamp_min(1.0e-15)
    active = raw_column_scale > maximum_scale * 1.0e-10
    if not bool(torch.any(active)):
        raise ValueError("least-squares design has no observable columns")
    matrix = full_matrix[:, active]
    column_scale = raw_column_scale[active].clamp_min(1.0e-15)
    normalized = matrix / column_scale
    rhs = original_rhs
    augmented_matrices = [normalized]
    augmented_targets = [rhs]
    if float(ridge_relative) > 0.0:
        regularizer_scale = torch.sqrt(
            torch.as_tensor(
                float(ridge_relative) * float(matrix.shape[0]),
                dtype=normalized.dtype,
                device=normalized.device,
            )
        )
        reference_column_scale = torch.median(column_scale)
        physical_coefficient_weights = reference_column_scale / column_scale
        regularizer = regularizer_scale * torch.diag(
            physical_coefficient_weights
        )
        augmented_matrices.append(regularizer)
        augmented_targets.append(
            torch.zeros(
                (int(regularizer.shape[0]), int(rhs.shape[1])),
                dtype=rhs.dtype,
                device=rhs.device,
            )
        )
    augmented_matrix = torch.cat(augmented_matrices, dim=0)
    augmented_rhs = torch.cat(augmented_targets, dim=0)
    solution_normalized = torch.linalg.lstsq(
        augmented_matrix,
        augmented_rhs,
    ).solution
    active_solution = solution_normalized / column_scale[:, None]
    solution = torch.zeros(
        (int(full_matrix.shape[1]), int(original_rhs.shape[1])),
        dtype=active_solution.dtype,
        device=active_solution.device,
    )
    solution[active] = active_solution
    singular_values = torch.linalg.svdvals(matrix)
    condition = float(
        (singular_values[0] / singular_values[-1].clamp_min(1.0e-15)).detach().cpu()
    )
    residual = full_matrix @ solution - original_rhs
    return solution, {
        "rows": int(full_matrix.shape[0]),
        "columns": int(full_matrix.shape[1]),
        "active_columns": int(active.sum().detach().cpu()),
        "inactive_columns": int((~active).sum().detach().cpu()),
        "right_hand_sides": int(targets.shape[1]),
        "condition_number": condition,
        "ridge_relative": float(ridge_relative),
        "ridge_parameterization": (
            "physical_coefficient_l2_relative_to_median_design_column_energy"
        ),
        "ridge_diagonal_scale": float(
            np.sqrt(float(ridge_relative) * float(full_matrix.shape[0]))
        ),
        "ridge_weight_min": float(
            (torch.median(column_scale) / column_scale).min().detach().cpu()
        ),
        "ridge_weight_max": float(
            (torch.median(column_scale) / column_scale).max().detach().cpu()
        ),
        "relative_residual_l2": _relative_l2(
            residual.detach().cpu().numpy(),
            original_rhs.detach().cpu().numpy(),
        ),
    }


def _relative_l2(error, reference) -> float:
    error_array = np.asarray(error, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(error_array.reshape(-1))
        / max(np.linalg.norm(reference_array.reshape(-1)), 1.0e-12)
    )


def _torch_relative_l2_rows(error, reference):
    torch = _require_torch()
    error_rows = error.reshape(int(error.shape[0]), -1)
    reference_rows = reference.reshape(int(reference.shape[0]), -1)
    return torch.linalg.vector_norm(error_rows, dim=1) / torch.linalg.vector_norm(
        reference_rows,
        dim=1,
    ).clamp_min(1.0e-12)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }
