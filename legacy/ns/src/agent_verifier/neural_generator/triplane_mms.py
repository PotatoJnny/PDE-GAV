from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.neural_generator.triplane_autoencoder import (
    CONTINUOUS_TRIPLANE_MMS_ABI_VERSION,
    canonical_geometry_tensor,
)
from agent_verifier.neural_generator.triplane_coordinate_jets import (
    COORDINATE_JET_BACKEND_ID,
    DEFAULT_COORDINATE_JET_BATCH_POINTS,
    TriplaneNSCoordinateJetBackend,
)
from agent_verifier.neural_generator.triplane_dataset import (
    GeometryContract,
    divergence as divergence_field,
    grad,
    mms_forcing_from_velocity_pressure,
    pressure_gradient,
    time_derivative,
    velocity_laplacian,
)
from agent_verifier.neural_generator.triplane_decoder import (
    TriplaneBatch,
    geometry_features_from_arrays,
    make_query_grid,
)

Array = np.ndarray

SOURCE_DERIVATION_SCHEMA_VERSION = "ns2d_continuous_exact_derivatives_v2"
LEGACY_SOURCE_DERIVATION_SCHEMA_VERSION = "ns2d_continuous_autodiff_v1"
REFERENCE_AUTOGRAD_BACKEND_ID = "torch.autograd"
DEFAULT_SOURCE_DERIVATIVE_BACKEND = COORDINATE_JET_BACKEND_ID
SOURCE_DERIVATIVE_TERMS = (
    "d_t_u",
    "d_t_v",
    "d_x_u",
    "d_y_u",
    "d_x_v",
    "d_y_v",
    "d2_x_u",
    "d2_y_u",
    "d2_x_v",
    "d2_y_v",
    "d_x_p",
    "d_y_p",
    "u_d_x_u_plus_v_d_y_u",
    "u_d_x_v_plus_v_d_y_v",
)


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for triplane MMS decoding") from exc
    return torch


@dataclass
class TriplaneDecodedFields:
    velocity: Array
    pressure: Array
    forcing: Array
    divergence: Array
    residual: Array
    psi: Array
    t: Array
    viscosity: float
    metadata: dict[str, Any]


class ContinuousTriplaneMMS:
    """Private continuous manufactured solution and its exact forcing."""

    def __init__(
        self,
        decoder,
        planes: TriplaneBatch | tuple[Any, Any, Any],
        *,
        canonical_geometry,
        domain_bounds: tuple[float, float, float, float],
        viscosity: float,
        t0: float,
        t1: float,
    ) -> None:
        torch = _require_torch()
        if getattr(decoder.config, "continuous_mms_abi", None) != CONTINUOUS_TRIPLANE_MMS_ABI_VERSION:
            raise ValueError("continuous MMS requires a continuous_triplane_mms_v1 decoder")
        if getattr(decoder.config, "plane_interpolation", None) != "quintic_bspline":
            raise ValueError("continuous MMS requires C4 quintic B-spline triplane interpolation")
        parameter = next(decoder.parameters())
        if parameter.dtype != torch.float64:
            raise ValueError("continuous MMS requires float64 autodiff evaluation")
        self.torch = torch
        self.decoder = decoder
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.planes = _planes_to(planes, device=self.device, dtype=self.dtype)
        self.canonical_geometry = torch.as_tensor(canonical_geometry, dtype=self.dtype, device=self.device)
        if tuple(self.canonical_geometry.shape[:1]) != (2,):
            raise ValueError(f"canonical geometry must be [2,X,Y], got {tuple(self.canonical_geometry.shape)}")
        self.domain_bounds = tuple(float(value) for value in domain_bounds)
        self.viscosity = float(viscosity)
        self.t0 = float(t0)
        self.t1 = float(t1)
        if self.t1 <= self.t0:
            raise ValueError("continuous MMS requires t1 > t0")

    def quantities(self, physical_coords, *, include_laplacian_components: bool = False):
        torch = self.torch
        coords = torch.as_tensor(physical_coords, dtype=self.dtype, device=self.device)
        if coords.ndim == 3 and int(coords.shape[0]) == 1:
            coords = coords[0]
        if coords.ndim != 2 or int(coords.shape[-1]) != 3:
            raise ValueError(f"physical_coords must be [N,3], got {tuple(coords.shape)}")
        coords = coords.clone().detach().requires_grad_(True)
        x_min, x_max, y_min, y_max = self.domain_bounds
        normalized = torch.stack(
            [
                (coords[:, 0] - x_min) / (x_max - x_min),
                (coords[:, 1] - y_min) / (y_max - y_min),
                (coords[:, 2] - self.t0) / (self.t1 - self.t0),
            ],
            dim=-1,
        )[None, ...]
        geometry_features = geometry_features_from_arrays(
            normalized,
            canonical_geometry=self.canonical_geometry,
            interpolation="c3_smoothstep",
        )
        psi, pressure = self.decoder.forward(self.planes, normalized, geometry_features)
        return navier_stokes_quantities_from_psi_pressure(
            psi,
            pressure,
            coords,
            viscosity=self.viscosity,
            include_laplacian_components=bool(include_laplacian_components),
        )

    def sample_grid(
        self,
        *,
        x,
        y,
        t,
        batch_points: int = DEFAULT_COORDINATE_JET_BATCH_POINTS,
        derivative_backend: str = DEFAULT_SOURCE_DERIVATIVE_BACKEND,
    ) -> TriplaneDecodedFields:
        backend = str(derivative_backend)
        if backend == COORDINATE_JET_BACKEND_ID:
            arrays, backend_metadata = TriplaneNSCoordinateJetBackend(self).materialize_cartesian(
                x=x,
                y=y,
                t=t,
                batch_points=int(batch_points),
            )
        elif backend == REFERENCE_AUTOGRAD_BACKEND_ID:
            arrays = self._sample_grid_autograd_arrays(
                x=x,
                y=y,
                t=t,
                batch_points=int(batch_points),
            )
            backend_metadata = {
                "backend_id": REFERENCE_AUTOGRAD_BACKEND_ID,
                "batch_points": int(batch_points),
            }
        else:
            raise ValueError(f"unsupported continuous source derivative backend: {backend}")
        arrays = {name: np.asarray(values, dtype=np.float64) for name, values in arrays.items()}
        velocity = np.stack([arrays["u"], arrays["v"]], axis=1)
        forcing = np.stack([arrays["forcing_u"], arrays["forcing_v"]], axis=1)
        residual = np.stack([arrays["residual_u"], arrays["residual_v"]], axis=1)
        return TriplaneDecodedFields(
            velocity=velocity,
            pressure=arrays["pressure"],
            forcing=forcing,
            divergence=arrays["divergence"],
            residual=residual,
            psi=arrays["psi"],
            t=np.asarray(t, dtype=np.float64),
            viscosity=self.viscosity,
            metadata={
                "truth_type": "continuous_triplane_streamfunction",
                "forcing_type": "continuous_exact_coordinate_derivatives_from_hidden_solution",
                "continuous_mms_abi": CONTINUOUS_TRIPLANE_MMS_ABI_VERSION,
                "time_interval": [self.t0, self.t1],
                "domain_bounds": list(self.domain_bounds),
                "canonical_state": True,
                "grid_arrays_are_materialized_views": True,
                "source_materialization": backend_metadata,
                "continuous_divergence_abs_max": float(np.max(np.abs(arrays["divergence"]))),
                "continuous_divergence_rel_max": continuous_divergence_relative_max(
                    arrays["divergence"], arrays["d_x_u"], arrays["d_y_v"]
                ),
                **continuous_mms_provenance(source_derivative_backend=backend),
            },
        )

    def _sample_grid_autograd_arrays(self, *, x, y, t, batch_points: int) -> dict[str, Array]:
        torch = self.torch
        x_values = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        y_values = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        t_values = torch.as_tensor(t, dtype=self.dtype, device=self.device)
        tt, xx, yy = torch.meshgrid(t_values, x_values, y_values, indexing="ij")
        coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
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
        )
        parts: dict[str, list[Array]] = {name: [] for name in names}
        chunk = max(1, int(batch_points))
        for start in range(0, int(coords.shape[0]), chunk):
            quantities = self.quantities(coords[start : start + chunk])
            for name in names:
                parts[name].append(quantities[name].detach().cpu().numpy())
            del quantities
        shape = (int(t_values.numel()), int(x_values.numel()), int(y_values.numel()))
        return {
            name: np.concatenate(values).reshape(shape).astype(np.float64)
            for name, values in parts.items()
        }


def navier_stokes_quantities_from_psi_pressure(
    psi,
    pressure,
    coords,
    *,
    viscosity: float,
    include_laplacian_components: bool = True,
):
    """Assemble every strong-form NS term from one physical-coordinate graph."""

    psi_values = psi.reshape(-1)
    pressure_values = pressure.reshape(-1)
    if not coords.requires_grad:
        raise ValueError("physical coordinates must require gradients")
    if psi_values.numel() != coords.shape[0] or pressure_values.numel() != coords.shape[0]:
        raise ValueError("psi, pressure, and physical coordinates must contain the same number of points")

    grad_psi = grad_scalar(psi_values, coords)
    u = grad_psi[:, 1]
    v = -grad_psi[:, 0]
    grad_u = grad_scalar(u, coords)
    grad_v = grad_scalar(v, coords)
    grad_p = grad_scalar(pressure_values, coords, create_graph=False)
    if include_laplacian_components:
        # Diagnostic reference path. The generator only needs each component's
        # exact sum, so it uses the fused streamfunction identity below.
        d2_x_u = grad_scalar(grad_u[:, 0], coords, create_graph=False)[:, 0]
        d2_y_u = grad_scalar(grad_u[:, 1], coords, create_graph=False)[:, 1]
        d2_x_v = grad_scalar(grad_v[:, 0], coords, create_graph=False)[:, 0]
        d2_y_v = grad_scalar(grad_v[:, 1], coords, create_graph=False, retain_graph=False)[:, 1]
        laplacian_u = d2_x_u + d2_y_u
        laplacian_v = d2_x_v + d2_y_v
    else:
        laplacian_psi = grad_u[:, 1] - grad_v[:, 0]
        grad_laplacian_psi = grad_scalar(
            laplacian_psi,
            coords,
            create_graph=False,
            retain_graph=False,
        )
        laplacian_u = grad_laplacian_psi[:, 1]
        laplacian_v = -grad_laplacian_psi[:, 0]
    advection_u = u * grad_u[:, 0] + v * grad_u[:, 1]
    advection_v = u * grad_v[:, 0] + v * grad_v[:, 1]
    forcing_u = grad_u[:, 2] + advection_u + grad_p[:, 0] - float(viscosity) * laplacian_u
    forcing_v = grad_v[:, 2] + advection_v + grad_p[:, 1] - float(viscosity) * laplacian_v
    residual_u = grad_u[:, 2] + advection_u + grad_p[:, 0] - float(viscosity) * laplacian_u - forcing_u
    residual_v = grad_v[:, 2] + advection_v + grad_p[:, 1] - float(viscosity) * laplacian_v - forcing_v
    quantities = {
        "psi": psi_values,
        "pressure": pressure_values,
        "u": u,
        "v": v,
        "d_x_u": grad_u[:, 0],
        "d_y_u": grad_u[:, 1],
        "d_t_u": grad_u[:, 2],
        "d_x_v": grad_v[:, 0],
        "d_y_v": grad_v[:, 1],
        "d_t_v": grad_v[:, 2],
        "laplacian_u": laplacian_u,
        "laplacian_v": laplacian_v,
        "d_x_p": grad_p[:, 0],
        "d_y_p": grad_p[:, 1],
        "advection_u": advection_u,
        "advection_v": advection_v,
        "forcing_u": forcing_u,
        "forcing_v": forcing_v,
        "divergence": grad_u[:, 0] + grad_v[:, 1],
        "residual_u": residual_u,
        "residual_v": residual_v,
    }
    if include_laplacian_components:
        quantities.update(
            {
                "d2_x_u": d2_x_u,
                "d2_y_u": d2_y_u,
                "d2_x_v": d2_x_v,
                "d2_y_v": d2_y_v,
            }
        )
    return quantities


def continuous_mms_provenance(
    *,
    source_derivative_backend: str = DEFAULT_SOURCE_DERIVATIVE_BACKEND,
    source_derivation_schema_version: str = SOURCE_DERIVATION_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "source_derivation_schema_version": str(source_derivation_schema_version),
        "source_origin": "continuous_hidden_solution",
        "derivation_method": "continuous_hidden_solution_to_public_source",
        "equation_contract_version": "incompressible_navier_stokes_2d_velocity_pressure_v1",
        "boundary_contract_version": "continuous_no_slip_c3_mask_v1",
        "geometry_contract_version": "canonical_geometry_sdf_wall_v2",
        "source_derivative_backend": str(source_derivative_backend),
        "source_derivative_terms": list(SOURCE_DERIVATIVE_TERMS),
        "finite_difference_source_operator_used": False,
        "post_hoc_projection_used": False,
        "hidden_representation": "triplane_conditioned_coordinate_network_streamfunction_pressure",
        "coordinate_convention": "physical_x_y_t",
        "spatial_interpolation_or_basis": "C4_quintic_cardinal_bspline_triplanes_plus_smooth_coordinate_mlp",
        "temporal_interpolation_or_basis": "C4_quintic_cardinal_bspline_triplanes_plus_smooth_coordinate_mlp",
        "boundary_representation": "actual_rectangular_boundary_C3_ramp_and_canonical_wall_distance_C3_interpolant",
        "geometry_representation": "C3_interpolated_canonical_obstacle_sdf_zero_level",
        "pressure_representation_or_method": "smooth_coordinate_mlp_with_C3_obstacle_mask",
        "streamfunction_required_spatial_derivative_order": 3,
        "autodiff_or_analytic_dtype": "float64",
    }


def continuous_divergence_relative_max(divergence, d_x_u, d_y_v) -> float:
    """Scale mixed-partial cancellation by the velocity-gradient magnitude."""

    divergence_abs_max = float(np.max(np.abs(np.asarray(divergence, dtype=np.float64))))
    gradient_scale = float(
        np.max(
            np.abs(np.asarray(d_x_u, dtype=np.float64))
            + np.abs(np.asarray(d_y_v, dtype=np.float64))
        )
    )
    if gradient_scale == 0.0:
        return 0.0 if divergence_abs_max == 0.0 else float("inf")
    return divergence_abs_max / gradient_scale


def project_decoded_velocity(
    decoded: TriplaneDecodedFields,
    *,
    geometry: GeometryContract,
    pressure_iterations: int = 400,
    pressure_relaxation: float = 0.8,
) -> TriplaneDecodedFields:
    from agent_verifier.neural_generator.triplane_d0 import project_velocity_to_incompressible

    if decoded.metadata.get("continuous_mms_abi") == CONTINUOUS_TRIPLANE_MMS_ABI_VERSION:
        raise ValueError("continuous MMS velocity is the canonical exact coordinate-derived solution and must not be numerically projected")
    projected_velocity, correction_pressure = project_velocity_to_incompressible(
        decoded.velocity,
        geometry,
        pressure_iterations=int(pressure_iterations),
        pressure_relaxation=float(pressure_relaxation),
    )
    forcing = mms_forcing_from_velocity_pressure(
        velocity=projected_velocity,
        pressure=decoded.pressure,
        t=decoded.t,
        viscosity=float(decoded.viscosity),
        geometry=geometry,
    )
    divergence = divergence_field(projected_velocity, geometry.x, geometry.y)
    metadata = dict(decoded.metadata)
    metadata.update(
        {
            "velocity_projection": "d0_helmholtz_jacobi_no_slip",
            "projection_pressure_iterations": int(pressure_iterations),
            "projection_pressure_relaxation": float(pressure_relaxation),
        }
    )
    return TriplaneDecodedFields(
        velocity=projected_velocity.astype(np.float32),
        pressure=decoded.pressure.astype(np.float32),
        forcing=forcing.astype(np.float32),
        divergence=divergence.astype(np.float32),
        residual=np.zeros_like(forcing, dtype=np.float32),
        psi=decoded.psi.astype(np.float32),
        t=decoded.t,
        viscosity=float(decoded.viscosity),
        metadata={**metadata, "projection_pressure_rms": float(np.sqrt(np.mean(np.square(correction_pressure))))},
    )


def decode_triplanes_to_fields(
    decoder,
    planes: TriplaneBatch | tuple[Any, Any, Any],
    *,
    geometry: GeometryContract,
    nt: int,
    viscosity: float,
    t0: float = 0.0,
    t1: float = 0.12,
    batch_points: int = DEFAULT_COORDINATE_JET_BATCH_POINTS,
) -> TriplaneDecodedFields:
    torch = _require_torch()
    if getattr(decoder.config, "continuous_mms_abi", None) == CONTINUOUS_TRIPLANE_MMS_ABI_VERSION:
        canonical = canonical_geometry_tensor(
            geometry=geometry,
            dtype=next(decoder.parameters()).dtype,
            device=next(decoder.parameters()).device,
        )
        continuous = ContinuousTriplaneMMS(
            decoder,
            planes,
            canonical_geometry=canonical,
            domain_bounds=(geometry.config.x_min, geometry.config.x_max, geometry.config.y_min, geometry.config.y_max),
            viscosity=float(viscosity),
            t0=float(t0),
            t1=float(t1),
        )
        times = np.linspace(float(t0), float(t1), int(nt), dtype=np.float64)
        decoded = continuous.sample_grid(
            x=geometry.x,
            y=geometry.y,
            t=times,
            batch_points=int(batch_points),
        )
        decoded.metadata["geometry_contract_hash"] = geometry.contract_hash
        return decoded
    if getattr(decoder.config, "field_stack_decode", False):
        return _decode_field_stack_to_fields(
            decoder,
            planes,
            geometry=geometry,
            nt=int(nt),
            viscosity=float(viscosity),
            t0=float(t0),
            t1=float(t1),
        )
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    coords_all = make_query_grid(nt=int(nt), nx=geometry.config.nx, ny=geometry.config.ny, device=str(device), dtype="float64" if dtype == torch.float64 else "float32")
    total = int(coords_all.shape[0])
    chunk = max(1, min(int(batch_points), total))
    parts: dict[str, list[Array]] = {"psi": [], "pressure": []}
    with torch.no_grad():
        for start in range(0, total, chunk):
            coords = coords_all[start : start + chunk][None, ...]
            geom = geometry_features_from_arrays(
                coords,
                signed_distance=geometry.obstacle_signed_distance,
                fluid_mask=geometry.fluid_mask,
                obstacle_mask=geometry.obstacle_mask,
                outer_boundary_mask=geometry.outer_boundary_mask,
            )
            psi, pressure = decoder.forward(planes, coords, geom)
            parts["psi"].append(psi.detach().cpu().numpy().reshape(-1))
            parts["pressure"].append(pressure.detach().cpu().numpy().reshape(-1))
    shape = (int(nt), geometry.config.nx, geometry.config.ny)

    def arr(name: str) -> Array:
        return np.concatenate(parts[name], axis=0).reshape(shape).astype(np.float64)

    t = np.linspace(float(t0), float(t1), int(nt), dtype=np.float64)
    psi = arr("psi")
    pressure = arr("pressure")
    u = grad(psi, geometry.y, axis=2)
    v = -grad(psi, geometry.x, axis=1)
    velocity = np.stack([u, v], axis=1)
    solid = ~np.asarray(geometry.fluid_mask, dtype=bool)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    velocity[:, :, solid | outer] = 0.0

    td = time_derivative(velocity, t)
    lap = velocity_laplacian(velocity, geometry.x, geometry.y)
    gradp = pressure_gradient(pressure, geometry.x, geometry.y)
    du_dx = grad(velocity[:, 0], geometry.x, axis=1)
    du_dy = grad(velocity[:, 0], geometry.y, axis=2)
    dv_dx = grad(velocity[:, 1], geometry.x, axis=1)
    dv_dy = grad(velocity[:, 1], geometry.y, axis=2)
    adv = np.empty_like(velocity, dtype=np.float64)
    adv[:, 0] = velocity[:, 0] * du_dx + velocity[:, 1] * du_dy
    adv[:, 1] = velocity[:, 0] * dv_dx + velocity[:, 1] * dv_dy
    forcing = td + adv + gradp - float(viscosity) * lap
    forcing[:, :, solid | outer] = 0.0
    div = divergence_field(velocity, geometry.x, geometry.y)
    residual = np.zeros_like(forcing, dtype=np.float64)
    return TriplaneDecodedFields(
        velocity=velocity.astype(np.float32),
        pressure=pressure.astype(np.float32),
        forcing=forcing.astype(np.float32),
        divergence=div.astype(np.float32),
        residual=residual.astype(np.float32),
        psi=psi.astype(np.float32),
        t=t,
        viscosity=float(viscosity),
        metadata={
            "truth_type": "direct_triplane_diffusion_decoded_streamfunction",
            "forcing_type": "finite_difference_manufactured_from_decoded_hidden_solution",
            "time_interval": [float(t0), float(t1)],
            "geometry_contract_hash": geometry.contract_hash,
        },
    )

def _decode_field_stack_to_fields(
    decoder,
    planes: TriplaneBatch | tuple[Any, Any, Any],
    *,
    geometry: GeometryContract,
    nt: int,
    viscosity: float,
    t0: float,
    t1: float,
) -> TriplaneDecodedFields:
    torch = _require_torch()
    p_xy = planes.p_xy if isinstance(planes, TriplaneBatch) else planes[0]
    tensor = p_xy if torch.is_tensor(p_xy) else torch.as_tensor(p_xy)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
        raise ValueError(f"field_stack decode expects one p_xy tensor with shape [1,C,X,Y] or [C,X,Y], got {tuple(tensor.shape)}")
    stack_nt = int(decoder.config.field_stack_nt)
    if int(nt) != stack_nt:
        raise ValueError(f"field_stack decode currently requires nt={stack_nt}, got {int(nt)}")
    if int(tensor.shape[1]) < 3 * stack_nt:
        raise ValueError(f"field_stack p_xy needs at least {3 * stack_nt} channels, got {int(tensor.shape[1])}")
    if tuple(tensor.shape[-2:]) != tuple(geometry.fluid_mask.shape):
        raise ValueError(f"field_stack p_xy grid {tuple(tensor.shape[-2:])} does not match geometry {tuple(geometry.fluid_mask.shape)}")
    data = tensor[0, : 3 * stack_nt].detach().cpu().numpy().astype(np.float64)
    u = data[0:stack_nt]
    v = data[stack_nt : 2 * stack_nt]
    pressure = data[2 * stack_nt : 3 * stack_nt]
    velocity = np.stack([u, v], axis=1)
    solid = ~np.asarray(geometry.fluid_mask, dtype=bool)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    velocity[:, :, solid | outer] = 0.0
    t = np.linspace(float(t0), float(t1), int(nt), dtype=np.float64)
    forcing = mms_forcing_from_velocity_pressure(
        velocity=velocity,
        pressure=pressure,
        t=t,
        viscosity=float(viscosity),
        geometry=geometry,
    )
    div = divergence_field(velocity, geometry.x, geometry.y)
    residual = np.zeros_like(forcing, dtype=np.float64)
    return TriplaneDecodedFields(
        velocity=velocity.astype(np.float32),
        pressure=pressure.astype(np.float32),
        forcing=forcing.astype(np.float32),
        divergence=div.astype(np.float32),
        residual=residual.astype(np.float32),
        psi=np.zeros((int(nt), geometry.config.nx, geometry.config.ny), dtype=np.float32),
        t=t,
        viscosity=float(viscosity),
        metadata={
            "truth_type": "direct_triplane_diffusion_decoded_field_stack",
            "forcing_type": "finite_difference_manufactured_from_decoded_hidden_solution",
            "time_interval": [float(t0), float(t1)],
            "geometry_contract_hash": geometry.contract_hash,
        },
    )


CONTINUOUS_CASE_STATE_FILENAME = "hidden_continuous_case.pt"
CONTINUOUS_CASE_MANIFEST_FILENAME = "hidden_continuous_manifest.json"

_CONTINUOUS_DECODER_CACHE: dict[tuple[str, str, str], Any] = {}
_ARTIFACT_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def write_continuous_decoder_artifact(path: str | Path, decoder) -> Path:
    torch = _require_torch()
    if getattr(decoder.config, "continuous_mms_abi", None) != CONTINUOUS_TRIPLANE_MMS_ABI_VERSION:
        raise ValueError("formal continuous decoder artifact requires continuous_triplane_mms_v1")
    if getattr(decoder.config, "plane_interpolation", None) != "quintic_bspline":
        raise ValueError("formal continuous decoder artifact requires quintic B-spline triplanes")
    if next(decoder.parameters()).dtype != torch.float64:
        raise ValueError("formal continuous decoder artifact requires float64 autodiff weights")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    config = decoder.config.to_dict()
    config["dtype"] = "float64" if next(decoder.parameters()).dtype == torch.float64 else "float32"
    state_dict = {name: value.detach().cpu() for name, value in decoder.state_dict().items()}
    torch.save(
        {
            "continuous_mms_abi": CONTINUOUS_TRIPLANE_MMS_ABI_VERSION,
            "decoder_config": config,
            "decoder_state_dict": state_dict,
        },
        target,
    )
    return target


def write_continuous_case_state(
    case_dir: str | Path,
    *,
    planes: TriplaneBatch | tuple[Any, Any, Any],
    decoder_artifact: str | Path,
    geometry: GeometryContract,
    viscosity: float,
    t0: float,
    t1: float,
) -> Path:
    torch = _require_torch()
    path = Path(case_dir)
    path.mkdir(parents=True, exist_ok=True)
    values = planes.as_tuple() if isinstance(planes, TriplaneBatch) else planes
    canonical = canonical_geometry_tensor(geometry=geometry, dtype=torch.float64, device="cpu")
    artifact_path = Path(decoder_artifact).resolve()
    artifact_sha256 = file_sha256(artifact_path)
    artifact_reference = os.path.relpath(artifact_path, start=path.resolve())
    provenance = continuous_mms_provenance()
    state_path = path / CONTINUOUS_CASE_STATE_FILENAME
    torch.save(
        {
            "continuous_mms_abi": CONTINUOUS_TRIPLANE_MMS_ABI_VERSION,
            "decoder_artifact": artifact_reference,
            "decoder_sha256": artifact_sha256,
            "planes": {
                "p_xy": values[0].detach().cpu(),
                "p_xt": values[1].detach().cpu(),
                "p_yt": values[2].detach().cpu(),
            },
            "canonical_geometry": canonical,
            "domain_bounds": [geometry.config.x_min, geometry.config.x_max, geometry.config.y_min, geometry.config.y_max],
            "time_interval": [float(t0), float(t1)],
            "viscosity": float(viscosity),
            "geometry_contract_hash": geometry.contract_hash,
            "source_provenance": provenance,
        },
        state_path,
    )
    manifest = {
        "continuous_mms_abi": CONTINUOUS_TRIPLANE_MMS_ABI_VERSION,
        "canonical_truth": CONTINUOUS_CASE_STATE_FILENAME,
        "decoder_artifact": artifact_reference,
        "decoder_sha256": artifact_sha256,
        "solution": "continuous streamfunction/pressure queried at arbitrary physical coordinates",
        "forcing": "continuous Navier-Stokes forcing derived by exact coordinate differentiation from the hidden solution",
        "sampled_grid_cache": "hidden_truth_cache.npz is a materialized view, not the truth definition",
        **provenance,
    }
    (path / CONTINUOUS_CASE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hidden_manifest_path = path / "hidden_truth_manifest.json"
    if hidden_manifest_path.exists():
        hidden_manifest = json.loads(hidden_manifest_path.read_text())
        hidden_manifest["truth_type"] = "continuous_triplane_streamfunction"
        hidden_manifest["continuous_mms_abi"] = CONTINUOUS_TRIPLANE_MMS_ABI_VERSION
        hidden_manifest["canonical_truth"] = CONTINUOUS_CASE_STATE_FILENAME
        hidden_manifest["hidden_arrays_role"] = "materialized_cache_only"
        hidden_manifest.update(provenance)
        hidden_manifest_path.write_text(json.dumps(hidden_manifest, indent=2, sort_keys=True) + "\n")
    public_question_path = path / "public_question.json"
    if public_question_path.exists():
        public_question = json.loads(public_question_path.read_text())
        public_question["case_source"] = "fixed_geometry_incompressible_flow"
        public_question["forcing_derivation"] = "provided_public_arrays_forcing"
        public_question.pop("truth_representation", None)
        public_question_path.write_text(json.dumps(public_question, indent=2, sort_keys=True) + "\n")
    return state_path


def _load_continuous_decoder(artifact_path: Path, *, device: str, expected_sha256: str):
    key = (str(artifact_path.resolve()), str(device), str(expected_sha256))
    cached = _CONTINUOUS_DECODER_CACHE.get(key)
    if cached is not None:
        return cached
    torch = _require_torch()
    artifact = torch.load(artifact_path, map_location=device, weights_only=False)
    if artifact.get("continuous_mms_abi") != CONTINUOUS_TRIPLANE_MMS_ABI_VERSION:
        raise ValueError(f"decoder artifact is not {CONTINUOUS_TRIPLANE_MMS_ABI_VERSION}: {artifact_path}")
    from agent_verifier.neural_generator.triplane_decoder import TriplaneDecoder

    decoder = TriplaneDecoder.build(artifact["decoder_config"]).to(device=device)
    decoder.load_state_dict(artifact["decoder_state_dict"])
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    _CONTINUOUS_DECODER_CACHE[key] = decoder
    return decoder


def load_continuous_case(case_dir: str | Path, *, device: str = "cpu"):
    torch = _require_torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(case_dir)
    state = torch.load(path / CONTINUOUS_CASE_STATE_FILENAME, map_location=device, weights_only=False)
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS,
        load_structured_continuous_case,
    )

    if state.get("continuous_mms_abi") in SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS:
        return load_structured_continuous_case(path, device=str(device))
    if state.get("continuous_mms_abi") != CONTINUOUS_TRIPLANE_MMS_ABI_VERSION:
        raise ValueError(f"unsupported continuous case ABI: {state.get('continuous_mms_abi')}")
    artifact_path = Path(state["decoder_artifact"])
    if not artifact_path.is_absolute():
        artifact_path = (path / artifact_path).resolve()
    expected_sha256 = str(state.get("decoder_sha256", ""))
    if not expected_sha256:
        raise ValueError("continuous case is missing decoder_sha256")
    decoder = _load_continuous_decoder(
        artifact_path,
        device=str(device),
        expected_sha256=expected_sha256,
    )
    planes = TriplaneBatch(
        p_xy=state["planes"]["p_xy"],
        p_xt=state["planes"]["p_xt"],
        p_yt=state["planes"]["p_yt"],
    )
    return ContinuousTriplaneMMS(
        decoder,
        planes,
        canonical_geometry=state["canonical_geometry"],
        domain_bounds=tuple(state["domain_bounds"]),
        viscosity=float(state["viscosity"]),
        t0=float(state["time_interval"][0]),
        t1=float(state["time_interval"][1]),
    )


def sample_continuous_case(
    case_dir: str | Path,
    *,
    x,
    y,
    t,
    batch_points: int = DEFAULT_COORDINATE_JET_BATCH_POINTS,
    device: str = "cpu",
) -> TriplaneDecodedFields:
    continuous = load_continuous_case(case_dir, device=device)
    return continuous.sample_grid(x=x, y=y, t=t, batch_points=batch_points)


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _ARTIFACT_SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _ARTIFACT_SHA256_CACHE[key] = value
    return value


def _planes_to(planes, *, device, dtype) -> TriplaneBatch:
    if isinstance(planes, TriplaneBatch):
        values = planes.as_tuple()
    else:
        values = planes
    return TriplaneBatch(
        p_xy=values[0].to(device=device, dtype=dtype),
        p_xt=values[1].to(device=device, dtype=dtype),
        p_yt=values[2].to(device=device, dtype=dtype),
    )


def grad_scalar(values, coords, *, create_graph: bool = True, retain_graph: bool = True):
    torch = _require_torch()
    return torch.autograd.grad(
        values.sum(),
        coords,
        create_graph=bool(create_graph),
        retain_graph=bool(retain_graph),
    )[0]
