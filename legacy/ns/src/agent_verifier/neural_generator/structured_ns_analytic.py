"""Exact derivatives of declared cosine/Chebyshev or B-spline NS fields.

This evaluates the existing continuous field, including its polygon masks, by
product rules. It does not discretize the PDE, change coefficients, or use
coordinate autograd. The public adapter decides which frozen fields may use
this implementation; other field types retain the reference implementation.
"""
from __future__ import annotations

from math import comb

from .structured_ns_geometry_jets import (
    geometry_mask_derivatives,
    pressure_mask_derivatives,
)


_C3_PRESSURE_ABI = "continuous_structured_ns_mms_v2"
_HARD_PRESSURE_ABI = "continuous_structured_ns_mms_v3"
_SMOOTH_PRESSURE_ABIS = frozenset((
    "continuous_structured_ns_mms_v4", "continuous_structured_ns_mms_v5",
    "continuous_structured_ns_mms_v6",
))
_POLYGON_ABIS = _SMOOTH_PRESSURE_ABIS | {_C3_PRESSURE_ABI, _HARD_PRESSURE_ABI}


def _coordinates(field, physical_coords):
    if field.geometry_spec is None or field.continuous_mms_abi not in _POLYGON_ABIS:
        raise NotImplementedError("analytic NS derivatives require a supported polygon field")
    torch = field.torch
    # A bare field.device="cuda" resolves against the calling thread. The
    # immutable coefficients retain the actual device chosen at construction.
    coords = torch.as_tensor(physical_coords, dtype=field.dtype,
                             device=field.coefficients.psi.device)
    if coords.ndim == 3 and int(coords.shape[0]) == 1:
        coords = coords[0]
    if coords.ndim != 2 or int(coords.shape[-1]) != 3:
        raise ValueError(f"physical_coords must be [N,3], got {tuple(coords.shape)}")
    # The reference quantities method also detaches supplied coordinates; its
    # coordinate graph is an internal mechanism, not an input gradient API.
    normalized = field._normalize_coordinates(coords.detach())
    xmin, xmax, ymin, ymax = field.domain_bounds
    return normalized, (1.0 / (xmax - xmin), 1.0 / (ymax - ymin),
                        1.0 / (field.t1 - field.t0))


def _cosine_derivatives(torch, coordinate, frequencies, order, *, spatial_extension=0.0):
    omega = torch.pi * frequencies
    if spatial_extension:
        width = 1.0 + 2.0 * float(spatial_extension)
        coordinate = (coordinate + spatial_extension) / width
        omega = omega / width
    # Preserve the reference cosine argument evaluation order.
    argument = torch.pi * coordinate[:, None] * frequencies
    cosine = torch.cos(argument)
    values = [cosine]
    if order:
        sine = torch.sin(argument)
        values.append(-omega * sine)
    if order >= 2:
        values.append(-(omega * omega) * cosine)
    if order >= 3:
        values.append((omega * omega * omega) * sine)
    return values


def _temporal_basis(torch, tau, order, *, differentiate):
    z = 2.0 * tau - 1.0
    values = [torch.ones_like(z)]
    derivatives = [torch.zeros_like(z)] if differentiate else None
    if order:
        values.append(z)
        if differentiate:
            derivatives.append(torch.full_like(z, 2.0))
    for _degree in range(2, order + 1):
        if differentiate:
            derivatives.append(4.0 * values[-1] + 2.0 * z * derivatives[-1] - derivatives[-2])
        values.append(2.0 * z * values[-1] - values[-2])
    return torch.stack(values, dim=-1), (
        torch.stack(derivatives, dim=-1) if differentiate else None
    )



def _uniform_time(field, physical_coords, hint):
    """Inspect only host inputs; a True hint is a trusted Controller contract.

    CUDA values are never copied, reduced to Python scalars, or synchronized
    for this optimization. GPU callers without a prevalidated hint use the
    general coordinate path, including during CUDA graph capture.
    """
    if hint is not None:
        if not isinstance(hint, bool):
            raise TypeError("uniform_time must be a boolean or None")
        return hint
    if isinstance(physical_coords, field.torch.Tensor):
        if physical_coords.device.type != "cpu":
            return False
        values = physical_coords.detach()
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        return bool(values.shape[0] and field.torch.all(values[:, 2] == values[0, 2]))
    import numpy as np

    values = np.asarray(physical_coords)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    return bool(values.shape[0] and np.all(values[:, 2] == values[0, 2]))


def _bases(field, normalized, *, order, time_derivative, uniform_time):
    torch = field.torch
    if not field.config.is_legacy_basis:
        from .structured_ns_basis import axis_basis_derivatives
        x = axis_basis_derivatives(normalized[:, 0], config=field.config, axis="x", order=order)
        y = axis_basis_derivatives(normalized[:, 1], config=field.config, axis="y", order=order)
        times = normalized[:1, 2] if uniform_time else normalized[:, 2]
        temporal = axis_basis_derivatives(times, config=field.config, axis="t", order=int(time_derivative))
        return x, y, temporal[0], temporal[1] if time_derivative else None
    frequencies = field._spatial_frequencies
    extension = float(field.config.spatial_extension)
    x = _cosine_derivatives(torch, normalized[:, 0], frequencies[0], order, spatial_extension=extension)
    y = _cosine_derivatives(torch, normalized[:, 1], frequencies[1], order, spatial_extension=extension)
    times = normalized[:1, 2] if uniform_time else normalized[:, 2]
    temporal, dt = _temporal_basis(torch, times, int(field.config.time_order),
                                   differentiate=time_derivative)
    return x, y, temporal, dt


def _spatial_jet(torch, effective_coefficients, x, y, order):
    """Contract separable bases, reusing each y contraction across x orders."""
    count = effective_coefficients.shape[0]
    values = effective_coefficients.reshape(count, x[0].shape[1], y[0].shape[1])
    result = {}
    for dy in range(order + 1):
        if count == 1:
            # Same-time queries share a single spatial coefficient matrix.
            # Do not materialize N copies of its Kx*Ky entries.
            along_x = y[dy] @ values[0].transpose(0, 1)
        else:
            along_x = torch.einsum("nxy,ny->nx", values, y[dy])
        for dx in range(order - dy + 1):
            result[dx, dy] = (along_x * x[dx]).sum(dim=-1)
    return result


def _product_jet(left, right, order):
    result = {}
    for dx in range(order + 1):
        for dy in range(order - dx + 1):
            value = left[0, 0] * right[dx, dy]
            for ix in range(dx + 1):
                for iy in range(dy + 1):
                    if ix or iy:
                        value = value + (comb(dx, ix) * comb(dy, iy)
                                         * left[ix, iy] * right[dx - ix, dy - iy])
            result[dx, dy] = value
    return result


def _physical_jet(jet, scales):
    sx, sy, _st = scales
    return {(dx, dy): values * (sx ** dx * sy ** dy)
            for (dx, dy), values in jet.items()}


def _pressure_jet(field, normalized, core, mask_value):
    abi = field.continuous_mms_abi
    if abi == _C3_PRESSURE_ABI:
        return _product_jet(pressure_mask_derivatives(field, normalized, order=1), core, 1)
    if abi == _HARD_PRESSURE_ABI:
        # Hard comparisons have zero derivative in the reference autograd
        # implementation. Do not differentiate the solid-domain indicator.
        return {key: mask_value * value for key, value in core.items()}
    return core


def _evaluate_velocity(field, physical_coords, *, uniform_time, return_pressure):
    """Return the unchanged continuous u, v, p without coordinate autograd."""
    normalized, scales = _coordinates(field, physical_coords)
    uniform = _uniform_time(field, physical_coords, uniform_time)
    x, y, temporal, _ = _bases(field, normalized, order=1, time_derivative=False,
                              uniform_time=uniform)
    torch = field.torch
    masks, pressure_mask = geometry_mask_derivatives(field, normalized, order=1)
    core = _spatial_jet(torch, temporal @ field.coefficients.psi, x, y, 1)
    psi = _physical_jet(_product_jet(masks, core, 1), scales)
    if not return_pressure:
        return psi[0, 1], -psi[1, 0]
    pressure = _spatial_jet(torch, temporal @ field.coefficients.pressure, x, y, 0)[0, 0]
    if field.continuous_mms_abi not in _SMOOTH_PRESSURE_ABIS:
        pressure = pressure_mask * pressure
    return psi[0, 1], -psi[1, 0], pressure


def _evaluate_quantities(field, physical_coords, *, include_laplacian_components=False,
                         uniform_time=None, full=True):
    """Return the reference quantities dictionary using exact field derivatives.

    Spatial derivatives of psi extend through order three. Only the mixed
    first-space/first-time derivatives are needed for the velocity time terms.
    All derivatives below are in physical coordinates and retain float64.
    """
    normalized, scales = _coordinates(field, physical_coords)
    uniform = _uniform_time(field, physical_coords, uniform_time)
    x, y, temporal, temporal_dt = _bases(field, normalized, order=3, time_derivative=True,
                                        uniform_time=uniform)
    torch = field.torch
    masks, indicator = geometry_mask_derivatives(field, normalized, order=3)
    core = _spatial_jet(torch, temporal @ field.coefficients.psi, x, y, 3)
    psi = _physical_jet(_product_jet(masks, core, 3), scales)
    core_dt = _spatial_jet(torch, temporal_dt @ field.coefficients.psi, x, y, 1)
    psi_dt = _physical_jet(_product_jet(masks, core_dt, 1), scales)
    p_core = _spatial_jet(torch, temporal @ field.coefficients.pressure, x, y, 1)
    pressure = _physical_jet(_pressure_jet(field, normalized, p_core, indicator), scales)

    u, v = psi[0, 1], -psi[1, 0]
    dx_u, dy_u, dt_u = psi[1, 1], psi[0, 2], psi_dt[0, 1] * scales[2]
    dx_v, dy_v, dt_v = -psi[2, 0], -psi[1, 1], -psi_dt[1, 0] * scales[2]
    laplacian_u = psi[2, 1] + psi[0, 3]
    laplacian_v = -psi[3, 0] - psi[1, 2]
    dx_p, dy_p = pressure[1, 0], pressure[0, 1]
    advection_u, advection_v = u * dx_u + v * dy_u, u * dx_v + v * dy_v
    viscosity = float(field.viscosity)
    forcing_u = dt_u + advection_u + dx_p - viscosity * laplacian_u
    forcing_v = dt_v + advection_v + dy_p - viscosity * laplacian_v
    if not full:
        if field.continuous_mms_abi in _SMOOTH_PRESSURE_ABIS:
            return forcing_u * indicator, forcing_v * indicator
        return forcing_u, forcing_v
    # Preserve the reference residual arithmetic, including propagation of
    # non-finite terms; do not hide overflow by replacing residuals with zero.
    residual_u = dt_u + advection_u + dx_p - viscosity * laplacian_u - forcing_u
    residual_v = dt_v + advection_v + dy_p - viscosity * laplacian_v - forcing_v
    result = {
        "psi": psi[0, 0], "pressure": pressure[0, 0], "u": u, "v": v,
        "d_x_u": dx_u, "d_y_u": dy_u, "d_t_u": dt_u,
        "d_x_v": dx_v, "d_y_v": dy_v, "d_t_v": dt_v,
        "laplacian_u": laplacian_u, "laplacian_v": laplacian_v,
        "d_x_p": dx_p, "d_y_p": dy_p,
        "advection_u": advection_u, "advection_v": advection_v,
        "forcing_u": forcing_u, "forcing_v": forcing_v,
        "divergence": dx_u + dy_v, "residual_u": residual_u, "residual_v": residual_v,
    }
    if include_laplacian_components:
        result.update(d2_x_u=psi[2, 1], d2_y_u=psi[0, 3],
                      d2_x_v=-psi[3, 0], d2_y_v=-psi[1, 2])
    if field.continuous_mms_abi in _SMOOTH_PRESSURE_ABIS:
        for key in ("forcing_u", "forcing_v", "residual_u", "residual_v"):
            result[key] = result[key] * indicator
        result["fluid_indicator"] = indicator
    return result


def analytic_velocity(field, physical_coords, *, uniform_time=None):
    """Return only u/v, avoiding the unused pressure coefficient contraction."""
    return _evaluate_velocity(field, physical_coords, uniform_time=uniform_time,
                              return_pressure=False)


def analytic_velocity_pressure(field, physical_coords, *, uniform_time=None):
    """Return u/v/p; True uniform_time is a Controller-verified same-time hint."""
    return _evaluate_velocity(field, physical_coords, uniform_time=uniform_time,
                              return_pressure=True)


def analytic_forcing(field, physical_coords, *, uniform_time=None):
    """Return only fu/fv, omitting diagnostic quantities and zero residuals."""
    return _evaluate_quantities(field, physical_coords, uniform_time=uniform_time, full=False)


def analytic_quantities(field, physical_coords, *, include_laplacian_components=False,
                        uniform_time=None):
    """Match the reference quantities; infer same-time batches only on the CPU.

    uniform_time=True is an internal hint asserting all input times are equal;
    callers must establish it before entering a GPU-only path. False disables
    that optimization. None inspects host inputs and leaves GPU inputs general.
    """
    return _evaluate_quantities(field, physical_coords,
                                include_laplacian_components=include_laplacian_components,
                                uniform_time=uniform_time, full=True)


__all__ = ["analytic_forcing", "analytic_quantities", "analytic_velocity",
           "analytic_velocity_pressure"]
