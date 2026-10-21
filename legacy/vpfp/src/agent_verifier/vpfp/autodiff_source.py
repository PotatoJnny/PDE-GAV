from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.basis import VPFPBasisConfig, velocity_basis
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
from agent_verifier.vpfp.triplane import TriplaneBatch, plane_tuple

AUTODIFF_SOURCE_ORIGIN = "generated_continuous_autodiff_residual"


@dataclass(frozen=True)
class VPFPAutodiffSourceResult:
    rho: np.ndarray
    g: np.ndarray
    basis_coeffs: np.ndarray
    source: np.ndarray
    phi: np.ndarray
    electric: np.ndarray
    diagnostics: dict[str, Any]


def compute_autodiff_source_from_triplanes(
    autoencoder,
    planes: TriplaneBatch | tuple[Any, Any, Any],
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    basis_config: VPFPBasisConfig | dict[str, Any] | None,
    collision_nu: float,
) -> VPFPAutodiffSourceResult:
    """Manufacture a continuous VPFP case using automatic differentiation.

    The hidden fields are queried from a continuous triplane decoder. Every
    derivative in the VPFP source is obtained with torch.autograd: t/x/y
    derivatives of rho and the basis coefficients, first and second velocity
    derivatives of the fixed atoms, and the spatial gradient of the Fourier
    Poisson potential. No repository finite-difference source operator is used.
    """

    torch = _torch()
    grid.validate()
    time.validate()
    if not hasattr(autoencoder, "decode_points"):
        raise TypeError("autodiff source construction requires the torch VPFP autoencoder")
    parameter = next(autoencoder.parameters())
    work_planes = _coerce_single_case_planes(
        torch,
        planes,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    _validate_native_plane_shapes(work_planes, grid=grid, time=time)
    config = basis_config if isinstance(basis_config, VPFPBasisConfig) else VPFPBasisConfig.from_dict(basis_config)

    points, time_indices = _spacetime_points(
        torch,
        grid=grid,
        time=time,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    features = autoencoder.decode_points(
        work_planes,
        points,
        x_domain=grid.x_domain,
        time_bounds=(time.t0, time.t1),
    )[0]
    (
        atoms,
        atom_dvx,
        atom_dvy,
        atom_laplacian,
        velocity_points,
        velocity_interpolation_rel_l2,
    ) = _velocity_atom_derivatives(
        torch,
        grid=grid,
        config=config,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    atom_count = int(atoms.shape[0])
    expected_channels = 1 + atom_count
    if int(features.shape[0]) != expected_channels:
        raise ValueError(
            f"autoencoder decoded {int(features.shape[0])} channels, "
            f"but the configured velocity basis requires {expected_channels}"
        )

    rho = torch.exp(torch.clamp(features[0], -20.0, 20.0))
    coefficients = torch.softmax(features[1:].transpose(0, 1), dim=-1)
    rho_gradient = _pointwise_coordinate_gradient(torch, rho, points)
    coefficient_gradients = torch.stack(
        [
            _pointwise_coordinate_gradient(torch, coefficients[:, atom], points)
            for atom in range(atom_count)
        ],
        dim=1,
    )

    rho_grid = rho.reshape(time.nt, grid.nx, grid.ny)
    phi_grid, electric, poisson_match = _poisson_field_with_autodiff(
        torch,
        rho_grid=rho_grid,
        points=points,
        time_indices=time_indices,
        grid=grid,
    )

    g_flat = coefficients @ atoms
    dg_dt = coefficient_gradients[:, :, 0] @ atoms
    dg_dx = coefficient_gradients[:, :, 1] @ atoms
    dg_dy = coefficient_gradients[:, :, 2] @ atoms
    dg_dvx = coefficients @ atom_dvx
    dg_dvy = coefficients @ atom_dvy
    laplacian_g = coefficients @ atom_laplacian
    df_dt = rho_gradient[:, 0, None] * g_flat + rho[:, None] * dg_dt
    df_dx = rho_gradient[:, 1, None] * g_flat + rho[:, None] * dg_dx
    df_dy = rho_gradient[:, 2, None] * g_flat + rho[:, None] * dg_dy

    electric_flat = electric.permute(0, 2, 3, 1).reshape(-1, 2)
    source_flat = (
        df_dt
        + velocity_points[:, 0][None, :] * df_dx
        + velocity_points[:, 1][None, :] * df_dy
        + rho[:, None]
        * (
            electric_flat[:, 0, None] * dg_dvx
            + electric_flat[:, 1, None] * dg_dvy
        )
        - float(collision_nu) * rho[:, None] * laplacian_g
    )

    g = g_flat.reshape(time.nt, grid.nx, grid.ny, grid.nvx, grid.nvy)
    source = source_flat.reshape(time.nt, grid.nx, grid.ny, grid.nvx, grid.nvy)
    coefficient_grid = coefficients.reshape(time.nt, grid.nx, grid.ny, atom_count)
    with torch.no_grad():
        native = autoencoder.decode_planes(
            work_planes,
            nt=time.nt,
            nx=grid.nx,
            ny=grid.ny,
        )[0].reshape(expected_channels, -1)
        native_decode_rel_l2 = _relative_l2(torch, features.detach() - native, native)
        g_mass = torch.sum(g, dim=(-2, -1)) * float(grid.velocity_cell_area)

    diagnostics = {
        "source_origin": AUTODIFF_SOURCE_ORIGIN,
        "source_derivative_backend": "torch_autograd_continuous_triplane_v1",
        "source_derivative_terms": [
            "d_t_f",
            "d_x_f",
            "d_y_f",
            "d_vx_g",
            "d_vy_g",
            "d2_vx_g",
            "d2_vy_g",
            "grad_x_phi",
        ],
        "triplane_interpolation": "periodic_space_nonperiodic_time_bicubic_align_corners",
        "poisson_reconstruction": "fft_coefficients_continuous_fourier_query_then_autograd",
        "velocity_derivative_method": "autograd_of_neumann_cosine_interpolated_velocity_atoms",
        "velocity_boundary_continuation": "positive_log_cosine_interpolant_with_exact_neumann_boundary",
        "finite_difference_source_operator_used": False,
        "autodiff_dtype": str(parameter.dtype).removeprefix("torch."),
        "velocity_native_grid_interpolation_rel_l2": velocity_interpolation_rel_l2,
        "native_grid_decode_rel_l2": native_decode_rel_l2,
        "poisson_autodiff_vs_spectral_grid_rel_l2": poisson_match,
        "g_normalization_max_abs_error": float(torch.max(torch.abs(g_mass - 1.0)).detach().cpu()),
        "rho_min": float(torch.min(rho).detach().cpu()),
        "g_min": float(torch.min(g).detach().cpu()),
        "source_rms": float(torch.sqrt(torch.mean(source * source)).detach().cpu()),
    }
    arrays = (rho_grid, g, coefficient_grid, source, phi_grid, electric)
    if not all(bool(torch.all(torch.isfinite(value))) for value in arrays):
        raise FloatingPointError("autodiff VPFP source construction produced nonfinite arrays")
    return VPFPAutodiffSourceResult(
        rho=_numpy(rho_grid),
        g=_numpy(g),
        basis_coeffs=_numpy(coefficient_grid),
        source=_numpy(source),
        phi=_numpy(phi_grid),
        electric=_numpy(electric),
        diagnostics=diagnostics,
    )


def _coerce_single_case_planes(torch, planes, *, device, dtype) -> TriplaneBatch:
    values = []
    for value in plane_tuple(planes):
        tensor = torch.as_tensor(value, device=device, dtype=dtype).detach()
        if tensor.ndim == 3:
            tensor = tensor[None, ...]
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise ValueError(f"autodiff source expects one triplane case, got {tuple(tensor.shape)}")
        values.append(tensor)
    return TriplaneBatch(*values)


def _validate_native_plane_shapes(planes: TriplaneBatch, *, grid: VPFPGrid, time: VPFPTime) -> None:
    actual = (
        tuple(int(value) for value in planes.p_xy.shape[-2:]),
        tuple(int(value) for value in planes.p_xt.shape[-2:]),
        tuple(int(value) for value in planes.p_yt.shape[-2:]),
    )
    expected = ((grid.nx, grid.ny), (grid.nx, time.nt), (grid.ny, time.nt))
    if actual != expected:
        raise ValueError(f"autodiff source requires native triplane shapes {expected}, got {actual}")


def _spacetime_points(torch, *, grid: VPFPGrid, time: VPFPTime, device, dtype):
    t = torch.linspace(float(time.t0), float(time.t1), time.nt, device=device, dtype=dtype)
    x = torch.arange(grid.nx, device=device, dtype=dtype) * float(grid.dx)
    y = torch.arange(grid.ny, device=device, dtype=dtype) * float(grid.dy)
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="ij")
    points = torch.stack((tt, xx, yy), dim=-1).reshape(1, -1, 3).detach().requires_grad_(True)
    time_indices = torch.arange(time.nt, device=device).view(-1, 1, 1).expand(time.nt, grid.nx, grid.ny).reshape(-1)
    return points, time_indices


def _pointwise_coordinate_gradient(torch, values, points):
    # Continuous point queries are independent, so grad(sum(values)) returns
    # each value's derivative with respect to its own query coordinate.
    return torch.autograd.grad(
        values.sum(),
        points,
        create_graph=False,
        retain_graph=True,
        allow_unused=False,
    )[0][0]


def _velocity_atom_derivatives(torch, *, grid: VPFPGrid, config: VPFPBasisConfig, device, dtype):
    vx_axis = torch.linspace(grid.v_bounds[0], grid.v_bounds[1], grid.nvx, device=device, dtype=dtype)
    vy_axis = torch.linspace(grid.v_bounds[2], grid.v_bounds[3], grid.nvy, device=device, dtype=dtype)
    vx, vy = torch.meshgrid(vx_axis, vy_axis, indexing="ij")
    points = torch.stack((vx, vy), dim=-1).reshape(-1, 2).detach().requires_grad_(True)
    atom_values = torch.as_tensor(
        velocity_basis(grid, config)[0],
        device=device,
        dtype=dtype,
    )
    interpolated_atoms = _neumann_cosine_interpolate_velocity_atoms(
        torch,
        atom_values,
        points,
        grid=grid,
    )
    interpolation_rel_l2 = _relative_l2(
        torch,
        interpolated_atoms.detach() - atom_values.reshape(atom_values.shape[0], -1),
        atom_values,
    )
    first_x = []
    first_y = []
    laplacians = []
    for atom in interpolated_atoms:
        first = torch.autograd.grad(
            atom.sum(),
            points,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        second_x = torch.autograd.grad(
            first[:, 0].sum(),
            points,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )[0][:, 0]
        second_y = torch.autograd.grad(
            first[:, 1].sum(),
            points,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )[0][:, 1]
        first_x.append(first[:, 0].detach())
        first_y.append(first[:, 1].detach())
        laplacians.append((second_x + second_y).detach())
    return (
        atom_values.reshape(atom_values.shape[0], -1),
        torch.stack(first_x),
        torch.stack(first_y),
        torch.stack(laplacians),
        points.detach(),
        interpolation_rel_l2,
    )


def _neumann_cosine_interpolate_velocity_atoms(torch, atom_values, points, *, grid: VPFPGrid):
    """Interpolate nodal atoms with a tensor cosine basis.

    Interpolating log(B) is exact at the native equally spaced velocity nodes
    and keeps B positive between them. Cosine modes have zero normal derivative
    at both ends, so autograd derivatives satisfy the Neumann boundary.
    """

    atom_count = int(atom_values.shape[0])
    mode_x = torch.arange(grid.nvx, device=points.device, dtype=points.dtype)
    mode_y = torch.arange(grid.nvy, device=points.device, dtype=points.dtype)
    node_x = torch.arange(grid.nvx, device=points.device, dtype=points.dtype)
    node_y = torch.arange(grid.nvy, device=points.device, dtype=points.dtype)
    cosine_x = torch.cos(np.pi * node_x[:, None] * mode_x[None, :] / float(grid.nvx - 1))
    cosine_y = torch.cos(np.pi * node_y[:, None] * mode_y[None, :] / float(grid.nvy - 1))
    log_atom_values = torch.log(torch.clamp_min(atom_values, torch.finfo(atom_values.dtype).tiny))
    coefficients_x = torch.linalg.solve(
        cosine_x.expand(atom_count, -1, -1),
        log_atom_values,
    )
    coefficients = torch.linalg.solve(
        cosine_y.expand(atom_count, -1, -1),
        coefficients_x.transpose(-1, -2),
    ).transpose(-1, -2)

    vx0, vx1, vy0, vy1 = (float(value) for value in grid.v_bounds)
    fraction_x = (points[:, 0] - vx0) / (vx1 - vx0)
    fraction_y = (points[:, 1] - vy0) / (vy1 - vy0)
    query_x = torch.cos(np.pi * fraction_x[:, None] * mode_x[None, :])
    query_y = torch.cos(np.pi * fraction_y[:, None] * mode_y[None, :])
    log_query = torch.einsum("pm,kmn,pn->kp", query_x, coefficients, query_y)
    return torch.exp(log_query)


def _poisson_field_with_autodiff(torch, *, rho_grid, points, time_indices, grid: VPFPGrid):
    centered = rho_grid - rho_grid.mean(dim=(-2, -1), keepdim=True)
    rho_hat = torch.fft.fftn(centered, dim=(-2, -1))
    kx = 2.0 * np.pi * torch.fft.fftfreq(
        grid.nx,
        d=float(grid.dx),
        device=rho_grid.device,
        dtype=rho_grid.dtype,
    )
    ky = 2.0 * np.pi * torch.fft.fftfreq(
        grid.ny,
        d=float(grid.dy),
        device=rho_grid.device,
        dtype=rho_grid.dtype,
    )
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    phi_hat = torch.where(k2 > 0.0, rho_hat / torch.where(k2 > 0.0, k2, torch.ones_like(k2)), torch.zeros_like(rho_hat))
    phi_grid = torch.fft.ifftn(phi_hat, dim=(-2, -1)).real

    query = points[0]
    phase = query[:, 1, None, None] * kx[None, :, None] + query[:, 2, None, None] * ky[None, None, :]
    selected_coefficients = phi_hat.detach().index_select(0, time_indices)
    phi_at_points = (
        selected_coefficients * torch.exp(1j * phase)
    ).sum(dim=(-2, -1)).real / float(grid.nx * grid.ny)
    phi_gradient = torch.autograd.grad(
        phi_at_points.sum(),
        points,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0][0]
    electric = torch.stack((-phi_gradient[:, 1], -phi_gradient[:, 2]), dim=-1)
    electric = electric.reshape(rho_grid.shape[0], grid.nx, grid.ny, 2).permute(0, 3, 1, 2)

    spectral = torch.stack(
        (
            torch.fft.ifftn(-1j * kx[None, :, None] * phi_hat, dim=(-2, -1)).real,
            torch.fft.ifftn(-1j * ky[None, None, :] * phi_hat, dim=(-2, -1)).real,
        ),
        dim=1,
    )
    return phi_grid, electric, _relative_l2(torch, electric - spectral, spectral)


def _relative_l2(torch, numerator, denominator) -> float:
    value = torch.linalg.vector_norm(numerator) / (torch.linalg.vector_norm(denominator) + 1.0e-12)
    return float(value.detach().cpu())


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(float, copy=False)


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for VPFP autodiff source construction") from exc
    return torch
