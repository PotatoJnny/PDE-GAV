"""Trusted, source-only VPFP kernels for on-demand public numerical queries.

The factory validates the fixed model/condition; the runtime validates public
coordinates once. This module intentionally performs no GPU scalar validation,
finite scans, asset reads, per-query model checks, or cross-call caching.
Physics formulas retain the pinned Fourier--Hermite decoder's precision,
normalization, floor, derivative signs and coefficient ordering. Vectorized
reductions can change roundoff; independent old-kernel tests fix tolerances.
"""
from __future__ import annotations

import math
import numpy as np


def _inputs(model, points, latent, condition):
    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 5 or len(query) == 0:
        raise ValueError("basic Oracle requires nonempty [N,5] coordinates")
    torch = model.torch
    parameter = next(model.parameters())
    row = torch.as_tensor(latent, dtype=parameter.dtype, device=parameter.device).reshape(-1)
    if row.numel() != model.config.latent_dim:
        raise ValueError("basic Oracle requires one complete latent")
    scale = float(model.config.latent_input_scale)
    if scale != 1.:
        row = row * scale
    physical = torch.as_tensor(condition, dtype=parameter.dtype, device=parameter.device).reshape(-1)
    if physical.numel() != 2:
        raise ValueError("basic Oracle requires one nu/T condition")
    return query, row, physical


def _tensor(model, value, *, indices=False):
    parameter = next(model.parameters())
    return model.torch.as_tensor(value, device=parameter.device,
                                 dtype=model.torch.long if indices else parameter.dtype)


def _spatial_basis_jet(torch, model, points, *, derivatives=True):
    # Preserve the original [1,cos(k0),sin(k0),...] ordering. All Fourier
    # modes are a single broadcast, not one Python/CUDA launch per mode.
    lx, ly = map(float, model.config.equation.x_domain)
    kappa = points.new_tensor([(2. * math.pi * kx / lx, 2. * math.pi * ky / ly)
                               for kx, ky in model.local_half_modes]).reshape(-1, 2)
    phase = points[:, 0:1] * kappa[:, 0] + points[:, 1:2] * kappa[:, 1]
    cosine, sine = math.sqrt(2.) * torch.cos(phase), math.sqrt(2.) * torch.sin(phase)
    interleave = lambda left, right: torch.stack((left, right), dim=-1).flatten(1)
    ones = torch.ones_like(points[:, :1])
    spatial = torch.cat((ones, interleave(cosine, sine)), dim=1)
    if not derivatives:
        return spatial, None, None
    zeros = torch.zeros_like(ones)
    return (spatial,
            torch.cat((zeros, interleave(-kappa[:, 0] * sine, kappa[:, 0] * cosine)), dim=1),
            torch.cat((zeros, interleave(-kappa[:, 1] * sine, kappa[:, 1] * cosine)), dim=1))


def _basis(model, velocity, temperature, *, derivatives=True):
    # Only the short 1-D recurrence is sequential. The 190 2-D modes and
    # every first/second derivative are gathered and multiplied in batches.
    torch = model.torch
    modes = torch.as_tensor(model.hermite_modes, dtype=torch.long, device=velocity.device)
    maximum = max(max(pair) for pair in model.hermite_modes)
    scale = torch.sqrt(2. * temperature)
    def one_dimensional(values):
        xi = values / scale
        functions = [torch.pi ** (-.25) / torch.sqrt(scale) * torch.exp(-.5 * xi.square())]
        final_degree = maximum + 2 if derivatives else maximum
        if final_degree >= 1:
            functions.append(math.sqrt(2.) * xi * functions[0])
        for order in range(1, final_degree):
            functions.append(math.sqrt(2. / (order + 1)) * xi * functions[order]
                             - math.sqrt(order / (order + 1)) * functions[order - 1])
        table = torch.stack(functions, dim=1)
        values_out = table[:, :maximum + 1]
        if not derivatives:
            return (values_out,)
        padded = torch.cat((torch.zeros_like(table[:, :1]).expand(-1, 2), table), dim=1)
        degree = torch.arange(maximum + 1, device=velocity.device)
        # Coefficients are fixed host constants, not online numeric checks.
        left = values.new_tensor([math.sqrt(n / 2.) for n in range(maximum + 1)])
        right = values.new_tensor([math.sqrt((n + 1) / 2.) for n in range(maximum + 1)])
        first = (padded[:, degree + 1] * left - padded[:, degree + 3] * right) / scale
        second_left = values.new_tensor([math.sqrt(n * (n - 1)) if n >= 2 else 0.
                                         for n in range(maximum + 1)])
        second_right = values.new_tensor([math.sqrt((n + 1) * (n + 2))
                                          for n in range(maximum + 1)])
        second = (padded[:, degree] * second_left
                  - (2. * degree + 1.) * values_out
                  + padded[:, degree + 4] * second_right) / (2. * scale.square())
        return values_out, first, second
    x, y = one_dimensional(velocity[:, 0]), one_dimensional(velocity[:, 1])
    result = (x[0][:, modes[:, 0]] * y[0][:, modes[:, 1]],)
    if derivatives:
        result += (x[1][:, modes[:, 0]] * y[0][:, modes[:, 1]],
                   x[0][:, modes[:, 0]] * y[1][:, modes[:, 1]],
                   x[2][:, modes[:, 0]] * y[0][:, modes[:, 1]],
                   x[0][:, modes[:, 0]] * y[2][:, modes[:, 1]])
    return result


def _normalized(torch, raw):
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    result = raw / norm.clamp_min(torch.finfo(raw.dtype).tiny)
    ground = torch.zeros_like(raw)
    ground[:, 0] = 1.
    return torch.where(norm > torch.finfo(raw.dtype).eps, result, ground)


def evaluate_source_basic(model, points, latent, *, equation_condition, chunk_size=8192):
    """Evaluate only requested source pairs; return a device Tensor."""
    from .perturbation_backend import _legacy_api
    query, row, condition = _inputs(model, points, latent, equation_condition)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    torch = model.torch
    outputs = []
    with _legacy_api().ieee_float32_coordinate_matmul(model):
        for start in range(0, len(query), chunk_size):
            requested = query[start:start + chunk_size]
            spacetime, inverse = np.unique(requested[:, :3], axis=0, return_inverse=True)
            velocity, vinverse = np.unique(requested[:, 3:], axis=0, return_inverse=True)
            coordinates = _tensor(model, spacetime)
            indices, vindices = _tensor(model, inverse, indices=True), _tensor(model, vinverse, indices=True)
            unique = [_macro_jet(model, coordinates[i:i + 512], row)
                      for i in range(0, len(coordinates), 512)]
            dimensions = {"rho": 0, "rho_derivatives": 1, "electric": 0,
                          "coefficients": 0, "coefficient_derivatives": 1}
            macro = {name: torch.cat([part[name] for part in unique], dim=dim).index_select(dim, indices)
                     for name, dim in dimensions.items()}
            v = _tensor(model, velocity)
            basis = tuple(value.index_select(0, vindices) for value in _basis(model, v, condition[1]))
            outputs.append(_source_pointwise(torch, model=model,
                velocity=v.index_select(0, vindices), condition=condition.expand(len(requested), -1),
                macro=macro, basis=basis))
    return torch.cat(outputs)


def evaluate_initial_basic(model, points, latent, *, equation_condition, field_name):
    """Evaluate rho or g at trusted t0 points without unrelated fields/jets.

    The caller checks t=t0. No Cartesian completion: exact requested space and
    velocity coordinates are deduplicated before converting to model precision.
    """
    from .perturbation_backend import _legacy_api
    if field_name not in ("rho", "g"):
        raise ValueError("initial field_name must be rho or g")
    query, row, condition = _inputs(model, points, latent, equation_condition)
    torch, cfg, layout = model.torch, model.config, model.structured_layout
    outputs = []
    with _legacy_api().ieee_float32_coordinate_matmul(model):
        # Select the exact t0 time combination once, before broadcasting space.
        t0 = row.new_tensor([float(cfg.time_domain[0])])
        order = cfg.structured_rho_time_order if field_name == "rho" else cfg.structured_local_time_order
        if cfg.structured_time_basis == "piecewise_cubic_hermite_v1":
            time_weights = None  # Hermite t0: first value=1, all other weights=0.
            normalization = 1.
        else:
            time_weights, _, normalization = _time_basis_jet(torch, cfg, t0, order=order)
        rho_end = layout["rho_latent_dim"]
        if field_name == "rho":
            values = row[:rho_end].reshape(layout["rho_time_feature_count"], -1)
            raw = values[0] if time_weights is None else (time_weights @ values)[0] / normalization
            raw = raw.reshape(len(model.rho_half_modes), 2)
            coefficients = (float(cfg.rho_contrast) * float(cfg.equation.background_density)
                            * raw / (1. + torch.linalg.vector_norm(raw, dim=-1).sum()))
        else:
            local = row[rho_end + cfg.structured_kinetic_global_dim:].reshape(
                cfg.structured_local_channels, layout["local_time_feature_count"],
                layout["local_spatial_feature_count"])
            selected_time = local[:, 0, :] if time_weights is None else (
                torch.einsum("t,cts->cs", time_weights[0], local) / normalization)
        for start in range(0, len(query), 8192):
            requested = query[start:start + 8192]
            space, inverse = np.unique(requested[:, 1:3], axis=0, return_inverse=True)
            xy, indices = _tensor(model, space), _tensor(model, inverse, indices=True)
            if field_name == "rho":
                kappa = model._rho_kappa.to(dtype=xy.dtype, device=xy.device)
                phase = xy[:, :1] * kappa[:, 0] + xy[:, 1:2] * kappa[:, 1]
                rho = float(cfg.equation.background_density) + (
                    coefficients[:, 0] * torch.cos(phase) + coefficients[:, 1] * torch.sin(phase)).sum(-1)
                outputs.append(rho.index_select(0, indices))
                continue
            spatial, _, _ = _spatial_basis_jet(torch, model, xy, derivatives=False)
            raw = (float(cfg.structured_local_scale) * (spatial @ selected_time.T)
                   / math.sqrt(float(spatial.shape[1])))
            coefficients = _normalized(torch, raw).index_select(0, indices)
            velocity, vinverse = np.unique(requested[:, 3:], axis=0, return_inverse=True)
            v, vindices = _tensor(model, velocity), _tensor(model, vinverse, indices=True)
            basis = _basis(model, v, condition[1], derivatives=False)[0].index_select(0, vindices)
            amplitude = (coefficients * basis).sum(-1)
            norm = coefficients.square().sum(-1).clamp_min(torch.finfo(coefficients.dtype).tiny)
            vrows = v.index_select(0, vindices)
            maxwellian = torch.exp(-.5 * vrows.square().sum(-1) / condition[1]) / (2. * torch.pi * condition[1])
            floor = float(cfg.velocity_floor_mixture)
            outputs.append((1. - floor) * amplitude.square() / norm + floor * maxwellian)
    return torch.cat(outputs)


def _time_basis_jet(torch, config, times, *, order: int):
    t0, t1 = (float(value) for value in config.time_domain)
    duration = t1 - t0
    tau = (times - t0) / duration
    tau = tau.clamp(0.0, 1.0)
    if config.structured_time_basis == "global_chebyshev_v1":
        scaled = 2.0 * tau
        scaled = scaled - 1.0
        values = [torch.ones_like(scaled)]
        derivatives = [torch.zeros_like(scaled)]
        if order >= 1:
            values.append(scaled)
            derivatives.append(torch.full_like(scaled, 2.0 / duration))
        for _degree in range(1, order):
            values.append(2.0 * scaled * values[-1] - values[-2])
            derivatives.append(
                (4.0 / duration) * values[-2]
                + 2.0 * scaled * derivatives[-1]
                - derivatives[-2]
            )
        basis = torch.stack(values, dim=-1)
        derivative = torch.stack(derivatives, dim=-1)
        return basis, derivative, math.sqrt(float(order + 1))

    count = int(config.structured_time_control_count)
    unit = tau * float(count - 1)
    lower = torch.floor(unit).to(dtype=torch.long).clamp(max=count - 2)
    local = unit - lower.to(dtype=unit.dtype)
    local2 = local.square()
    local3 = local2 * local
    h00 = 2.0 * local3 - 3.0 * local2 + 1.0
    h10 = local3 - 2.0 * local2 + local
    h01 = -2.0 * local3 + 3.0 * local2
    h11 = local3 - local2
    dh00 = 6.0 * local2 - 6.0 * local
    dh10 = 3.0 * local2 - 4.0 * local + 1.0
    dh01 = -6.0 * local2 + 6.0 * local
    dh11 = 3.0 * local2 - 2.0 * local
    interval = 1.0 / float(count - 1)
    local_to_time = float(count - 1) / duration
    rows = torch.arange(int(times.shape[0]), device=times.device)
    basis = torch.zeros((int(times.shape[0]), 2 * count), dtype=times.dtype, device=times.device)
    derivative = torch.zeros_like(basis)
    basis[rows, lower] = h00
    basis[rows, lower + 1] = h01
    basis[rows, count + lower] = interval * h10
    basis[rows, count + lower + 1] = interval * h11
    derivative[rows, lower] = local_to_time * dh00
    derivative[rows, lower + 1] = local_to_time * dh01
    derivative[rows, count + lower] = local_to_time * interval * dh10
    derivative[rows, count + lower + 1] = local_to_time * interval * dh11
    return basis, derivative, 1.0


def _macro_jet(model, spacetime_points, latent):
    """Return rho, Poisson field, coefficients, and their t/x/y jets."""

    torch = model.torch
    parameter = next(model.parameters())
    points = torch.as_tensor(
        spacetime_points, dtype=parameter.dtype, device=parameter.device
    )
    if points.ndim != 2 or int(points.shape[1]) != 3:
        raise ValueError("spacetime_points must have shape [Q,3]")
    latent_row = torch.as_tensor(
        latent, dtype=parameter.dtype, device=parameter.device
    )
    if latent_row.ndim == 2 and int(latent_row.shape[0]) == 1:
        latent_row = latent_row[0]
    if latent_row.ndim != 1 or int(latent_row.numel()) != int(
        model.config.latent_dim
    ):
        raise ValueError("analytic macro jet requires one full decoder latent")
    layout = model.structured_layout
    if layout is None:
        raise RuntimeError("structured latent layout is unavailable")

    rho_time_basis, rho_time_derivative, rho_time_normalizer = _time_basis_jet(
        torch,
        model.config,
        points[:, 0],
        order=int(model.config.structured_rho_time_order),
    )
    rho_end = int(layout["rho_latent_dim"])
    rho_values = latent_row[:rho_end].reshape(
        int(layout["rho_time_feature_count"]),
        2 * len(model.rho_half_modes),
    )
    raw_rho = (rho_time_basis @ rho_values) / float(rho_time_normalizer)
    raw_rho_t = (rho_time_derivative @ rho_values) / float(
        rho_time_normalizer
    )
    rho_data = _rho_coordinate_jet(model, points, raw_rho, raw_rho_t)

    local_start = rho_end + int(model.config.structured_kinetic_global_dim)
    local_values = latent_row[local_start:].reshape(
        int(model.config.structured_local_channels),
        int(layout["local_time_feature_count"]),
        int(layout["local_spatial_feature_count"]),
    )
    if (model.config.structured_time_basis == "piecewise_cubic_hermite_v1"
            or model.config.structured_local_time_order == model.config.structured_rho_time_order):
        local_time_basis, local_time_derivative, local_time_normalizer = (
            rho_time_basis, rho_time_derivative, rho_time_normalizer)
    else:
        local_time_basis, local_time_derivative, local_time_normalizer = _time_basis_jet(
            torch, model.config, points[:, 0], order=model.config.structured_local_time_order)
    spatial, spatial_x, spatial_y = _spatial_basis_jet(
        torch, model, points[:, 1:3]
    )
    local_scale = (
        float(model.config.structured_local_scale)
        / math.sqrt(float(spatial.shape[1]))
        / float(local_time_normalizer)
    )
    raw_coefficients = local_scale * torch.einsum(
        "qt,cts,qs->qc", local_time_basis, local_values, spatial
    )
    raw_derivatives = torch.stack(
        (
            local_scale
            * torch.einsum(
                "qt,cts,qs->qc", local_time_derivative, local_values, spatial
            ),
            local_scale
            * torch.einsum(
                "qt,cts,qs->qc", local_time_basis, local_values, spatial_x
            ),
            local_scale
            * torch.einsum(
                "qt,cts,qs->qc", local_time_basis, local_values, spatial_y
            ),
        ),
        dim=0,
    )
    norm = torch.linalg.vector_norm(
        raw_coefficients, dim=-1, keepdim=True
    )
    tiny = torch.finfo(raw_coefficients.dtype).tiny
    safe_norm = norm.clamp_min(tiny)
    normalized = raw_coefficients / safe_norm
    ground = torch.zeros_like(normalized)
    ground[:, 0] = 1.0
    valid = norm > torch.finfo(raw_coefficients.dtype).eps
    normalized = torch.where(valid, normalized, ground)
    projection = torch.sum(
        normalized[None] * raw_derivatives, dim=-1, keepdim=True
    )
    normalized_derivatives = (
        raw_derivatives - normalized[None] * projection
    ) / safe_norm[None]
    normalized_derivatives = torch.where(
        valid[None], normalized_derivatives, torch.zeros_like(normalized_derivatives)
    )
    return {
        **rho_data,
        "coefficients": normalized,
        "coefficient_derivatives": normalized_derivatives,
    }


def _rho_coordinate_jet(model, points, raw_values, raw_time_derivative):
    torch = model.torch
    mode_count = len(model.rho_half_modes)
    raw = raw_values.reshape(int(points.shape[0]), mode_count, 2)
    raw_t = raw_time_derivative.reshape_as(raw)
    norms = torch.linalg.vector_norm(raw, dim=-1)
    safe_norms = norms.clamp_min(torch.finfo(raw.dtype).tiny)
    norm_t = torch.sum(raw * raw_t, dim=-1) / safe_norms
    # At the answer-first t=0 anchor both the new path and its derivative vanish.
    norm_t = torch.where(norms > 0.0, norm_t, torch.zeros_like(norm_t))
    denominator = 1.0 + torch.sum(norms, dim=-1, keepdim=True)
    denominator_t = torch.sum(norm_t, dim=-1, keepdim=True)
    scale = float(model.config.rho_contrast) * float(
        model.config.equation.background_density
    )
    coefficients = scale * raw / denominator[:, :, None]
    coefficients_t = scale * (
        raw_t / denominator[:, :, None]
        - raw
        * denominator_t[:, :, None]
        / denominator[:, :, None].square()
    )
    kappa = model._rho_kappa.to(dtype=points.dtype, device=points.device)
    phase = points[:, 1:2] * kappa[None, :, 0] + points[:, 2:3] * kappa[None, :, 1]
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    mode_values = coefficients[:, :, 0] * cosine + coefficients[:, :, 1] * sine
    background = float(model.config.equation.background_density)
    rho = background + torch.sum(mode_values, dim=-1)
    rho_t = torch.sum(
        coefficients_t[:, :, 0] * cosine + coefficients_t[:, :, 1] * sine,
        dim=-1,
    )
    rho_x = torch.sum(
        -coefficients[:, :, 0] * kappa[None, :, 0] * sine
        + coefficients[:, :, 1] * kappa[None, :, 0] * cosine,
        dim=-1,
    )
    rho_y = torch.sum(
        -coefficients[:, :, 0] * kappa[None, :, 1] * sine
        + coefficients[:, :, 1] * kappa[None, :, 1] * cosine,
        dim=-1,
    )
    kappa_squared = torch.sum(kappa.square(), dim=-1)
    electric_x = torch.sum(
        (
            coefficients[:, :, 0] * kappa[None, :, 0] * sine
            - coefficients[:, :, 1] * kappa[None, :, 0] * cosine
        )
        / kappa_squared[None, :],
        dim=-1,
    )
    electric_y = torch.sum(
        (
            coefficients[:, :, 0] * kappa[None, :, 1] * sine
            - coefficients[:, :, 1] * kappa[None, :, 1] * cosine
        )
        / kappa_squared[None, :],
        dim=-1,
    )
    return {
        "rho": rho,
        "rho_derivatives": torch.stack((rho_t, rho_x, rho_y), dim=0),
        "electric": torch.stack((electric_x, electric_y), dim=-1),
    }


def _source_pointwise(torch, *, model, velocity, condition, macro, basis):
    basis_value, basis_vx, basis_vy, basis_vx2, basis_vy2 = basis
    coefficients = macro["coefficients"]
    coefficient_d = macro["coefficient_derivatives"]
    amplitude = torch.sum(coefficients * basis_value, dim=-1)
    amplitude_d = torch.einsum("dqm,qm->dq", coefficient_d, basis_value)
    amplitude_vx = torch.sum(coefficients * basis_vx, dim=-1)
    amplitude_vy = torch.sum(coefficients * basis_vy, dim=-1)
    amplitude_vx2 = torch.sum(coefficients * basis_vx2, dim=-1)
    amplitude_vy2 = torch.sum(coefficients * basis_vy2, dim=-1)
    return _source_from_amplitudes(
        torch,
        model=model,
        velocity_x=velocity[:, 0],
        velocity_y=velocity[:, 1],
        condition=condition,
        macro=macro,
        amplitude=amplitude,
        amplitude_d=amplitude_d,
        amplitude_vx=amplitude_vx,
        amplitude_vy=amplitude_vy,
        amplitude_vx2=amplitude_vx2,
        amplitude_vy2=amplitude_vy2,
        cartesian=False,
    )


def _source_from_amplitudes(
    torch,
    *,
    model,
    velocity_x,
    velocity_y,
    condition,
    macro,
    amplitude,
    amplitude_d,
    amplitude_vx,
    amplitude_vy,
    amplitude_vx2,
    amplitude_vy2,
    cartesian: bool,
):
    coefficients = macro["coefficients"]
    coefficient_d = macro["coefficient_derivatives"]
    norm_squared = torch.sum(coefficients.square(), dim=-1).clamp_min(
        torch.finfo(coefficients.dtype).tiny
    )
    norm_squared_d = 2.0 * torch.sum(
        coefficients[None] * coefficient_d, dim=-1
    )
    rho = macro["rho"]
    rho_d = macro["rho_derivatives"]
    electric = macro["electric"]
    if cartesian:
        norm_squared = norm_squared[:, None]
        norm_squared_d = norm_squared_d[:, :, None]
        rho = rho[:, None]
        rho_d = rho_d[:, :, None]
        electric_x = electric[:, 0:1]
        electric_y = electric[:, 1:2]
        collision_nu = condition[0]
        theta = condition[1]
    else:
        electric_x = electric[:, 0]
        electric_y = electric[:, 1]
        collision_nu = condition[:, 0]
        theta = condition[:, 1]

    amplitude_squared = amplitude.square()
    kinetic_density = amplitude_squared / norm_squared
    kinetic_density_d = (
        2.0 * amplitude[None] * amplitude_d / norm_squared[None]
        - amplitude_squared[None]
        * norm_squared_d
        / norm_squared.square()[None]
    )
    kinetic_density_vx = (
        2.0 * amplitude * amplitude_vx / norm_squared
    )
    kinetic_density_vy = (
        2.0 * amplitude * amplitude_vy / norm_squared
    )
    kinetic_density_vx2 = (
        2.0
        * (amplitude_vx.square() + amplitude * amplitude_vx2)
        / norm_squared
    )
    kinetic_density_vy2 = (
        2.0
        * (amplitude_vy.square() + amplitude * amplitude_vy2)
        / norm_squared
    )

    velocity_squared = velocity_x.square() + velocity_y.square()
    maxwellian = torch.exp(-0.5 * velocity_squared / theta) / (
        2.0 * torch.pi * theta
    )
    maxwellian_vx = -velocity_x * maxwellian / theta
    maxwellian_vy = -velocity_y * maxwellian / theta
    maxwellian_vx2 = (
        velocity_x.square() / theta.square() - 1.0 / theta
    ) * maxwellian
    maxwellian_vy2 = (
        velocity_y.square() / theta.square() - 1.0 / theta
    ) * maxwellian
    floor = float(model.config.velocity_floor_mixture)
    kinetic_weight = 1.0 - floor
    g = kinetic_weight * kinetic_density + floor * maxwellian
    g_d = kinetic_weight * kinetic_density_d
    g_vx = kinetic_weight * kinetic_density_vx + floor * maxwellian_vx
    g_vy = kinetic_weight * kinetic_density_vy + floor * maxwellian_vy
    g_vx2 = kinetic_weight * kinetic_density_vx2 + floor * maxwellian_vx2
    g_vy2 = kinetic_weight * kinetic_density_vy2 + floor * maxwellian_vy2

    f = rho * g
    f_d = rho_d * g[None] + rho[None] * g_d
    f_vx = rho * g_vx
    f_vy = rho * g_vy
    f_vx2 = rho * g_vx2
    f_vy2 = rho * g_vy2
    f_velocity_laplacian = f_vx2 + f_vy2
    collision = (
        2.0 * f
        + velocity_x * f_vx
        + velocity_y * f_vy
        + theta * f_velocity_laplacian
    )
    source = (
        f_d[0]
        + velocity_x * f_d[1]
        + velocity_y * f_d[2]
        + electric_x * f_vx
        + electric_y * f_vy
        - collision_nu * collision
    )
    return source
