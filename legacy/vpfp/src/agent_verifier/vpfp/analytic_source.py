from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Sequence

ANALYTIC_SOURCE_BACKEND = "structured_fourier_hermite_jet_v1"
AUTODIFF_SOURCE_BACKEND = "torch_nested_coordinate_autodiff_v1"
IEEE_COORDINATE_MATMUL_PRECISION = "ieee_float32"


@contextmanager
def ieee_float32_coordinate_matmul(model):
    """Use IEEE FP32 for scientific coordinate evaluation on CUDA.

    Training and diffusion may keep TF32 enabled. Coordinate jets are much
    smaller and must not change with pointwise versus Cartesian batch shape.
    """

    torch = model.torch
    parameter = next(model.parameters())
    if parameter.device.type != "cuda" or parameter.dtype != torch.float32:
        yield
        return
    backend = torch.backends.cuda.matmul
    try:
        previous = backend.fp32_precision
    except AttributeError:
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous)
        return
    backend.fp32_precision = "ieee"
    try:
        yield
    finally:
        backend.fp32_precision = previous


def require_analytic_structured_source(model, *, micro_latent=None) -> None:
    """Fail closed unless the decoder has the exact supported factorization."""

    config = model.config
    failures = []
    if getattr(model, "structured_layout", None) is None:
        failures.append("structured_spectral_latent")
    if not bool(getattr(config, "structured_direct_coefficients", False)):
        failures.append("structured_direct_coefficients")
    if float(getattr(config, "structured_residual_scale", 1.0)) != 0.0:
        failures.append("zero_structured_residual_scale")
    if str(getattr(config, "rho_positivity_gauge", "")) != (
        "phase_invariant_group_l1_v2"
    ):
        failures.append("phase_invariant_group_l1_v2")
    if str(getattr(config, "structured_time_basis", "")) not in {
        "global_chebyshev_v1",
        "piecewise_cubic_hermite_v1",
    }:
        failures.append("supported_structured_time_basis")
    if micro_latent is not None:
        failures.append("no_micro_latent")
    if failures:
        raise ValueError(
            "analytic VPFP source backend does not support this decoder: "
            + ", ".join(failures)
        )


def resolve_structured_source_backend(
    model,
    *,
    micro_latent=None,
    requested: str = "auto",
) -> str:
    """Resolve a coordinate backend while keeping fallback explicit."""

    mode = str(requested)
    if mode not in {"auto", "analytic", "autodiff"}:
        raise ValueError(
            "source derivative backend must be auto, analytic, or autodiff"
        )
    if mode == "autodiff":
        return AUTODIFF_SOURCE_BACKEND
    if mode == "analytic":
        require_analytic_structured_source(
            model, micro_latent=micro_latent
        )
        return ANALYTIC_SOURCE_BACKEND
    try:
        require_analytic_structured_source(
            model, micro_latent=micro_latent
        )
    except ValueError:
        return AUTODIFF_SOURCE_BACKEND
    return ANALYTIC_SOURCE_BACKEND


def manufacture_standard_vpfp_source_analytic(
    model,
    points,
    latent,
    *,
    micro_latent=None,
    equation_condition=None,
) -> dict[str, Any]:
    """Evaluate the standard VPFP residual with analytic coordinate jets.

    This pointwise interface preserves arbitrary numerical oracle access.  It
    differentiates only the represented coordinate basis explicitly; ordinary
    PyTorch reverse mode can still differentiate the result with respect to the
    latent or equation condition.
    """

    require_analytic_structured_source(model, micro_latent=micro_latent)
    torch = model.torch
    parameter = next(model.parameters())
    query = torch.as_tensor(
        points, dtype=parameter.dtype, device=parameter.device
    )
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2 or int(query.shape[1]) != 5:
        raise ValueError("points must have shape [N,5] in (t,x,y,vx,vy) order")
    condition = _condition_rows(
        model, equation_condition, count=int(query.shape[0])
    )
    macro = analytic_structured_macro_jet(model, query[:, :3], latent)
    basis = hermite_basis_coordinate_jet(
        torch,
        query[:, 3:5],
        modes=model.hermite_modes,
        temperature=condition[:, 1],
    )
    assembled = _assemble_pointwise(
        torch,
        model=model,
        velocity=query[:, 3:5],
        condition=condition,
        macro=macro,
        basis=basis,
    )
    return {
        **assembled,
        "rho": macro["rho"],
        "phi": macro["phi"],
        "electric": macro["electric"],
        "hermite_coefficients": macro["coefficients"],
        "rho_coefficients": macro["rho_coefficients"],
        "equation_condition": condition,
        "points": query,
        "coordinate_derivative_backend": ANALYTIC_SOURCE_BACKEND,
    }


def prepare_standard_vpfp_cartesian_velocity_jet(
    model,
    velocity_points,
    *,
    micro_latent=None,
    equation_condition=None,
) -> dict[str, Any]:
    """Prepare velocity-only Hermite jets for repeated Cartesian chunks."""

    require_analytic_structured_source(model, micro_latent=micro_latent)
    torch = model.torch
    parameter = next(model.parameters())
    velocity = torch.as_tensor(
        velocity_points, dtype=parameter.dtype, device=parameter.device
    )
    if velocity.ndim != 2 or int(velocity.shape[1]) != 2:
        raise ValueError("velocity_points must have shape [V,2]")
    condition = _condition_rows(model, equation_condition, count=1)[0]
    basis = hermite_basis_coordinate_jet(
        torch,
        velocity,
        modes=model.hermite_modes,
        temperature=condition[1],
    )
    return {
        "model_identity": id(model),
        "velocity": velocity,
        "condition": condition,
        "basis": basis,
    }


def manufacture_standard_vpfp_cartesian_source_analytic(
    model,
    spacetime_points,
    velocity_points,
    latent,
    *,
    micro_latent=None,
    equation_condition=None,
    prepared_velocity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a factorized Cartesian ``(t,x,y) x (vx,vy)`` query."""

    require_analytic_structured_source(model, micro_latent=micro_latent)
    torch = model.torch
    parameter = next(model.parameters())
    spacetime = torch.as_tensor(
        spacetime_points, dtype=parameter.dtype, device=parameter.device
    )
    if spacetime.ndim != 2 or int(spacetime.shape[1]) != 3:
        raise ValueError("spacetime_points must have shape [Q,3]")
    if prepared_velocity is None:
        prepared_velocity = prepare_standard_vpfp_cartesian_velocity_jet(
            model,
            velocity_points,
            micro_latent=micro_latent,
            equation_condition=equation_condition,
        )
    else:
        if velocity_points is not None or equation_condition is not None:
            raise ValueError(
                "prepared_velocity owns velocity_points and equation_condition"
            )
        if int(prepared_velocity.get("model_identity", -1)) != id(model):
            raise ValueError("prepared velocity jet belongs to another model")
    velocity = prepared_velocity["velocity"]
    condition = prepared_velocity["condition"]
    basis = prepared_velocity["basis"]
    if velocity.ndim != 2 or int(velocity.shape[1]) != 2:
        raise ValueError("prepared velocity jet is malformed")
    macro = analytic_structured_macro_jet(model, spacetime, latent)
    assembled = _assemble_cartesian(
        torch,
        model=model,
        velocity=velocity,
        condition=condition,
        macro=macro,
        basis=basis,
    )
    return {
        **assembled,
        "rho": macro["rho"],
        "phi": macro["phi"],
        "electric": macro["electric"],
        "hermite_coefficients": macro["coefficients"],
        "rho_coefficients": macro["rho_coefficients"],
        "equation_condition": condition,
        "spacetime_points": spacetime,
        "velocity_points": velocity,
        "coordinate_derivative_backend": ANALYTIC_SOURCE_BACKEND,
    }


def analytic_structured_macro_jet(model, spacetime_points, latent):
    """Return rho, Poisson field, coefficients, and their t/x/y jets."""

    require_analytic_structured_source(model)
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
    latent_row = latent_row * float(model.config.latent_input_scale)
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
    local_time_basis, local_time_derivative, local_time_normalizer = (
        _time_basis_jet(
            torch,
            model.config,
            points[:, 0],
            order=int(model.config.structured_local_time_order),
        )
    )
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


def hermite_basis_coordinate_jet(
    torch,
    velocity,
    *,
    modes: Sequence[tuple[int, int]],
    temperature,
):
    """Evaluate squared-Hermite amplitude basis derivatives through order two."""

    values = torch.as_tensor(velocity)
    if values.ndim != 2 or int(values.shape[1]) != 2:
        raise ValueError("velocity must have shape [N,2]")
    if not modes:
        raise ValueError("Hermite modes must be nonempty")
    theta = torch.as_tensor(
        temperature, dtype=values.dtype, device=values.device
    )
    if theta.ndim == 0:
        theta = theta.expand(int(values.shape[0]))
    theta = torch.broadcast_to(theta.reshape(-1), (int(values.shape[0]),))
    if bool(torch.any(theta <= 0.0).item()):
        raise ValueError("temperature must be positive")
    scale = torch.sqrt(2.0 * theta)
    maximum = max(max(int(first), int(second)) for first, second in modes)
    x = _hermite_1d_jet(torch, values[:, 0], maximum, scale)
    y = _hermite_1d_jet(torch, values[:, 1], maximum, scale)
    outputs = [[] for _ in range(5)]
    for first, second in modes:
        outputs[0].append(x[0][:, first] * y[0][:, second])
        outputs[1].append(x[1][:, first] * y[0][:, second])
        outputs[2].append(x[0][:, first] * y[1][:, second])
        outputs[3].append(x[2][:, first] * y[0][:, second])
        outputs[4].append(x[0][:, first] * y[2][:, second])
    return tuple(torch.stack(parts, dim=-1) for parts in outputs)


def _hermite_1d_jet(torch, values, maximum_degree: int, scale):
    xi = values / scale
    functions = [
        torch.pi ** (-0.25)
        / torch.sqrt(scale)
        * torch.exp(-0.5 * xi.square())
    ]
    if maximum_degree + 2 >= 1:
        functions.append(math.sqrt(2.0) * xi * functions[0])
    for order in range(1, maximum_degree + 2):
        functions.append(
            math.sqrt(2.0 / float(order + 1))
            * xi
            * functions[order]
            - math.sqrt(float(order) / float(order + 1))
            * functions[order - 1]
        )
    stacked = torch.stack(functions, dim=-1)
    first_rows = []
    second_rows = []
    for order in range(maximum_degree + 1):
        left = (
            torch.zeros_like(values)
            if order == 0
            else math.sqrt(float(order) / 2.0) * stacked[:, order - 1]
        )
        right = math.sqrt(float(order + 1) / 2.0) * stacked[:, order + 1]
        first_rows.append((left - right) / scale)
        left_second = (
            torch.zeros_like(values)
            if order < 2
            else math.sqrt(float(order * (order - 1)))
            * stacked[:, order - 2]
        )
        right_second = math.sqrt(float((order + 1) * (order + 2))) * stacked[
            :, order + 2
        ]
        second_rows.append(
            (
                left_second
                - float(2 * order + 1) * stacked[:, order]
                + right_second
            )
            / (2.0 * scale.square())
        )
    return (
        stacked[:, : maximum_degree + 1],
        torch.stack(first_rows, dim=-1),
        torch.stack(second_rows, dim=-1),
    )


def _time_basis_jet(torch, config, times, *, order: int):
    t0, t1 = (float(value) for value in config.time_domain)
    duration = t1 - t0
    tau = (times - t0) / duration
    tolerance = 32.0 * torch.finfo(times.dtype).eps
    if bool(torch.any((tau < -tolerance) | (tau > 1.0 + tolerance)).item()):
        raise ValueError("analytic time query is outside the decoder domain")
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


def _spatial_basis_jet(torch, model, points):
    x = points[:, 0:1]
    y = points[:, 1:2]
    values = [torch.ones_like(x)]
    dx_values = [torch.zeros_like(x)]
    dy_values = [torch.zeros_like(x)]
    lx, ly = (float(value) for value in model.config.equation.x_domain)
    root_two = math.sqrt(2.0)
    for kx, ky in model.local_half_modes:
        wave_x = 2.0 * torch.pi * float(kx) / lx
        wave_y = 2.0 * torch.pi * float(ky) / ly
        phase = wave_x * x + wave_y * y
        cosine = root_two * torch.cos(phase)
        sine = root_two * torch.sin(phase)
        values.extend((cosine, sine))
        dx_values.extend((-wave_x * sine, wave_x * cosine))
        dy_values.extend((-wave_y * sine, wave_y * cosine))
    outputs = tuple(
        torch.cat(parts, dim=-1)
        for parts in (values, dx_values, dy_values)
    )
    expected = int(model.structured_layout["local_spatial_feature_count"])
    if any(int(value.shape[1]) != expected for value in outputs):
        raise RuntimeError("analytic spatial basis size mismatch")
    return outputs


def _rho_coordinate_jet(model, points, raw_values, raw_time_derivative):
    torch = model.torch
    mode_count = len(model.rho_half_modes)
    raw = raw_values.reshape(int(points.shape[0]), mode_count, 2)
    raw_t = raw_time_derivative.reshape_as(raw)
    norms = torch.linalg.vector_norm(raw, dim=-1)
    raw_t_norms = torch.linalg.vector_norm(raw_t, dim=-1)
    nondifferentiable = (norms == 0.0) & (raw_t_norms > 0.0)
    if bool(torch.any(nondifferentiable).item()):
        raise ValueError(
            "phase-invariant density group norm crosses zero with nonzero "
            "time derivative"
        )
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
    phi = torch.sum(mode_values / kappa_squared[None, :], dim=-1)
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
        "phi": phi,
        "electric": torch.stack((electric_x, electric_y), dim=-1),
        "rho_coefficients": coefficients,
    }


def _assemble_pointwise(torch, *, model, velocity, condition, macro, basis):
    basis_value, basis_vx, basis_vy, basis_vx2, basis_vy2 = basis
    coefficients = macro["coefficients"]
    coefficient_d = macro["coefficient_derivatives"]
    amplitude = torch.sum(coefficients * basis_value, dim=-1)
    amplitude_d = torch.einsum("dqm,qm->dq", coefficient_d, basis_value)
    amplitude_vx = torch.sum(coefficients * basis_vx, dim=-1)
    amplitude_vy = torch.sum(coefficients * basis_vy, dim=-1)
    amplitude_vx2 = torch.sum(coefficients * basis_vx2, dim=-1)
    amplitude_vy2 = torch.sum(coefficients * basis_vy2, dim=-1)
    return _assemble_from_amplitudes(
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


def _assemble_cartesian(torch, *, model, velocity, condition, macro, basis):
    basis_value, basis_vx, basis_vy, basis_vx2, basis_vy2 = basis
    coefficients = macro["coefficients"]
    coefficient_d = macro["coefficient_derivatives"]
    return _assemble_from_amplitudes(
        torch,
        model=model,
        velocity_x=velocity[:, 0][None, :],
        velocity_y=velocity[:, 1][None, :],
        condition=condition,
        macro=macro,
        amplitude=coefficients @ basis_value.transpose(0, 1),
        amplitude_d=torch.einsum("dqm,vm->dqv", coefficient_d, basis_value),
        amplitude_vx=coefficients @ basis_vx.transpose(0, 1),
        amplitude_vy=coefficients @ basis_vy.transpose(0, 1),
        amplitude_vx2=coefficients @ basis_vx2.transpose(0, 1),
        amplitude_vy2=coefficients @ basis_vy2.transpose(0, 1),
        cartesian=True,
    )


def _assemble_from_amplitudes(
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
    f_gradient = torch.stack(
        (f_d[0], f_d[1], f_d[2], f_vx, f_vy), dim=-1
    )
    return {
        "g": g,
        "f": f,
        "maxwellian": torch.broadcast_to(maxwellian, g.shape),
        "source": source,
        "f_gradient": f_gradient,
        "f_velocity_laplacian": f_velocity_laplacian,
        "collision": collision,
    }


def _condition_rows(model, equation_condition, *, count: int):
    rows, _normalized = model._equation_condition_rows(
        equation_condition, count=int(count)
    )
    return rows
