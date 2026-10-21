"""Exact macro-only query path for the existing continuous A decoder.

This changes computation, not the represented answer: native rho/Poisson,
original macro interpolation, and density-weighted Parseval recentering are
reused. It intentionally does not evaluate microscopic Hermite moments or f.
Consequently finite macro output is NOT a certificate that f or its source is
finite/physical; those require the unchanged complete-answer source audit.
"""
from __future__ import annotations

import torch

from .moving_frame import MovingFrameDecoder, _finite
from .moving_frame_backend import _query
from .value_control_diffusion import lift_value_controls


def evaluate_moving_frame_macros_fast(decoder, points_txy, carrier, condition):
    """Same seven fields and t/x/y first derivatives as the native macro API.

    All input coordinates retain the original FP64 query / native decoder
    dtype split. Shared or per-query carriers remain supported; no unique-time
    cache joins otherwise independent derivative rows. The existing native
    lift is retained (including its input-finiteness and inactive checks).

    Only the original direct ``MovingFrameDecoder`` contract is supported.
    Unsupported decoder variants raise rather than silently approximate or
    omit their custom field semantics. Source manufacturing is never changed.
    """
    if type(decoder) is not MovingFrameDecoder:
        raise NotImplementedError("fast macros require the original MovingFrameDecoder, not a custom fields override")
    base = decoder.base_decoder
    cfg = base.config
    if (not cfg.structured_spectral_latent or not cfg.structured_direct_coefficients
            or cfg.structured_residual_scale != 0
            or cfg.structured_time_basis != "piecewise_cubic_hermite_v1"
            or cfg.rho_positivity_gauge not in ("coordinate_l1_v1", "phase_invariant_group_l1_v2")):
        raise NotImplementedError("fast macros require the original direct Hermite/value rho ABI and registered gauge")
    if not isinstance(carrier, dict) or set(carrier) != {"m", "z"}:
        raise ValueError("fast moving-frame macros require the exact complete m/z carrier")
    if any(not isinstance(carrier[key], torch.Tensor) for key in ("m", "z")):
        raise ValueError("native floating m/z tensors required")
    for key in ("m", "z"):
        _finite(carrier[key], key)
    points = _query(decoder, points_txy, 3)
    phase = torch.cat((points, points.new_zeros((len(points), 2))), dim=1)
    z = carrier["z"]
    full = lift_value_controls(z[None] if z.ndim == 1 else z, decoder.value_config)
    physical, _ = base._equation_condition_rows(condition, count=len(points))
    _finite(physical, "equation_condition")
    # This method exactly replays native latent scaling, time-basis contraction
    # and native dtype. Reusing it also preserves the decoder's fixed-zero
    # group semantics; a nonconstant zero crossing is NOT epsilon-smoothed.
    native_query, raw, constant_zero = decoder._native_rho_groups(phase, full)
    if cfg.rho_positivity_gauge == "phase_invariant_group_l1_v2" and bool(constant_zero.any()):
        rho, phi, electric, coefficients = decoder._rho_with_constant_zero_groups(native_query, raw, constant_zero)
    else:
        rho, phi, electric, coefficients = base._rho_poisson(native_query, raw)
    for key, value in (("rho", rho), ("phi", phi), ("electric", electric), ("rho_coefficients", coefficients)):
        _finite(value, key)
    target = decoder.macro_at(phase, carrier["m"], {"rho_coefficients": coefficients})
    rho = rho.double()
    velocity, covariance = target["mean"], target["covariance"]
    momentum = rho[:, None] * velocity
    kinetic = .5 * rho * (velocity.square().sum(-1) + covariance.diagonal(dim1=-2, dim2=-1).sum(-1))
    raw_second = rho[:, None, None] * (covariance + velocity[:, :, None] * velocity[:, None, :])
    result = {"rho": rho, "momentum": momentum, "kinetic_energy": kinetic,
        "raw_second_moment": torch.stack((raw_second[:, 0, 0], raw_second[:, 0, 1], raw_second[:, 1, 1]), dim=1),
        "electric": electric, "velocity": velocity, "central_covariance": covariance}
    for key, value in result.items():
        _finite(value, key)
    return result


def transport_from_geometry(geometry, decoder):
    """Add v2 values only; these third moments never enter the source JVP graph."""
    from .transport_moments import (affine_central_third,
        heat_flux_from_central_third, squared_hermite_transport_moments)
    moments = squared_hermite_transport_moments(
        geometry["source_hermite_coefficients"], modes=decoder.base_decoder.hermite_modes,
        temperature=geometry["equation_condition"][:, 1],
        floor_mixture=decoder.base_decoder.config.velocity_floor_mixture)
    source, target = geometry["_source_factor"], geometry["_target_factor"]
    # A = L_target L_source^-1; the inverse density map uses the opposite A.
    affine = torch.linalg.solve_triangular(source.transpose(-1, -2),
        target.transpose(-1, -2), upper=True).transpose(-1, -2)
    central = affine_central_third(moments["central_third_moment"], affine)
    rho, u, covariance = geometry["rho"].double(), geometry["velocity"], geometry["central_covariance"]
    pressure = rho[:, None, None]*covariance
    raw = pressure+rho[:, None, None]*u[:, :, None]*u[:, None, :]
    result = {"rho": rho, "momentum": rho[:, None]*u,
              "kinetic_energy": .5*(raw[:, 0, 0]+raw[:, 1, 1]),
              "raw_second_moment": torch.stack((raw[:, 0, 0], raw[:, 0, 1], raw[:, 1, 1]), -1),
              "electric": geometry["electric"].double(),
              "pressure": torch.stack((pressure[:, 0, 0], pressure[:, 0, 1], pressure[:, 1, 1]), -1),
              "heat_flux": heat_flux_from_central_third(rho, central)}
    for name, value in result.items():
        _finite(value, name)
    return result


def evaluate_moving_frame_transport_macros(decoder, points_txy, carrier, condition):
    """Full-shape transport query without velocity grids or source derivatives."""
    from .moving_frame import source_local_moments
    points = _query(decoder, points_txy, 3)
    phase = torch.cat((points, points.new_zeros((len(points), 2))), -1)
    native = decoder.source_fields(phase, carrier["z"], condition)
    original = source_local_moments(native, decoder.base_decoder)
    target = decoder.macro_at(phase, carrier["m"], native)
    geometry = {"rho": native["rho"], "electric": native["electric"],
                "velocity": target["mean"], "central_covariance": target["covariance"],
                "source_hermite_coefficients": native["hermite_coefficients"],
                "equation_condition": native["equation_condition"],
                "_source_factor": original["factor"], "_target_factor": target["factor"]}
    return transport_from_geometry(geometry, decoder)
