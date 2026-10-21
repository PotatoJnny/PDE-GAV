from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_v2 import (
    VELOCITY_DENSITY_DICTIONARY,
    VELOCITY_SQUARE_HERMITE,
    manufacture_standard_vpfp_source,
)


@dataclass(frozen=True)
class VPFPMicroRegularityConfig:
    """Validity budget for a zero-order adversarial micro perturbation.

    Defaults are conservative engineering starting points. Main-experiment
    thresholds must be calibrated from equation-first reference cases.
    """

    velocity_order: float = 2.0
    max_relative_f_l2: float = 0.12
    max_kl_g: float = 0.04
    max_relative_fisher: float = 0.50
    max_spacetime_log_gradient_rms: float = 2.0
    max_weighted_velocity_sobolev: float = 0.10
    max_source_rms_ratio: float = 2.0
    max_source_delta_rms_ratio: float = 0.75
    max_tail_mass: float = 1.0e-5
    max_ou_boundary_flux: float = 1.0e-4
    velocity_windows: tuple[float, ...] = (4.0, 5.0, 6.0)
    require_source_metrics: bool = True
    require_tail_metric: bool = True

    def validate(self) -> None:
        nonnegative = (
            self.velocity_order,
            self.max_relative_f_l2,
            self.max_kl_g,
            self.max_relative_fisher,
            self.max_spacetime_log_gradient_rms,
            self.max_weighted_velocity_sobolev,
            self.max_source_rms_ratio,
            self.max_source_delta_rms_ratio,
            self.max_tail_mass,
            self.max_ou_boundary_flux,
        )
        if min(nonnegative) < 0.0:
            raise ValueError("regularity thresholds must be nonnegative")
        if not self.velocity_windows or min(self.velocity_windows) <= 0.0:
            raise ValueError("velocity_windows must be positive")
        if tuple(sorted(self.velocity_windows)) != self.velocity_windows:
            raise ValueError("velocity_windows must be increasing")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["velocity_windows"] = [float(value) for value in self.velocity_windows]
        return payload


def weighted_hermite_sobolev_penalty(
    coefficient_delta,
    multi_indices: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    order: float = 2.0,
):
    """Return the harmonic-oscillator weighted coefficient energy.

    Hermite coefficients diagonalize the Gaussian harmonic oscillator, so a
    weight growing with total degree is the natural whole-velocity analogue of
    a Fourier Sobolev penalty.
    """

    torch = _torch()
    values = torch.as_tensor(coefficient_delta)
    if values.ndim < 1 or int(values.shape[-1]) != len(multi_indices):
        raise ValueError("coefficient_delta trailing dimension does not match multi_indices")
    weights = torch.as_tensor(
        [(1.0 + ax + ay) ** float(order) for ax, ay in multi_indices],
        dtype=values.dtype,
        device=values.device,
    )
    return torch.mean(torch.sum(values.square() * weights, dim=-1))


def weighted_density_dictionary_penalty(model, coefficient_delta, *, order: float = 2.0):
    """Penalize coefficient changes more strongly on narrow density atoms."""

    torch = _torch()
    values = torch.as_tensor(coefficient_delta)
    stds = model.density_atom_stds()
    if values.ndim < 1 or int(values.shape[-1]) != int(stds.shape[0]):
        raise ValueError("coefficient_delta does not match the density dictionary")
    inverse_scale = torch.mean(stds.reciprocal().square(), dim=-1)
    weights = (1.0 + inverse_scale) ** float(order)
    return torch.mean(torch.sum(values.square() * weights[None, :], dim=-1))


def micro_regularity_metrics(
    model,
    points,
    latent,
    micro_latent,
    *,
    point_weights=None,
    include_source: bool = True,
    velocity_order: float = 2.0,
    create_graph: bool = False,
) -> dict[str, Any]:
    """Measure micro complexity while retaining gradients through validity terms.

    Only the external solver reward needs a zero-order estimator. These
    regularity and PDE construction terms remain exactly differentiable.
    """

    torch = _torch()
    parameter = next(model.parameters())
    query = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
    if query.ndim == 1:
        query = query[None, :]
    query = query.detach().clone().requires_grad_(True)
    base = model.fields(query, latent)
    perturbed = model.fields(query, latent, micro_latent=micro_latent)
    weights = _normalized_weights(
        torch,
        point_weights,
        count=int(query.shape[0]),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    epsilon = torch.finfo(parameter.dtype).eps
    f_denominator = torch.sum(weights * base["f"].square()).clamp_min(epsilon)
    relative_f_l2 = torch.sqrt(
        torch.sum(weights * (perturbed["f"] - base["f"]).square()) / f_denominator
    )
    log_ratio = torch.log(perturbed["g"].clamp_min(epsilon)) - torch.log(
        base["g"].clamp_min(epsilon)
    )
    probability_weight = weights * perturbed["g"]
    probability_mass = torch.sum(probability_weight).clamp_min(epsilon)
    kl_g = torch.sum(probability_weight * log_ratio) / probability_mass
    log_ratio_gradient = torch.autograd.grad(
        log_ratio.sum(),
        query,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    relative_fisher = torch.sum(
        probability_weight
        * (log_ratio_gradient[:, 3].square() + log_ratio_gradient[:, 4].square())
    ) / probability_mass
    spacetime_log_gradient_rms = torch.sqrt(
        torch.sum(
            probability_weight[:, None] * log_ratio_gradient[:, :3].square()
        )
        / (3.0 * probability_mass)
    )
    coefficient_delta = (
        perturbed["velocity_coefficients"] - base["velocity_coefficients"]
    )
    if model.config.velocity_representation == VELOCITY_SQUARE_HERMITE:
        spectral_penalty = weighted_hermite_sobolev_penalty(
            coefficient_delta, model.velocity_multi_indices, order=velocity_order
        )
    else:
        spectral_penalty = weighted_density_dictionary_penalty(
            model, coefficient_delta, order=velocity_order
        )

    output: dict[str, Any] = {
        "relative_f_l2": relative_f_l2,
        "kl_g": kl_g,
        "relative_fisher": relative_fisher,
        "spacetime_log_gradient_rms": spacetime_log_gradient_rms,
        "weighted_velocity_sobolev": spectral_penalty,
    }
    if include_source:
        base_source = manufacture_standard_vpfp_source(
            model,
            points,
            latent,
            create_graph=create_graph,
        )["source"]
        perturbed_source = manufacture_standard_vpfp_source(
            model,
            points,
            latent,
            micro_latent=micro_latent,
            create_graph=create_graph,
        )["source"]
        source_scale = torch.sqrt(torch.sum(weights * base_source.square())).clamp_min(epsilon)
        output["source_rms_ratio"] = (
            torch.sqrt(torch.sum(weights * perturbed_source.square())) / source_scale
        )
        output["source_delta_rms_ratio"] = (
            torch.sqrt(torch.sum(weights * (perturbed_source - base_source).square()))
            / source_scale
        )
    return output


def regularity_gate(
    metrics: dict[str, Any],
    config: VPFPMicroRegularityConfig,
) -> dict[str, Any]:
    config.validate()
    limits = {
        "relative_f_l2": config.max_relative_f_l2,
        "kl_g": config.max_kl_g,
        "relative_fisher": config.max_relative_fisher,
        "spacetime_log_gradient_rms": config.max_spacetime_log_gradient_rms,
        "weighted_velocity_sobolev": config.max_weighted_velocity_sobolev,
        "source_rms_ratio": config.max_source_rms_ratio,
        "source_delta_rms_ratio": config.max_source_delta_rms_ratio,
        "tail_mass_largest_window": config.max_tail_mass,
        "ou_boundary_flux_largest_window": config.max_ou_boundary_flux,
    }
    numeric = {
        name: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
        for name, value in metrics.items()
    }
    required = {
        "relative_f_l2",
        "kl_g",
        "relative_fisher",
        "spacetime_log_gradient_rms",
        "weighted_velocity_sobolev",
    }
    if config.require_source_metrics:
        required.update(("source_rms_ratio", "source_delta_rms_ratio"))
    if config.require_tail_metric:
        required.add("tail_mass_largest_window")
        required.add("ou_boundary_flux_largest_window")
    failures = [f"missing_{name}" for name in sorted(required - set(numeric))]
    failures.extend(
        name
        for name, limit in limits.items()
        if name in numeric and (not np.isfinite(numeric[name]) or numeric[name] > limit)
    )
    return {
        "accepted": not failures,
        "failures": failures,
        "metrics": numeric,
        "limits": limits,
        "config": config.to_dict(),
    }


def gauss_hermite_velocity_rule(order: int, temperature: float = 1.0):
    """Tensor rule for expectations under the normalized 2-D Maxwellian."""

    if order < 2:
        raise ValueError("Gauss-Hermite order must be at least two")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    standard_nodes, standard_weights = _gauss_hermite_1d_rule(order, temperature)
    vx, vy = np.meshgrid(standard_nodes, standard_nodes, indexing="ij")
    wx, wy = np.meshgrid(standard_weights, standard_weights, indexing="ij")
    return vx.reshape(-1), vy.reshape(-1), (wx * wy).reshape(-1)



def _gauss_hermite_1d_rule(order: int, temperature: float):
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    standard_nodes = np.sqrt(2.0 * float(temperature)) * nodes
    standard_weights = weights / np.sqrt(np.pi)
    return standard_nodes, standard_weights

def velocity_truncation_audit(
    model,
    spacetime_points,
    latent,
    *,
    micro_latent=None,
    windows: tuple[float, ...] = (4.0, 5.0, 6.0),
    quadrature_order: int = 24,
) -> dict[str, Any]:
    """Audit tail probability without making a velocity box part of truth."""

    torch = _torch()
    parameter = next(model.parameters())
    tx = torch.as_tensor(
        spacetime_points,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if tx.ndim == 1:
        tx = tx[None, :]
    if tx.ndim != 2 or int(tx.shape[-1]) != 3:
        raise ValueError("spacetime_points must have shape [N,3] in (t,x,y) order")
    vx, vy, weights = gauss_hermite_velocity_rule(
        quadrature_order,
        model.config.equation.temperature,
    )
    if model.config.velocity_representation == VELOCITY_DENSITY_DICTIONARY:
        coefficient_points = torch.cat(
            (tx, torch.zeros((int(tx.shape[0]), 2), dtype=tx.dtype, device=tx.device)),
            dim=-1,
        )
        with torch.no_grad():
            mixture_coefficients = model.fields(
                coefficient_points, latent, micro_latent=micro_latent
            )["velocity_coefficients"]
        ratio = None
        quadrature_weights = None
        total_mass = torch.ones(int(tx.shape[0]), dtype=tx.dtype, device=tx.device)
        normalization_backend = "analytic_normalized_density_dictionary"
    else:
        velocity_count = int(vx.size)
        repeated_tx = tx[:, None, :].expand(-1, velocity_count, -1).reshape(-1, 3)
        velocity = torch.as_tensor(
            np.stack((vx, vy), axis=-1),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        repeated_velocity = velocity[None, :, :].expand(
            int(tx.shape[0]), -1, -1
        ).reshape(-1, 2)
        points = torch.cat((repeated_tx, repeated_velocity), dim=-1)
        with torch.no_grad():
            values = model.fields(points, latent, micro_latent=micro_latent)
        ratio = (values["g"] / values["maxwellian"]).reshape(
            int(tx.shape[0]), velocity_count
        )
        quadrature_weights = torch.as_tensor(
            weights, dtype=parameter.dtype, device=parameter.device
        )
        total_mass = torch.sum(ratio * quadrature_weights[None, :], dim=-1)
        normalization_backend = "gauss_hermite_independent_audit"
    tangent_nodes, tangent_weights = _gauss_hermite_1d_rule(
        quadrature_order, model.config.equation.temperature
    )
    rows = []
    for window in windows:
        boundary_flux = _ou_boundary_flux(
            model,
            tx,
            latent,
            micro_latent=micro_latent,
            window=float(window),
            tangent_nodes=tangent_nodes,
            tangent_weights=tangent_weights,
        )
        if model.config.velocity_representation == VELOCITY_DENSITY_DICTIONARY:
            tail = _density_dictionary_square_tail(
                model, mixture_coefficients, float(window)
            )
        else:
            outside = torch.as_tensor(
                (np.abs(vx) > float(window)) | (np.abs(vy) > float(window)),
                device=parameter.device,
            )
            tail = torch.sum(
                ratio[:, outside] * quadrature_weights[None, outside],
                dim=-1,
            ) / total_mass
        rows.append(
            {
                "window": float(window),
                "mean_tail_mass": float(torch.mean(tail).detach().cpu()),
                "max_tail_mass": float(torch.max(tail).detach().cpu()),
                "mean_ou_boundary_flux": float(
                    torch.mean(boundary_flux).detach().cpu()
                ),
                "max_ou_boundary_flux": float(
                    torch.max(boundary_flux).detach().cpu()
                ),
            }
        )
    monotone = all(
        rows[index + 1]["max_tail_mass"] <= rows[index]["max_tail_mass"] + 1.0e-14
        for index in range(len(rows) - 1)
    )
    return {
        "schema": "vpfp_continuous_velocity_truncation_audit_v1",
        "truth_velocity_domain": "R^2",
        "quadrature": "tensor_gauss_hermite_under_maxwellian",
        "quadrature_order": int(quadrature_order),
        "windows": rows,
        "normalization_backend": normalization_backend,
        "tail_mass_monotone": monotone,
        "normalization_max_abs_error": float(
            torch.max(torch.abs(total_mass - 1.0)).detach().cpu()
        ),
        "tail_mass_largest_window": rows[-1]["max_tail_mass"],
        "ou_boundary_flux_largest_window": rows[-1]["max_ou_boundary_flux"],
    }


def _density_dictionary_square_tail(model, coefficients, window: float):
    torch = _torch()
    means = model.density_atom_means
    stds = model.density_atom_stds()
    scale = float(2.0**0.5)
    upper = (float(window) - means) / (scale * stds)
    lower = (-float(window) - means) / (scale * stds)
    inside_axis = 0.5 * (torch.erf(upper) - torch.erf(lower))
    atom_tail = (1.0 - inside_axis[:, 0] * inside_axis[:, 1]).clamp(0.0, 1.0)
    return torch.sum(coefficients * atom_tail[None, :], dim=-1)


def _ou_boundary_flux(
    model,
    spacetime_points,
    latent,
    *,
    micro_latent,
    window: float,
    tangent_nodes,
    tangent_weights,
):
    torch = _torch()
    parameter = next(model.parameters())
    tx = spacetime_points
    nodes = torch.as_tensor(tangent_nodes, dtype=parameter.dtype, device=parameter.device)
    weights = torch.as_tensor(tangent_weights, dtype=parameter.dtype, device=parameter.device)
    count = int(tx.shape[0])
    node_count = int(nodes.numel())
    theta = float(model.config.equation.temperature)
    marginal = torch.exp(-0.5 * nodes.square() / theta) / (
        2.0 * torch.pi * theta
    ) ** 0.5
    rows = []
    for normal_axis in (0, 1):
        for sign in (-1.0, 1.0):
            repeated_tx = tx[:, None, :].expand(-1, node_count, -1).reshape(-1, 3)
            velocity = torch.empty(
                (count, node_count, 2),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            velocity[:, :, normal_axis] = sign * float(window)
            velocity[:, :, 1 - normal_axis] = nodes[None, :]
            points = torch.cat((repeated_tx, velocity.reshape(-1, 2)), dim=-1)
            points = points.detach().clone().requires_grad_(True)
            values = model.fields(points, latent, micro_latent=micro_latent)
            gradient = torch.autograd.grad(
                values["f"].sum(),
                points,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
            normal_current = (
                points[:, 3 + normal_axis] * values["f"]
                + theta * gradient[:, 3 + normal_axis]
            ).reshape(count, node_count)
            integrated = torch.sum(
                weights[None, :] * torch.abs(normal_current) / marginal[None, :],
                dim=-1,
            )
            rows.append(integrated)
    return torch.stack(rows, dim=-1)


def _normalized_weights(torch, values, *, count: int, dtype, device):
    if values is None:
        return torch.full((count,), 1.0 / float(count), dtype=dtype, device=device)
    weights = torch.as_tensor(values, dtype=dtype, device=device).reshape(-1)
    if int(weights.numel()) != count or bool(torch.any(weights < 0.0)):
        raise ValueError("point_weights must be nonnegative with one value per point")
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for VPFP regularity audits") from exc
    return torch
