"""Structure-preserving perturbations for continuous answer-first VPFP cases.

The base procedural generator uses direct Fourier/time/Hermite coefficients.
This module embeds one generated answer into a strictly larger representation
and adds private smooth components in newly available Fourier and Hermite
directions.  Positivity and full-space velocity normalization remain exact
properties of the represented function; no sampled velocity quadrature is
used to repair the perturbation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .analytic_source import (
    analytic_structured_macro_jet,
    manufacture_standard_vpfp_source_analytic,
    require_analytic_structured_source,
)
from .continuous_hermite import (
    VPFPContinuousHermiteConfig,
    VPFPContinuousHermiteField,
)
from .continuous_macros import evaluate_continuous_hermite_macro_points
from .continuous_v2 import _real_fourier_half_modes
from .hermite_density import total_degree_mode_pairs


VPFP_ROBUST_PERTURBATION_SCHEMA = "vpfp_structured_answer_perturbation_v2"
VPFP_ROBUST_PERTURBATION_AUDIT_SCHEMA = (
    "vpfp_structured_answer_perturbation_audit_v2"
)


@dataclass(frozen=True)
class VPFPRobustPerturbationConfig:
    base_hermite_total_degree: int = 14
    expanded_rho_max_mode_x: int = 8
    expanded_rho_max_mode_y: int = 8
    expanded_local_spatial_mode: int = 6
    expanded_hermite_total_degree: int = 18
    density_rank: int = 6
    kinetic_rank: int = 12
    maximum_temporal_frequency: int = 5
    density_raw_scale: float = 0.025
    kinetic_raw_scale: float = 0.055
    initial_scale: float = 1.0
    backtrack_factor: float = 0.5
    maximum_backtracks: int = 9
    macro_relative_l2_minimum: float = 2.0e-3
    macro_relative_l2_maximum: float = 0.20
    microscopic_coefficient_rms_minimum: float = 5.0e-3
    microscopic_coefficient_rms_maximum: float = 0.35
    expanded_hermite_energy_minimum: float = 1.0e-5
    source_increment_relative_minimum: float = 1.0e-4
    source_increment_relative_maximum: float = 0.80
    temporal_derivative_rms_maximum: float = 4.0
    initial_macro_relative_tolerance: float = 2.0e-6
    initial_coefficient_rms_tolerance: float = 2.0e-6
    spacetime_probe_count: int = 256
    phase_probe_count: int = 512

    def __post_init__(self) -> None:
        integers = (
            self.base_hermite_total_degree,
            self.expanded_rho_max_mode_x,
            self.expanded_rho_max_mode_y,
            self.expanded_local_spatial_mode,
            self.expanded_hermite_total_degree,
            self.density_rank,
            self.kinetic_rank,
            self.maximum_temporal_frequency,
            self.maximum_backtracks,
            self.spacetime_probe_count,
            self.phase_probe_count,
        )
        if any(isinstance(value, bool) or int(value) < 1 for value in integers):
            raise ValueError("VPFP perturbation integer settings must be positive")
        positive = (
            self.density_raw_scale,
            self.kinetic_raw_scale,
            self.initial_scale,
            self.backtrack_factor,
            self.macro_relative_l2_minimum,
            self.macro_relative_l2_maximum,
            self.microscopic_coefficient_rms_minimum,
            self.microscopic_coefficient_rms_maximum,
            self.expanded_hermite_energy_minimum,
            self.source_increment_relative_minimum,
            self.source_increment_relative_maximum,
            self.temporal_derivative_rms_maximum,
            self.initial_macro_relative_tolerance,
            self.initial_coefficient_rms_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("VPFP perturbation thresholds must be positive")
        if not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("VPFP perturbation backtrack factor must lie in (0,1)")
        if self.macro_relative_l2_minimum >= self.macro_relative_l2_maximum:
            raise ValueError("VPFP perturbation macro interval is empty")
        if (
            self.microscopic_coefficient_rms_minimum
            >= self.microscopic_coefficient_rms_maximum
        ):
            raise ValueError("VPFP perturbation microscopic interval is empty")
        if (
            self.source_increment_relative_minimum
            >= self.source_increment_relative_maximum
        ):
            raise ValueError("VPFP perturbation source interval is empty")


@dataclass
class VPFPRobustPerturbationResult:
    model: Any
    embedded_base_latent: Any
    perturbed_latent: Any | None
    raw_perturbation_latent: Any
    report: dict[str, Any]


def expanded_vpfp_perturbation_config(
    base: VPFPContinuousHermiteConfig | dict[str, Any],
    perturbation: VPFPRobustPerturbationConfig,
) -> VPFPContinuousHermiteConfig:
    config = (
        base
        if isinstance(base, VPFPContinuousHermiteConfig)
        else VPFPContinuousHermiteConfig.from_dict(dict(base))
    )
    _require_direct_structured_config(config)
    if int(perturbation.base_hermite_total_degree) != int(
        config.hermite_total_degree
    ):
        raise ValueError("VPFP perturbation policy does not match the base Hermite degree")
    if (
        perturbation.expanded_rho_max_mode_x <= config.rho_max_mode_x
        or perturbation.expanded_rho_max_mode_y <= config.rho_max_mode_y
        or perturbation.expanded_local_spatial_mode
        <= config.structured_local_spatial_mode
        or perturbation.expanded_hermite_total_degree
        <= config.hermite_total_degree
    ):
        raise ValueError("VPFP perturbation representation must strictly expand every axis")
    hermite_count = len(
        total_degree_mode_pairs(perturbation.expanded_hermite_total_degree)
    )
    provisional = replace(
        config,
        latent_dim=1,
        rho_max_mode_x=int(perturbation.expanded_rho_max_mode_x),
        rho_max_mode_y=int(perturbation.expanded_rho_max_mode_y),
        structured_local_spatial_mode=int(
            perturbation.expanded_local_spatial_mode
        ),
        hermite_total_degree=int(perturbation.expanded_hermite_total_degree),
        structured_local_channels=hermite_count,
    )
    expanded = replace(
        provisional,
        latent_dim=int(provisional.structured_latent_layout()["total_latent_dim"]),
    )
    expanded.validate()
    return expanded


def build_deterministic_direct_model(
    config: VPFPContinuousHermiteConfig | dict[str, Any],
    *,
    device: Any = "cpu",
):
    """Build a direct-coefficient decoder whose unused neural weights are zero."""

    model = VPFPContinuousHermiteField.build(config)
    torch = model.torch
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require_analytic_structured_source(model)
    return model


def embed_vpfp_structured_latent(
    base_config: VPFPContinuousHermiteConfig | dict[str, Any],
    expanded_config: VPFPContinuousHermiteConfig | dict[str, Any],
    latent: Any,
) -> Any:
    """Embed direct coefficients exactly into a larger Fourier/Hermite layout."""

    base = _coerce_config(base_config)
    expanded = _coerce_config(expanded_config)
    _require_direct_structured_config(base)
    _require_direct_structured_config(expanded)
    if (
        base.structured_time_basis != expanded.structured_time_basis
        or base.structured_time_control_count
        != expanded.structured_time_control_count
        or base.time_domain != expanded.time_domain
        or base.equation != expanded.equation
        or base.latent_input_scale != expanded.latent_input_scale
    ):
        raise ValueError("VPFP latent embedding requires identical time and equation conventions")

    import torch

    source = torch.as_tensor(latent)
    squeeze = source.ndim == 1
    if squeeze:
        source = source[None, :]
    if source.ndim != 2 or int(source.shape[1]) != int(base.latent_dim):
        raise ValueError("base VPFP latent has the wrong shape")
    result = torch.zeros(
        (int(source.shape[0]), int(expanded.latent_dim)),
        dtype=source.dtype,
        device=source.device,
    )
    base_layout = base.structured_latent_layout()
    expanded_layout = expanded.structured_latent_layout()

    base_rho_modes = _real_fourier_half_modes(
        base.rho_max_mode_x, base.rho_max_mode_y
    )
    expanded_rho_modes = _real_fourier_half_modes(
        expanded.rho_max_mode_x, expanded.rho_max_mode_y
    )
    rho_map = {mode: index for index, mode in enumerate(expanded_rho_modes)}
    base_rho = source[:, : base_layout["rho_latent_dim"]].reshape(
        int(source.shape[0]),
        int(base_layout["rho_time_feature_count"]),
        2 * len(base_rho_modes),
    )
    expanded_rho = result[:, : expanded_layout["rho_latent_dim"]].reshape(
        int(source.shape[0]),
        int(expanded_layout["rho_time_feature_count"]),
        2 * len(expanded_rho_modes),
    )
    for base_index, mode in enumerate(base_rho_modes):
        target = rho_map[mode]
        expanded_rho[:, :, 2 * target : 2 * target + 2] = base_rho[
            :, :, 2 * base_index : 2 * base_index + 2
        ]

    base_global_start = int(base_layout["rho_latent_dim"])
    expanded_global_start = int(expanded_layout["rho_latent_dim"])
    global_width = int(base_layout["kinetic_global_latent_dim"])
    if global_width != int(expanded_layout["kinetic_global_latent_dim"]):
        raise ValueError("VPFP latent embedding changed the global kinetic width")
    result[:, expanded_global_start : expanded_global_start + global_width] = source[
        :, base_global_start : base_global_start + global_width
    ]

    base_local_start = base_global_start + global_width
    expanded_local_start = expanded_global_start + global_width
    base_modes = total_degree_mode_pairs(base.hermite_total_degree)
    expanded_modes = total_degree_mode_pairs(expanded.hermite_total_degree)
    hermite_map = {mode: index for index, mode in enumerate(expanded_modes)}
    base_features = _local_spatial_feature_labels(
        base.structured_local_spatial_mode
    )
    expanded_features = _local_spatial_feature_labels(
        expanded.structured_local_spatial_mode
    )
    feature_map = {label: index for index, label in enumerate(expanded_features)}
    time_width = int(base_layout["local_time_feature_count"])
    if time_width != int(expanded_layout["local_time_feature_count"]):
        raise ValueError("VPFP latent embedding changed the local time basis")
    base_local = source[:, base_local_start:].reshape(
        int(source.shape[0]), len(base_modes), time_width, len(base_features)
    )
    expanded_local = result[:, expanded_local_start:].reshape(
        int(source.shape[0]), len(expanded_modes), time_width, len(expanded_features)
    )
    for old_channel, mode in enumerate(base_modes):
        new_channel = hermite_map[mode]
        for old_feature, label in enumerate(base_features):
            expanded_local[:, new_channel, :, feature_map[label]] = base_local[
                :, old_channel, :, old_feature
            ]
    return result[0] if squeeze else result


def sample_vpfp_private_perturbation_latent(
    base_config: VPFPContinuousHermiteConfig | dict[str, Any],
    expanded_config: VPFPContinuousHermiteConfig | dict[str, Any],
    *,
    seed: int,
    perturbation: VPFPRobustPerturbationConfig,
    dtype: Any,
    device: Any,
) -> Any:
    """Sample smooth low-rank coefficients only in expanded directions."""

    import torch

    base = _coerce_config(base_config)
    expanded = _coerce_config(expanded_config)
    rng = np.random.default_rng(int(seed))
    delta = torch.zeros(int(expanded.latent_dim), dtype=dtype, device=device)
    layout = expanded.structured_latent_layout()
    count = int(expanded.structured_time_control_count)
    time_width = int(layout["local_time_feature_count"])
    if time_width != 2 * count:
        raise ValueError("VPFP perturbation requires piecewise cubic Hermite time controls")
    latent_scale = float(expanded.latent_input_scale)

    base_rho_modes = set(
        _real_fourier_half_modes(base.rho_max_mode_x, base.rho_max_mode_y)
    )
    expanded_rho_modes = _real_fourier_half_modes(
        expanded.rho_max_mode_x, expanded.rho_max_mode_y
    )
    new_rho_indices = [
        index for index, mode in enumerate(expanded_rho_modes) if mode not in base_rho_modes
    ]
    if not new_rho_indices:
        raise ValueError("VPFP perturbation has no expanded density modes")
    rho = delta[: layout["rho_latent_dim"]].reshape(
        int(layout["rho_time_feature_count"]), 2 * len(expanded_rho_modes)
    )
    if int(perturbation.density_rank) > len(new_rho_indices):
        raise ValueError(
            "VPFP density perturbation rank exceeds the expanded mode count"
        )
    density_modes = rng.choice(
        new_rho_indices,
        size=int(perturbation.density_rank),
        replace=False,
    )
    for sampled_mode in density_modes:
        mode_index = int(sampled_mode)
        values, slopes = _smooth_density_time_controls(
            rng, count, perturbation
        )
        amplitude = (
            float(perturbation.density_raw_scale)
            / math.sqrt(float(perturbation.density_rank))
            / latent_scale
        )
        rho[:count, 2 * mode_index : 2 * mode_index + 2] += torch.as_tensor(
            amplitude * values,
            dtype=dtype,
            device=device,
        )
        rho[count:, 2 * mode_index : 2 * mode_index + 2] += torch.as_tensor(
            amplitude * slopes,
            dtype=dtype,
            device=device,
        )

    global_width = int(layout["kinetic_global_latent_dim"])
    local_start = int(layout["rho_latent_dim"]) + global_width
    expanded_modes = total_degree_mode_pairs(expanded.hermite_total_degree)
    base_modes = set(total_degree_mode_pairs(base.hermite_total_degree))
    new_channels = [
        index for index, mode in enumerate(expanded_modes) if mode not in base_modes
    ]
    expanded_features = _local_spatial_feature_labels(
        expanded.structured_local_spatial_mode
    )
    base_features = set(
        _local_spatial_feature_labels(base.structured_local_spatial_mode)
    )
    new_features = [
        index for index, label in enumerate(expanded_features) if label not in base_features
    ]
    if not new_channels or not new_features:
        raise ValueError("VPFP perturbation has no expanded kinetic directions")
    local = delta[local_start:].reshape(
        len(expanded_modes), time_width, len(expanded_features)
    )
    for rank_index in range(int(perturbation.kinetic_rank)):
        values, slopes = _smooth_time_controls(rng, count, perturbation)
        if rank_index % 2 == 0:
            channel = int(rng.choice(new_channels))
            feature = int(rng.integers(0, len(expanded_features)))
        else:
            channel = int(rng.integers(0, len(expanded_modes)))
            feature = int(rng.choice(new_features))
        amplitude = (
            float(perturbation.kinetic_raw_scale)
            * float(rng.choice((-1.0, 1.0)))
            / math.sqrt(float(perturbation.kinetic_rank))
            / latent_scale
        )
        local[channel, :count, feature] += torch.as_tensor(
            amplitude * values, dtype=dtype, device=device
        )
        local[channel, count:, feature] += torch.as_tensor(
            amplitude * slopes, dtype=dtype, device=device
        )
    return delta


def calibrate_vpfp_robust_perturbation(
    base_model: Any,
    base_latent: Any,
    *,
    equation_condition: Any,
    seed: int,
    perturbation: VPFPRobustPerturbationConfig | None = None,
) -> VPFPRobustPerturbationResult:
    """Embed, perturb, fully re-evaluate the operator, and backtrack."""

    cfg = perturbation or VPFPRobustPerturbationConfig()
    require_analytic_structured_source(base_model)
    expanded_config = expanded_vpfp_perturbation_config(base_model.config, cfg)
    parameter = next(base_model.parameters())
    model = build_deterministic_direct_model(
        expanded_config, device=parameter.device
    )
    embedded = embed_vpfp_structured_latent(
        base_model.config, expanded_config, base_latent
    ).to(dtype=parameter.dtype, device=parameter.device)
    raw_delta = sample_vpfp_private_perturbation_latent(
        base_model.config,
        expanded_config,
        seed=int(seed),
        perturbation=cfg,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    attempts: list[dict[str, Any]] = []
    accepted_latent = None
    for backtrack in range(int(cfg.maximum_backtracks) + 1):
        scale = float(cfg.initial_scale) * float(cfg.backtrack_factor) ** backtrack
        candidate = embedded + scale * raw_delta
        audit = audit_vpfp_robust_perturbation(
            model,
            embedded,
            candidate,
            equation_condition=equation_condition,
            seed=int(seed) + 104729,
            perturbation=cfg,
        )
        audit["scale"] = scale
        audit["backtrack_index"] = backtrack
        attempts.append(audit)
        if audit["accepted"]:
            accepted_latent = candidate
            break
    selected = attempts[-1]
    report = {
        "schema": VPFP_ROBUST_PERTURBATION_SCHEMA,
        "accepted": accepted_latent is not None,
        "seed": int(seed),
        "config": asdict(cfg),
        "base_config": base_model.config.to_dict(),
        "expanded_config": expanded_config.to_dict(),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "selected_attempt": selected,
        "generator_parameters_modified": False,
        "base_answer_embedded_exactly": True,
        "hidden_parameterization_public": False,
        "positivity_mechanism": "bounded_Fourier_density_and_squared_Hermite_amplitude",
        "velocity_normalization": "exact_fullspace_Parseval_normalization",
        "initial_condition_preserved": True,
    }
    return VPFPRobustPerturbationResult(
        model=model,
        embedded_base_latent=embedded,
        perturbed_latent=accepted_latent,
        raw_perturbation_latent=raw_delta,
        report=report,
    )


def audit_vpfp_robust_perturbation(
    model: Any,
    base_latent: Any,
    perturbed_latent: Any,
    *,
    equation_condition: Any,
    seed: int,
    perturbation: VPFPRobustPerturbationConfig | None = None,
) -> dict[str, Any]:
    """Audit actual answer, macro, derivative, and nonlinear source changes."""

    cfg = perturbation or VPFPRobustPerturbationConfig()
    require_analytic_structured_source(model)
    rng = np.random.default_rng(int(seed))
    t0, t1 = (float(value) for value in model.config.time_domain)
    lx, ly = (float(value) for value in model.config.equation.x_domain)
    count = int(cfg.spacetime_probe_count)
    txy = np.column_stack(
        (
            rng.uniform(t0, t1, size=count),
            rng.uniform(0.0, lx, size=count),
            rng.uniform(0.0, ly, size=count),
        )
    )
    initial_txy = txy.copy()
    initial_txy[:, 0] = t0
    base_macro = evaluate_continuous_hermite_macro_points(
        model, base_latent, txy, equation_condition=equation_condition
    )
    perturbed_macro = evaluate_continuous_hermite_macro_points(
        model, perturbed_latent, txy, equation_condition=equation_condition
    )
    field_relative: dict[str, float] = {}
    numerator = 0.0
    denominator = 0.0
    for field in ("rho", "momentum", "kinetic_energy"):
        base = np.asarray(base_macro[field], dtype=np.float64)
        delta = np.asarray(perturbed_macro[field], dtype=np.float64) - base
        num = float(np.sum(np.square(delta)))
        den = float(np.sum(np.square(base)))
        field_relative[field] = math.sqrt(num / max(den, np.finfo(float).tiny))
        numerator += num
        denominator += den
    macro_relative = math.sqrt(
        numerator / max(denominator, np.finfo(float).tiny)
    )

    torch = model.torch
    parameter = next(model.parameters())
    spacetime = torch.as_tensor(
        txy, dtype=parameter.dtype, device=parameter.device
    )
    initial_spacetime = torch.as_tensor(
        initial_txy, dtype=parameter.dtype, device=parameter.device
    )
    with torch.no_grad():
        base_jet = analytic_structured_macro_jet(model, spacetime, base_latent)
        perturbed_jet = analytic_structured_macro_jet(
            model, spacetime, perturbed_latent
        )
        initial_base_jet = analytic_structured_macro_jet(
            model, initial_spacetime, base_latent
        )
        initial_perturbed_jet = analytic_structured_macro_jet(
            model, initial_spacetime, perturbed_latent
        )
    coefficient_delta = (
        perturbed_jet["coefficients"] - base_jet["coefficients"]
    )
    microscopic_rms = float(
        torch.sqrt(torch.mean(torch.sum(coefficient_delta.square(), dim=-1))).item()
    )
    duration = t1 - t0
    derivative_delta = (
        perturbed_jet["coefficient_derivatives"][0]
        - base_jet["coefficient_derivatives"][0]
    )
    temporal_derivative_rms = float(
        duration
        * torch.sqrt(torch.mean(torch.sum(derivative_delta.square(), dim=-1))).item()
    )
    base_mode_count = len(
        total_degree_mode_pairs(int(cfg.base_hermite_total_degree))
    )
    tail = perturbed_jet["coefficients"][:, base_mode_count:]
    expanded_energy = float(torch.mean(torch.sum(tail.square(), dim=-1)).item())

    phase_count = int(cfg.phase_probe_count)
    host_condition = equation_condition
    if hasattr(host_condition, "detach"):
        host_condition = host_condition.detach()
    if hasattr(host_condition, "cpu"):
        host_condition = host_condition.cpu()
    temperature = float(np.asarray(host_condition, dtype=float).reshape(-1)[1])
    phase_txy = np.column_stack(
        (
            rng.uniform(t0, t1, size=phase_count),
            rng.uniform(0.0, lx, size=phase_count),
            rng.uniform(0.0, ly, size=phase_count),
        )
    )
    velocity = rng.normal(
        0.0, math.sqrt(max(temperature, np.finfo(float).tiny)), size=(phase_count, 2)
    )
    phase = np.column_stack((phase_txy, velocity))
    with torch.no_grad():
        base_source = manufacture_standard_vpfp_source_analytic(
            model,
            phase,
            base_latent,
            equation_condition=equation_condition,
        )["source"]
        perturbed_source = manufacture_standard_vpfp_source_analytic(
            model,
            phase,
            perturbed_latent,
            equation_condition=equation_condition,
        )["source"]
    source_delta = perturbed_source - base_source
    source_scale = max(
        float(torch.linalg.vector_norm(base_source).item()),
        float(torch.linalg.vector_norm(perturbed_source).item()),
        torch.finfo(base_source.dtype).tiny,
    )
    source_increment = float(
        torch.linalg.vector_norm(source_delta).item() / source_scale
    )

    initial_base_macro = evaluate_continuous_hermite_macro_points(
        model, base_latent, initial_txy, equation_condition=equation_condition
    )
    initial_perturbed_macro = evaluate_continuous_hermite_macro_points(
        model, perturbed_latent, initial_txy, equation_condition=equation_condition
    )
    initial_num = sum(
        float(
            np.sum(
                np.square(
                    np.asarray(initial_perturbed_macro[field])
                    - np.asarray(initial_base_macro[field])
                )
            )
        )
        for field in ("rho", "momentum", "kinetic_energy")
    )
    initial_den = sum(
        float(np.sum(np.square(np.asarray(initial_base_macro[field]))))
        for field in ("rho", "momentum", "kinetic_energy")
    )
    initial_macro_relative = math.sqrt(
        initial_num / max(initial_den, np.finfo(float).tiny)
    )
    initial_coefficient_rms = float(
        torch.sqrt(
            torch.mean(
                torch.sum(
                    (
                        initial_perturbed_jet["coefficients"]
                        - initial_base_jet["coefficients"]
                    ).square(),
                    dim=-1,
                )
            )
        ).item()
    )
    rho_min = float(np.min(perturbed_macro["rho"]))
    rho_floor = (
        1.0 - float(model.config.rho_contrast)
    ) * float(model.config.equation.background_density)

    finite = all(
        math.isfinite(value)
        for value in (
            macro_relative,
            microscopic_rms,
            expanded_energy,
            temporal_derivative_rms,
            source_increment,
            initial_macro_relative,
            initial_coefficient_rms,
            rho_min,
        )
    )
    gates = {
        "finite": finite,
        "rho_positive_by_sample_and_analytic_bound": rho_min >= rho_floor - 1.0e-6,
        "macro_displacement_visible": macro_relative >= cfg.macro_relative_l2_minimum,
        "macro_displacement_bounded": macro_relative <= cfg.macro_relative_l2_maximum,
        "microscopic_displacement_visible": microscopic_rms
        >= cfg.microscopic_coefficient_rms_minimum,
        "microscopic_displacement_bounded": microscopic_rms
        <= cfg.microscopic_coefficient_rms_maximum,
        "expanded_hermite_energy_visible": expanded_energy
        >= cfg.expanded_hermite_energy_minimum,
        "source_increment_visible": source_increment
        >= cfg.source_increment_relative_minimum,
        "source_increment_bounded": source_increment
        <= cfg.source_increment_relative_maximum,
        "temporal_derivative_bounded": temporal_derivative_rms
        <= cfg.temporal_derivative_rms_maximum,
        "initial_macro_preserved": initial_macro_relative
        <= cfg.initial_macro_relative_tolerance,
        "initial_coefficients_preserved": initial_coefficient_rms
        <= cfg.initial_coefficient_rms_tolerance,
    }
    return {
        "schema": VPFP_ROBUST_PERTURBATION_AUDIT_SCHEMA,
        "accepted": all(gates.values()),
        "gates": gates,
        "metrics": {
            "macro_relative_l2": macro_relative,
            "macro_field_relative_l2": field_relative,
            "microscopic_coefficient_rms": microscopic_rms,
            "expanded_hermite_energy_fraction": expanded_energy,
            "duration_scaled_temporal_derivative_rms": temporal_derivative_rms,
            "source_increment_relative_l2": source_increment,
            "initial_macro_relative_l2": initial_macro_relative,
            "initial_coefficient_rms": initial_coefficient_rms,
            "rho_sample_minimum": rho_min,
            "rho_analytic_floor": rho_floor,
        },
        "velocity_normalization_exact_not_quadrature_repaired": True,
        "source_recomputed_after_perturbation": True,
        "source_derivative_backend": "structured_fourier_hermite_jet_v1",
    }


def write_vpfp_direct_decoder_checkpoint(
    source_checkpoint: str | Path,
    destination: str | Path,
    *,
    model: Any,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the deterministic expanded decoder used to define hidden truth."""

    import torch

    source = Path(source_checkpoint).resolve()
    target = Path(destination).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") != "vpfp_continuous_hermite_autoencoder_checkpoint_v1":
        raise ValueError("unexpected VPFP decoder checkpoint schema")
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload["decoder_config"] = model.config.to_dict()
    payload["decoder_state_dict"] = state
    payload["case_latents"] = torch.empty((0, int(model.config.latent_dim)))
    payload["case_ids"] = []
    payload["train_indices"] = []
    payload["validation_indices"] = []
    payload["optimizer_state_dict"] = None
    payload["robust_answer_first_adapter"] = {
        "schema": "vpfp_robust_answer_first_decoder_adapter_v2",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": _sha256(source),
        "unused_neural_parameters_zeroed": True,
        "continuous_hidden_answer_is_primary": True,
        "generator_parameterization_public": False,
        **dict(metadata or {}),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def _smooth_time_controls(
    rng: np.random.Generator,
    count: int,
    config: VPFPRobustPerturbationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tau = np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    frequency = int(rng.integers(1, int(config.maximum_temporal_frequency) + 1))
    phase = float(rng.uniform(-math.pi, math.pi))
    omega = math.pi * float(frequency)
    carrier = np.sin(omega * tau + phase)
    envelope = tau**2
    values = envelope * carrier
    slopes = 2.0 * tau * carrier + envelope * omega * np.cos(omega * tau + phase)
    scale = max(float(np.sqrt(np.mean(values**2))), np.finfo(float).tiny)
    values /= scale
    slopes /= scale
    values[0] = 0.0
    slopes[0] = 0.0
    return values, slopes


def _smooth_density_time_controls(
    rng: np.random.Generator,
    count: int,
    config: VPFPRobustPerturbationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exactly cubic, internally nonzero random complex-mode path.

    Each density Fourier mode is one two-coordinate positivity group.  A scalar
    carrier times a fixed phase crosses the origin whenever the carrier changes
    sign, making the group norm (and hence the density gauge) nonsmooth.  The
    polynomial path below has the amplitude/phase form

        tau**2 R(phi) (1 + radial_skew*tau, angular_skew*tau),

    whose first component before rotation is strictly positive for tau in
    (0, 1].  Both coordinates are global cubics, so piecewise cubic Hermite
    reconstruction is exact rather than merely sampled away from zero.
    """

    del config  # Kept in the helper ABI alongside the kinetic control sampler.
    tau = np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    radial_skew = float(rng.uniform(-0.45, 0.45))
    angular_skew = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.15, 0.75))
    phase = float(rng.uniform(-math.pi, math.pi))
    cosine = math.cos(phase)
    sine = math.sin(phase)

    local_x = tau**2 * (1.0 + radial_skew * tau)
    local_y = angular_skew * tau**3
    local_x_d = 2.0 * tau + 3.0 * radial_skew * tau**2
    local_y_d = 3.0 * angular_skew * tau**2
    values = np.column_stack(
        (cosine * local_x - sine * local_y, sine * local_x + cosine * local_y)
    )
    slopes = np.column_stack(
        (
            cosine * local_x_d - sine * local_y_d,
            sine * local_x_d + cosine * local_y_d,
        )
    )
    scale = max(
        float(np.sqrt(np.mean(np.sum(values**2, axis=-1)))),
        np.finfo(float).tiny,
    )
    values /= scale
    slopes /= scale
    values[0] = 0.0
    slopes[0] = 0.0
    return values, slopes


def _local_spatial_feature_labels(maximum_mode: int) -> tuple[tuple[Any, ...], ...]:
    labels: list[tuple[Any, ...]] = [("constant", 0, 0)]
    for kx, ky in _real_fourier_half_modes(maximum_mode, maximum_mode):
        labels.extend((("cos", int(kx), int(ky)), ("sin", int(kx), int(ky))))
    return tuple(labels)


def _coerce_config(
    value: VPFPContinuousHermiteConfig | dict[str, Any],
) -> VPFPContinuousHermiteConfig:
    return (
        value
        if isinstance(value, VPFPContinuousHermiteConfig)
        else VPFPContinuousHermiteConfig.from_dict(dict(value))
    )


def _require_direct_structured_config(config: VPFPContinuousHermiteConfig) -> None:
    config.validate()
    if (
        not config.structured_spectral_latent
        or not config.structured_direct_coefficients
        or float(config.structured_residual_scale) != 0.0
        or config.structured_time_basis != "piecewise_cubic_hermite_v1"
        or config.rho_positivity_gauge != "phase_invariant_group_l1_v2"
    ):
        raise ValueError("VPFP robust perturbation requires the direct structured Jet representation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
