"""Opt-in research primitives for clean-prediction EDM source budgets.

Not connected to the shared trainer or deployed assets. A nonlinear source
penalty intentionally changes the learned distribution; it is not an exactly
equivalent score objective. Budgets are explicit original-unit experiment
inputs, not admission thresholds and not fitted from candidate samples.
The fast audit evaluator detaches its carrier and must never be used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .coherent_diffusion import VPFPMacroConditionalEDM
from .structured_diffusion import _coefficients
from .moving_frame_backend import manufacture_moving_frame_source
from .value_control_diffusion import extract_value_controls, lift_value_controls
from .innovation_diffusion import decode_complete_innovation_answer


@dataclass(frozen=True)
class SourceBudgetObjective:
    # No empirical defaults: a caller must register every training parameter.
    weight: float
    sigma_cutoff: float
    rms_budget: float
    point_budget: float
    schema: str = "vpfp_clean_prediction_source_budget_v1"

    def __post_init__(self):
        for name in ("weight", "sigma_cutoff", "rms_budget", "point_budget"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name != "weight" and value == 0)):
                raise ValueError(f"invalid explicit source objective {name}")
        if self.schema != "vpfp_clean_prediction_source_budget_v1":
            raise ValueError("unknown source objective schema")


def edm_clean_prediction_loss(model, clean, conditions, *, generator=None,
                              sigma=None, noise=None):
    """One noise draw and one denoiser call, with the existing block EDM loss.

Returns attached clean prediction for an explicitly separate auxiliary loss.
Tests compare both values and parameter gradients to model.loss at fixed noise.
No second denoising pass, old residual_metric, or hidden high-noise projection.
"""
    if not isinstance(model, VPFPMacroConditionalEDM):
        raise TypeError("requires the existing conditional VALUE/innovation EDM")
    p = next(model.parameters())
    cfg = model.config
    clean = _coefficients(clean, cfg, device=p.device, dtype=p.dtype)
    if not len(clean):
        raise ValueError("EDM loss needs at least one answer")
    if sigma is None:
        sigma = (torch.randn(len(clean), device=clean.device, dtype=clean.dtype,
            generator=generator)*cfg.log_sigma_std + cfg.log_sigma_mean).exp().clamp(
                cfg.sigma_min, cfg.sigma_max)
    sigma = torch.as_tensor(sigma, device=clean.device, dtype=clean.dtype)
    if (sigma.shape != (len(clean),) or not bool(torch.isfinite(sigma).all())
            or bool((sigma < cfg.sigma_min).any()) or bool((sigma > cfg.sigma_max).any())):
        raise ValueError("training sigma leaves the registered interval")
    noise = (model._noise(clean.shape, generator=generator, prototype=clean)
        if noise is None else _coefficients(noise, cfg, device=clean.device, dtype=clean.dtype))
    if noise.shape != clean.shape:
        raise ValueError("EDM noise batch shape differs")
    predicted = model.denoise(clean + sigma[:, None]*noise, sigma, conditions)
    error = (predicted-clean).square()
    weight = (sigma.square()+cfg.sigma_data**2)/(sigma*cfg.sigma_data).square()
    rho = error[:, :cfg.rho_dimension].mean(1)*weight
    kinetic = error[:, cfg.rho_dimension+cfg.inactive_dim:].mean(1)*weight
    per_case = cfg.macro_loss_fraction*rho+(1-cfg.macro_loss_fraction)*kinetic
    return per_case.mean(), {"prediction": predicted, "sigma": sigma,
        "loss_per_case": per_case, "rho_loss_per_case": rho,
        "kinetic_loss_per_case": kinetic}


def innovation_prediction_to_values(white, normalizer, transform, physical_macro):
    """Use the canonical v2 inverse; A expects VALUE z, not native FULL z."""
    full = decode_complete_innovation_answer(white, normalizer, transform,
                                            physical_macro=physical_macro)
    return extract_value_controls(full, normalizer.config, validate_slopes=False)


def validate_trainable_source_domain(decoder, points, carrier):
    """Coordinate differentiability alone does not certify carrier gradients.

In the group-L1 density gauge, constant-zero groups are safe for coordinate
derivatives but may be nonsmooth under trainable-coefficient perturbations.
Reject exact zeros when z is attached; do not insert epsilon or silently omit
the affected source/loss. Smooth nonzero groups retain the original formula.
"""
    decoder.validate_source_gradient_domain(points, carrier)
    if (carrier["z"].requires_grad and
            decoder.base_decoder.config.rho_positivity_gauge == "phase_invariant_group_l1_v2"):
        z = carrier["z"]
        if z.ndim == 1:
            z = z[None]
        _, raw, _ = decoder._native_rho_groups(points, lift_value_controls(z, decoder.value_config))
        if bool((torch.linalg.vector_norm(raw.detach(), dim=-1) == 0).any()):
            raise ValueError("trainable z at a zero density group: carrier source gradient not certified")


def source_budget_loss(decoder, points, carrier, condition, *, sigma, objective):
    """Single clean-prediction carrier, soft RMS/tail budgets in original units.

Only eligible low-sigma predictions invoke the differentiable source. The
caller averages over the ENTIRE original batch, not just eligible examples.
RMS is over the declared probe measure, not a certified full-space norm. A
separate pointwise excess term discourages hiding spikes in a mean. Neither
penalty matches a coarse trajectory or its prescribed forcing function.
"""
    if not isinstance(objective, SourceBudgetObjective):
        raise TypeError("an explicit versioned source objective is required")
    if set(carrier) != {"m", "z"} or any(v.ndim != 2 or len(v) != 1 for v in carrier.values()):
        raise ValueError("source loss requires one complete {m,z} VALUE carrier")
    if not all(bool(torch.isfinite(v).all()) for v in carrier.values()):
        raise ValueError("nonfinite clean carrier; no clipping or filtering")
    level = float(torch.as_tensor(sigma).detach())
    if not math.isfinite(level) or level <= 0:
        raise ValueError("positive finite scalar noise level required")
    zero = carrier["m"].sum()*0 + carrier["z"].sum()*0
    if objective.weight == 0 or level >= objective.sigma_cutoff:
        return zero, {"evaluated": False, "reason": "zero_weight" if objective.weight == 0
                      else "outside_low_sigma_gate", "sigma": level}
    points = torch.as_tensor(points, dtype=torch.float64,
                             device=next(decoder.parameters()).device)
    if points.ndim != 2 or points.shape[1] != 5 or not len(points) or not bool(torch.isfinite(points).all()):
        raise ValueError("finite nonempty [N,5] source probes required")
    validate_trainable_source_domain(decoder, points, carrier)
    fields = manufacture_moving_frame_source(decoder, points, carrier, condition, create_graph=True)
    source = fields["source"]
    rms = torch.linalg.vector_norm(source)/math.sqrt(source.numel())
    rms_excess = torch.relu(rms/objective.rms_budget-1).square()
    tail_excess = torch.relu(source.abs()/objective.point_budget-1).square().mean()
    gate = (1-(level/objective.sigma_cutoff)**2)**2
    loss = objective.weight*gate*(rms_excess+tail_excess)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite source loss; retain failed attempt")
    return loss, {"evaluated": True, "sigma": level, "gate": gate,
        "source_probe_rms": rms.detach(), "source_observed_abs_max": source.abs().max().detach(),
        "rms_excess": rms_excess.detach(), "point_excess": tail_excess.detach(),
        "loss": loss.detach(), "rms_budget": objective.rms_budget,
        "point_budget": objective.point_budget,
        "source_units": "original_equation_units", "physical_admission": False}
