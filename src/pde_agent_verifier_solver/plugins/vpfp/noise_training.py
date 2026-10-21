"""Small training-only VPFP noise-level coverage ablation.

The mixture changes how often noise levels are trained, not the corruption
kernel, EDM target/preconditioner, state coordinates, network, or sampler.
Every level with positive training density retains the same population
conditional-mean denoiser. This is not evidence of physical sample fidelity.
The default mixture retains the complete original sigma support.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from .structured_diffusion import _coefficients
from .value_control_diffusion import VPFPValueControlEDM


def _settings(config, low_probability, low_min, low_max):
    settings = {
        "sigma_min": config.sigma_min, "sigma_max": config.sigma_max,
        "log_sigma_mean": config.log_sigma_mean, "log_sigma_std": config.log_sigma_std,
        "low_probability": low_probability,
        "low_min": config.sigma_min if low_min is None else low_min, "low_max": low_max,
    }
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
        settings[name] = float(value)
    if not 0 < settings["sigma_min"] < settings["sigma_max"]:
        raise ValueError("sigma_min/max must be positive and increasing")
    if settings["log_sigma_std"] <= 0:
        raise ValueError("log_sigma_std must be positive")
    if not 0 <= settings["low_probability"] <= 1:
        raise ValueError("low_probability must lie in [0,1]")
    if not settings["sigma_min"] <= settings["low_min"] < settings["low_max"] <= settings["sigma_max"]:
        raise ValueError("noise band requires sigma_min <= low_min < low_max <= sigma_max")
    if math.log(settings["low_max"]) <= math.log(settings["low_min"]):
        raise ValueError("low noise log interval is not numerically distinguishable")
    return settings


def low_noise_mixture_report(config, *, low_probability=.5, low_min=None, low_max=.1):
    """Finite JSON metadata; trainer must save it with the selected weights."""
    settings = _settings(config, low_probability, low_min, low_max)
    p, lower, upper = settings["low_probability"], settings["sigma_min"], settings["sigma_max"]
    mean, std = settings["log_sigma_mean"], settings["log_sigma_std"]
    low_log_lower = math.log(settings["low_min"])
    low_log_width = math.log(settings["low_max"]) - low_log_lower

    def interior_cdf(value):
        normal = .5 * math.erfc(-(math.log(value) - mean) / (std*math.sqrt(2)))
        low = min(1., max(0., (math.log(value) - low_log_lower) / low_log_width))
        return (1-p)*normal + p*low

    # The end bins include the clamped lognormal's endpoint atoms. Interior
    # boundaries use the continuous CDF; no random draw enters this report.
    edges = [math.exp(math.log(lower) + (math.log(upper)-math.log(lower))*i/4) for i in range(5)]
    cumulative = [0.] + [interior_cdf(edge) for edge in edges[1:-1]] + [1.]
    return {"schema": "vpfp_training_low_sigma_mixture_v1", **settings,
        "mixture": "Bernoulli_per_case_low_log_uniform_else_original_clamped_lognormal",
        "low_component": "log_uniform_low_min_to_low_max",
        "low_min_policy": "sigma_min_default" if low_min is None else "explicit_band_lower_bound",
        "original_component": "original_lognormal_then_clamp_to_declared_interval",
        "sigma_bin_probabilities": [cumulative[i+1]-cumulative[i] for i in range(4)],
        "same_corruption_kernel_and_edm_target": True,
        "same_preconditioner_architecture_checkpoint_schema_and_reverse_sampler": True,
        "importance_reweighting": False,
        "full_original_sigma_support": p < 1 or (settings["low_min"] == lower and settings["low_max"] == upper),
        "population_optimum": "unchanged_conditional_mean_at_each_positive_density_sigma",
        "explicit_sigma_policy": "bypass_mixture_without_extra_random_draws",
        "pde_solves": 0, "physical_distribution_qualification": False}


def sample_low_noise_mixture(config, batch, *, generator, prototype,
                            low_probability=.5, low_min=None, low_max=.1):
    """Draw only sigma, on the prototype dtype/device and within its domain.

    A probability-zero ablation exactly preserves the original lognormal RNG
    stream. With explicit Generator, this has no other mutable external
    state; None follows the standard Torch default-generator convention.
    """
    settings = _settings(config, low_probability, low_min, low_max)
    if isinstance(batch, bool) or not isinstance(batch, Integral) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if not isinstance(prototype, torch.Tensor) or prototype.dtype not in (torch.float32, torch.float64):
        raise ValueError("prototype must be a float32/float64 tensor")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator or None")
    dtype, device = prototype.dtype, prototype.device
    bounds = torch.tensor([settings["sigma_min"], settings["sigma_max"], settings["low_min"], settings["low_max"]],
        dtype=dtype, device=device)
    if (not torch.isfinite(bounds).all() or torch.any(bounds <= 0)
            or not bool(bounds[2] < bounds[3]) or not bool(bounds[0] < bounds[1])):
        raise ValueError("noise bounds are not representable in the prototype dtype")
    normal_parameters = torch.tensor([settings["log_sigma_mean"], settings["log_sigma_std"]],
        dtype=dtype, device=device)
    if not torch.isfinite(normal_parameters).all() or normal_parameters[1] <= 0:
        raise ValueError("lognormal parameters are not representable in the prototype dtype")
    p = settings["low_probability"]
    # Keep this formula and RNG call identical to the original .loss when p=0.
    original = None
    if p < 1:
        original = (torch.randn(int(batch), device=device, dtype=dtype, generator=generator)
            * settings["log_sigma_std"] + settings["log_sigma_mean"]).exp().clamp(
                settings["sigma_min"], settings["sigma_max"])
        if not torch.isfinite(original).all():
            raise ValueError("lognormal sampling produced nonfinite sigma")
    if p == 0:
        return original
    log_lower = math.log(settings["low_min"])
    log_width = math.log(settings["low_max"]) - log_lower
    low = (log_lower + torch.rand(int(batch), device=device, dtype=dtype, generator=generator)*log_width).exp()
    # Exp/log roundoff must not leave the declared physical noise domain.
    low = low.clamp(settings["low_min"], settings["low_max"])
    if p == 1:
        return low
    choose_low = torch.rand(int(batch), device=device, dtype=dtype, generator=generator) < p
    return torch.where(choose_low, low, original)


class VPFPLowNoiseValueControlEDM(VPFPValueControlEDM):
    """Training-only sigma override; loaded inference is the original class."""

    def __init__(self, config, *, low_probability=.5, low_min=None, low_max=.1):
        report = low_noise_mixture_report(config, low_probability=low_probability, low_min=low_min, low_max=low_max)
        super().__init__(config)
        self.low_probability, self.low_max = report["low_probability"], report["low_max"]
        self.low_min = None if low_min is None else report["low_min"]

    def training_noise_report(self):
        return low_noise_mixture_report(self.config,
            low_probability=self.low_probability, low_min=self.low_min, low_max=self.low_max)

    def loss(self, clean_normalized_coefficients, conditions, *, generator=None,
             sigma=None, noise=None, residual_metric=None):
        if sigma is None:
            parameter = next(self.parameters())
            clean_normalized_coefficients = _coefficients(clean_normalized_coefficients, self.config,
                device=parameter.device, dtype=parameter.dtype)
            sigma = sample_low_noise_mixture(self.config, len(clean_normalized_coefficients),
                generator=generator, prototype=parameter,
                low_probability=self.low_probability, low_min=self.low_min, low_max=self.low_max)
        return super().loss(clean_normalized_coefficients, conditions, generator=generator,
            sigma=sigma, noise=noise, residual_metric=residual_metric)


__all__ = ["sample_low_noise_mixture", "low_noise_mixture_report", "VPFPLowNoiseValueControlEDM"]
