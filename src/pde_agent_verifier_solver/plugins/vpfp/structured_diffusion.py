"""Conditional EDM on complete VPFP answer coefficients, not coarse solves.

The periodic field is a redundant *computational* view, never a noise space or
an autoencoder. Gaussian noise, EDM preconditioning and losses live in the
original active Fourier/Hermite coefficients. The physical positive decoder
and its induced MMS source remain outside this learned distribution module.
Coefficient fidelity does not establish physical-answer fidelity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VPFPStructuredEDMConfig:
    control_count: int = 33
    hermite_channels: int = 120
    rho_max_mode: int = 6
    kinetic_max_mode: int = 4
    inactive_dim: int = 64
    grid_size: int = 13
    width: int = 256
    depth: int = 4
    temporal_dilation_cycle: tuple[int, ...] = (1, 2, 4, 9)
    collision_nu_range: tuple[float, float] = (0.02, 0.08)
    temperature_range: tuple[float, float] = (0.8, 1.2)
    time_domain: tuple[float, float] = (0.0, 1.0)
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 20.0
    rho: float = 7.0
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.2
    macro_loss_fraction: float = 0.5
    schema: str = "vpfp_direct_full_coefficient_edm_v1"

    def __post_init__(self):
        for name in ("control_count", "hermite_channels", "rho_max_mode", "grid_size", "width", "depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("kinetic_max_mode", "inactive_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.control_count < 2:
            raise ValueError("piecewise Hermite needs at least two controls")
        if self.grid_size < 2 * max(self.rho_max_mode, self.kinetic_max_mode) + 1:
            raise ValueError("periodic grid aliases the declared Fourier modes")
        if self.width < self.field_channels:
            raise ValueError("network width must preserve full field-channel output rank")
        dilations = tuple(self.temporal_dilation_cycle)
        if not dilations or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dilations):
            raise ValueError("temporal dilations must be positive integers")
        object.__setattr__(self, "temporal_dilation_cycle", dilations)
        for name in ("collision_nu_range", "temperature_range", "time_domain"):
            bounds = tuple(float(value) for value in getattr(self, name))
            if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"invalid {name}")
            if name != "time_domain" and bounds[0] <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, bounds)
        for name in ("sigma_data", "sigma_min", "sigma_max", "rho", "log_sigma_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if self.sigma_max <= self.sigma_min or not math.isfinite(self.log_sigma_mean):
            raise ValueError("invalid EDM noise distribution")
        if not 0 < self.macro_loss_fraction < 1:
            raise ValueError("macro_loss_fraction must lie in (0,1)")
        if self.schema != "vpfp_direct_full_coefficient_edm_v1":
            raise ValueError("unknown VPFP structured EDM schema")

    @property
    def rho_modes(self):
        return (2 * self.rho_max_mode + 1) ** 2 - 1

    @property
    def kinetic_modes(self):
        return (2 * self.kinetic_max_mode + 1) ** 2

    @property
    def rho_dimension(self):
        return 2 * self.control_count * self.rho_modes

    @property
    def kinetic_dimension(self):
        return self.hermite_channels * 2 * self.control_count * self.kinetic_modes

    @property
    def latent_dimension(self):
        return self.rho_dimension + self.inactive_dim + self.kinetic_dimension

    @property
    def field_channels(self):
        return 2 * (self.hermite_channels + 1)

    @property
    def slope_interval(self):
        # Stored slopes are d/dtau, tau=(t-t0)/(t1-t0), NOT d/dt.
        return 1.0 / (self.control_count - 1)

    @property
    def convolutional_time_receptive_field(self):
        # Local convolutional span; GroupNorm and condition embeddings are
        # separate. Boundary positions still see zero padding, not future wrap.
        return 1 + 2 * sum(self.temporal_dilation_cycle[index % len(self.temporal_dilation_cycle)] for index in range(self.depth))

    @property
    def sigma_bin_edges(self):
        return tuple(self.sigma_min * (self.sigma_max / self.sigma_min) ** (i / 4) for i in range(5))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**dict(payload))


def _coefficients(values, config, *, device=None, dtype=None):
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.ndim != 2 or tensor.shape[1] != config.latent_dimension or not tensor.is_floating_point():
        raise ValueError("expected floating complete coefficients with shape [B,D]")
    if not torch.isfinite(tensor).all():
        raise ValueError("VPFP coefficients must be finite")
    start = config.rho_dimension
    if torch.count_nonzero(tensor[:, start:start + config.inactive_dim]):
        raise ValueError("inactive compatibility coefficients must be exactly zero")
    return tensor


def _split(values, config):
    batch = values.shape[0]
    rho = values[:, :config.rho_dimension].reshape(batch, 2, config.control_count, config.rho_modes)
    kinetic = values[:, config.rho_dimension + config.inactive_dim:].reshape(
        batch, config.hermite_channels, 2, config.control_count, config.kinetic_modes)
    return rho, kinetic


def _join(rho, kinetic, config):
    return torch.cat((rho.flatten(1), rho.new_zeros((rho.shape[0], config.inactive_dim)), kinetic.flatten(1)), dim=1)


class VPFPCoefficientNormalizer(nn.Module):
    """Train-only, translation-pair-preserving group RMS scaling, no centering.

    Values and interval-scaled d/dtau slopes have separate scales. Rho uses
    two groups; each kinetic Hermite channel uses two. No empirical near-zero
    coordinate variance is inverted. No physical/gauge qualification is made.
    """

    def __init__(self, config, group_scales, *, fit_metadata: Mapping):
        super().__init__()
        self.config = config
        scales = torch.as_tensor(group_scales, dtype=torch.float64).detach()
        if scales.shape != (1 + config.hermite_channels, 2) or not torch.isfinite(scales).all() or torch.any(scales <= 0):
            raise ValueError("invalid VPFP normalization group scales")
        if fit_metadata.get("split") != "train" or fit_metadata.get("generated_samples_used_for_fit") is not False:
            raise ValueError("normalization must be fitted on training answers only")
        self.register_buffer("group_scales", scales.clone())
        self.fit_metadata = dict(fit_metadata)

    @classmethod
    def fit(cls, config, training_latents, *, sample_ids: Sequence[int], split="train",
            relative_floor=0.05, absolute_floor=1.e-6):
        if split != "train":
            raise ValueError("normalization only accepts the train split")
        if not math.isfinite(relative_floor) or not 0 < relative_floor <= 1 or not math.isfinite(absolute_floor) or absolute_floor <= 0:
            raise ValueError("normalization floors must be positive and bounded")
        values = _coefficients(training_latents, config)
        ids = _seeds(sample_ids)
        if len(values) < 2 or len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("normalization needs >=2 distinct training sample identities")
        rho, kinetic = _split(values, config)
        interval = values.new_tensor((1., config.slope_interval), dtype=torch.float64)
        rho_rms = torch.sqrt(rho.to(torch.float64).square().mean(dim=(0, 2, 3))) * interval
        kinetic_rms = torch.sqrt(kinetic.to(torch.float64).square().mean(dim=(0, 3, 4))) * interval
        rho_floor = max(float(absolute_floor), float(relative_floor) * float(rho_rms.square().mean().sqrt()))
        kinetic_floor = max(float(absolute_floor), float(relative_floor) * float(kinetic_rms.square().mean().sqrt()))
        scales = torch.cat((rho_rms.clamp_min(rho_floor)[None], kinetic_rms.clamp_min(kinetic_floor)), dim=0)
        return cls(config, scales, fit_metadata={
            "schema": "vpfp_train_group_rms_normalization_v1", "split": "train",
            "sample_ids": list(ids), "generated_samples_used_for_fit": False,
            "centering": "none", "scale": "group_rms_with_block_relative_floor",
            "relative_floor": float(relative_floor), "absolute_floor": float(absolute_floor),
            "rho_floor": rho_floor, "kinetic_floor": kinetic_floor,
            "slope_transform": "d_d_tau_times_delta_tau", "delta_tau": config.slope_interval,
            "inactive_policy": "exact_zero", "physical_answer_fidelity_certified": False,
        })

    def _transform(self, values, inverse):
        values = _coefficients(values, self.config)
        rho, kinetic = _split(values, self.config)
        scales = self.group_scales.to(values)
        interval = values.new_tensor((1., self.config.slope_interval))
        rho_factor = interval / scales[0]
        kinetic_factor = interval / scales[1:]
        if inverse:
            rho_factor, kinetic_factor = rho_factor.reciprocal(), kinetic_factor.reciprocal()
        return _join(rho * rho_factor[None, :, None, None],
                     kinetic * kinetic_factor[None, :, :, None, None], self.config)

    def encode(self, coefficients):
        return self._transform(coefficients, False)

    def decode(self, normalized_coefficients):
        return self._transform(normalized_coefficients, True)

    def to_payload(self):
        return {"config": self.config.to_dict(), "group_scales": self.group_scales.detach().cpu().tolist(),
                "fit_metadata": dict(self.fit_metadata)}

    @classmethod
    def from_payload(cls, payload):
        return cls(VPFPStructuredEDMConfig.from_dict(payload["config"]), payload["group_scales"],
                   fit_metadata=payload["fit_metadata"])


def _fourier_basis(maximum_mode, grid_size, *, constant, normalized):
    coordinate = torch.arange(grid_size, dtype=torch.float64) * (2 * math.pi / grid_size)
    x, y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    rows = [torch.ones_like(x)] if constant else []
    # Exactly continuous_v2._real_fourier_half_modes ordering.
    for kx in range(-maximum_mode, maximum_mode + 1):
        for ky in range(-maximum_mode, maximum_mode + 1):
            if kx > 0 or (kx == 0 and ky > 0):
                phase = kx * x + ky * y
                factor = math.sqrt(2.) if normalized else 1.
                rows.extend((factor * torch.cos(phase), factor * torch.sin(phase)))
    return torch.stack(rows)


class VPFPFourierCoefficientTransform(nn.Module):
    """Exact-supported Fourier view; rho lacks DC, kinetic includes DC.

    Output channels: rho value/slope, then Hermite0 value/slope, etc. Temporal
    controls are an explicit nonperiodic axis. Analysis discards only grid
    modes outside the declared coefficient space, never declared coefficients.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("rho_basis", _fourier_basis(config.rho_max_mode, config.grid_size, constant=False, normalized=False))
        self.register_buffer("kinetic_basis", _fourier_basis(config.kinetic_max_mode, config.grid_size, constant=True, normalized=True))

    def to_field(self, coefficients):
        values = _coefficients(coefficients, self.config)
        rho, kinetic = _split(values, self.config)
        rho_field = torch.einsum("btnm,mxy->btnxy", rho, self.rho_basis.to(values))
        kinetic_field = torch.einsum("bctnm,mxy->bctnxy", kinetic, self.kinetic_basis.to(values))
        return torch.cat((rho_field, kinetic_field.flatten(1, 2)), dim=1)

    def to_coefficients(self, field):
        cfg = self.config
        if field.ndim != 5 or tuple(field.shape[1:]) != (cfg.field_channels, cfg.control_count, cfg.grid_size, cfg.grid_size):
            raise ValueError("wrong VPFP coefficient-field shape")
        if not field.is_floating_point() or not torch.isfinite(field).all():
            raise ValueError("VPFP coefficient field must be finite")
        rho = torch.einsum("btnxy,mxy->btnm", field[:, :2], self.rho_basis.to(field)) * (2. / cfg.grid_size ** 2)
        kinetic_field = field[:, 2:].reshape(field.shape[0], cfg.hermite_channels, 2, cfg.control_count, cfg.grid_size, cfg.grid_size)
        kinetic = torch.einsum("bctnxy,mxy->bctnm", kinetic_field, self.kinetic_basis.to(field)) / cfg.grid_size ** 2
        return _join(rho, kinetic, cfg)


class _SpatiotemporalBlock(nn.Module):
    def __init__(self, width, temporal_dilation=1):
        super().__init__()
        self.temporal_dilation = temporal_dilation
        groups = max(divisor for divisor in range(1, min(16, width) + 1) if width % divisor == 0)
        self.norm = nn.GroupNorm(groups, width)
        self.film = nn.Linear(width, 2 * width)
        self.depthwise = nn.Conv3d(width, width, kernel_size=3, groups=width,
                                   dilation=(temporal_dilation, 1, 1))
        self.mix = nn.Sequential(nn.Conv3d(width, 2 * width, 1), nn.SiLU(), nn.Conv3d(2 * width, width, 1))

    def forward(self, values, embedding):
        scale, shift = self.film(embedding).chunk(2, dim=1)
        hidden = self.norm(values) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        # Only x/y wrap; physical time/control index must not wrap.
        hidden = F.pad(F.silu(hidden), (1, 1, 1, 1, 0, 0), mode="circular")
        hidden = F.pad(hidden, (0, 0, 0, 0, self.temporal_dilation, self.temporal_dilation), mode="constant", value=0.)
        return values + self.mix(self.depthwise(hidden))


class VPFPStructuredEDM(nn.Module):
    """Joint full-coefficient denoiser; accepts normalized coefficients."""

    def __init__(self, config: VPFPStructuredEDMConfig):
        super().__init__()
        self.config = config
        self.transform = VPFPFourierCoefficientTransform(config)
        self.input = nn.Conv3d(config.field_channels, config.width, 1)
        self.context = nn.Sequential(nn.Linear(18, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.time_position = nn.Linear(3, config.width)
        self.blocks = nn.ModuleList(_SpatiotemporalBlock(config.width,
            config.temporal_dilation_cycle[index % len(config.temporal_dilation_cycle)]) for index in range(config.depth))
        self.output = nn.Conv3d(config.width, config.field_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _conditions(self, conditions, count, prototype):
        condition = torch.as_tensor(conditions, dtype=prototype.dtype, device=prototype.device)
        if condition.shape != (count, 2) or not torch.isfinite(condition).all():
            raise ValueError("VPFP physical conditions must have shape [B,2] and be finite")
        bounds = condition.new_tensor((self.config.collision_nu_range, self.config.temperature_range))
        if torch.any(condition < bounds[:, 0]) or torch.any(condition > bounds[:, 1]):
            raise ValueError("VPFP conditions leave the declared regime")
        transformed = torch.stack((condition[:, 0].log(), condition[:, 1]), dim=1)
        lower = torch.stack((bounds[0, 0].log(), bounds[1, 0]))
        upper = torch.stack((bounds[0, 1].log(), bounds[1, 1]))
        return 2 * (transformed - lower) / (upper - lower) - 1

    def denoise(self, noisy_normalized_coefficients, sigma, conditions):
        parameter = next(self.parameters())
        noisy = _coefficients(noisy_normalized_coefficients, self.config, device=parameter.device, dtype=parameter.dtype)
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        if sigma.shape != (len(noisy),) or not torch.isfinite(sigma).all() or torch.any(sigma <= 0):
            raise ValueError("sigma must be finite and positive with shape [B]")
        condition = self._conditions(conditions, len(noisy), noisy)
        denominator = torch.sqrt(sigma.square() + self.config.sigma_data ** 2)
        c_skip = self.config.sigma_data ** 2 / denominator.square()
        c_out = sigma * self.config.sigma_data / denominator
        field = self.transform.to_field(noisy / denominator[:, None])
        frequencies = torch.exp(torch.linspace(0., math.log(100.), 8, device=noisy.device, dtype=noisy.dtype))
        phase = sigma.log()[:, None] * frequencies[None]
        embedding = self.context(torch.cat((phase.sin(), phase.cos(), condition), dim=1))
        tau = torch.linspace(0., 1., self.config.control_count, device=noisy.device, dtype=noisy.dtype)
        position = self.time_position(torch.stack((tau, (math.pi * tau).sin(), (math.pi * tau).cos()), dim=1))
        hidden = self.input(field) + position.T[None, :, :, None, None]
        for block in self.blocks:
            hidden = block(hidden, embedding)
        residual = self.transform.to_coefficients(self.output(F.silu(hidden)))
        return c_skip[:, None] * noisy + c_out[:, None] * residual

    def _noise(self, shape, *, generator, prototype):
        noise = torch.randn(shape, dtype=prototype.dtype, device=prototype.device, generator=generator)
        start = self.config.rho_dimension
        noise[:, start:start + self.config.inactive_dim] = 0.
        return noise

    def loss(self, clean_normalized_coefficients, conditions, *, generator=None, sigma=None, noise=None,
             residual_metric=None):
        parameter = next(self.parameters())
        clean = _coefficients(clean_normalized_coefficients, self.config, device=parameter.device, dtype=parameter.dtype)
        if not len(clean):
            raise ValueError("EDM loss needs at least one answer")
        if sigma is None:
            sigma = (torch.randn(len(clean), device=clean.device, dtype=clean.dtype, generator=generator)
                     * self.config.log_sigma_std + self.config.log_sigma_mean).exp().clamp(self.config.sigma_min, self.config.sigma_max)
        sigma = torch.as_tensor(sigma, device=clean.device, dtype=clean.dtype)
        if sigma.shape != (len(clean),) or not torch.isfinite(sigma).all() or torch.any(sigma < self.config.sigma_min) or torch.any(sigma > self.config.sigma_max):
            raise ValueError("training sigma leaves the registered interval")
        noise = self._noise(clean.shape, generator=generator, prototype=clean) if noise is None else _coefficients(noise, self.config, device=clean.device, dtype=clean.dtype)
        if noise.shape != clean.shape:
            raise ValueError("EDM noise batch shape differs")
        denoised = self.denoise(clean + sigma[:, None] * noise, sigma, conditions)
        error = (denoised - clean).square()
        weight = (sigma.square() + self.config.sigma_data ** 2) / (sigma * self.config.sigma_data).square()
        rho_loss = error[:, :self.config.rho_dimension].mean(dim=1) * weight
        kinetic_loss = error[:, self.config.rho_dimension + self.config.inactive_dim:].mean(dim=1) * weight
        per_case = self.config.macro_loss_fraction * rho_loss + (1 - self.config.macro_loss_fraction) * kinetic_loss
        base_per_case = per_case
        if residual_metric is not None and residual_metric.config != self.config:
            raise ValueError("residual metric config differs from EDM config")
        metric_enabled = residual_metric is not None and residual_metric.enabled
        sobolev = None
        if metric_enabled:
            sobolev = residual_metric.components(denoised - clean)
            rho_loss = sobolev["rho_total"] * weight
            kinetic_loss = sobolev["kinetic_total"] * weight
            per_case = self.config.macro_loss_fraction * rho_loss + (1 - self.config.macro_loss_fraction) * kinetic_loss
        # The paired zero-residual EDM is the posterior mean for a zero-mean
        # isotropic Gaussian with variance sigma_data**2, not a fitted VPFP
        # Gaussian. Keep its raw sigma-bin losses to detect unlearned noise
        # ranges; do not call an improvement here a distribution qualification.
        with torch.no_grad():
            skip = self.config.sigma_data ** 2 / (sigma.square() + self.config.sigma_data ** 2)
            baseline_error = (skip[:, None] * (clean + sigma[:, None] * noise) - clean).square()
            baseline_rho = baseline_error[:, :self.config.rho_dimension].mean(dim=1) * weight
            baseline_kinetic = baseline_error[:, self.config.rho_dimension + self.config.inactive_dim:].mean(dim=1) * weight
            baseline = self.config.macro_loss_fraction * baseline_rho + (1 - self.config.macro_loss_fraction) * baseline_kinetic
            base_baseline = baseline
            if metric_enabled:
                baseline_sobolev = residual_metric.components(skip[:, None] * (clean + sigma[:, None] * noise) - clean)
                baseline_rho = baseline_sobolev["rho_total"] * weight
                baseline_kinetic = baseline_sobolev["kinetic_total"] * weight
                baseline = self.config.macro_loss_fraction * baseline_rho + (1 - self.config.macro_loss_fraction) * baseline_kinetic
        bins = torch.bucketize(sigma.detach(), sigma.new_tensor(self.config.sigma_bin_edges[1:-1]))
        metrics = {"loss_per_case": per_case.detach(), "rho_loss_per_case": rho_loss.detach(),
                   "kinetic_loss_per_case": kinetic_loss.detach(), "sigma": sigma.detach(),
                   "zero_mean_isotropic_baseline_per_case": baseline,
                   "base_loss_per_case": base_per_case.detach(),
                   "base_isotropic_baseline_per_case": base_baseline,
                   "sigma_bin_counts": torch.bincount(bins, minlength=4),
                   "sigma_bin_isotropic_baseline_sums": torch.stack([baseline[bins == index].sum() for index in range(4)]),
                   "sigma_bin_loss_sums": torch.stack([per_case.detach()[bins == index].sum() for index in range(4)]),
                   "sigma_bin_rho_loss_sums": torch.stack([rho_loss.detach()[bins == index].sum() for index in range(4)]),
                   "sigma_bin_kinetic_loss_sums": torch.stack([kinetic_loss.detach()[bins == index].sum() for index in range(4)])}
        if metric_enabled:
            metrics["sobolev_spatial_loss_per_case"] = (sobolev["spatial"] * weight).detach()
            metrics["sobolev_temporal_loss_per_case"] = (sobolev["temporal"] * weight).detach()
        return per_case.mean(), metrics

    @torch.no_grad()
    def sample(self, conditions, seeds: Sequence[int], *, steps=64, churn=0., churn_min=0., churn_max=float("inf"), noise_scale=1.):
        seed_values = _seeds(seeds)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("Heun sampling needs at least two steps")
        if not math.isfinite(churn) or churn < 0 or not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError("invalid EDM churn/noise scale")
        if not math.isfinite(churn_min) or churn_min < 0 or math.isnan(churn_max) or churn_max < churn_min:
            raise ValueError("invalid EDM churn interval")
        parameter = next(self.parameters())
        conditions = torch.as_tensor(conditions, device=parameter.device, dtype=parameter.dtype)
        self._conditions(conditions, len(seed_values), parameter)
        ramp = torch.linspace(0., 1., steps, device=parameter.device, dtype=parameter.dtype)
        cfg = self.config
        sigmas = (cfg.sigma_max ** (1 / cfg.rho) + ramp * (cfg.sigma_min ** (1 / cfg.rho) - cfg.sigma_max ** (1 / cfg.rho))) ** cfg.rho
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        rows = []
        # Per-case streams deliberately preserve grouping/order invariance.
        for condition, seed in zip(conditions, seed_values):
            random = torch.Generator(device=parameter.device).manual_seed(seed)
            values = self._noise((1, cfg.latent_dimension), generator=random, prototype=parameter) * sigmas[0]
            for index in range(steps):
                current, following = sigmas[index], sigmas[index + 1]
                gamma = min(churn / steps, math.sqrt(2.) - 1.) if churn_min <= float(current) <= churn_max else 0.
                increased = current * (1 + gamma)
                if gamma:
                    values = values + (increased.square() - current.square()).sqrt() * noise_scale * self._noise(values.shape, generator=random, prototype=values)
                clean = self.denoise(values, increased[None], condition[None])
                derivative = (values - clean) / increased
                provisional = values + (following - increased) * derivative
                if index < steps - 1:
                    next_clean = self.denoise(provisional, following[None], condition[None])
                    next_derivative = (provisional - next_clean) / following
                    values = values + (following - increased) * 0.5 * (derivative + next_derivative)
                else:
                    values = provisional
            rows.append(_coefficients(values, cfg))
        return torch.cat(rows, dim=0)


def _seeds(seeds):
    values = tuple(seeds)
    if not values or any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2 ** 63 for seed in values):
        raise ValueError("sample seeds must be nonempty nonnegative signed-64-bit integers")
    return tuple(int(seed) for seed in values)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


class VPFPDiffusionBatchSampler:
    """Canonical private sampler ABI, with no legacy/PDE runtime imports."""

    def __init__(self, model: VPFPStructuredEDM, normalizer: VPFPCoefficientNormalizer, *,
                 base_decoder_sha256: str, checkpoint_sha256: str, steps=64, churn=0.):
        if model.config != normalizer.config:
            raise ValueError("VPFP model and normalizer layouts/configurations differ")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2 or not math.isfinite(churn) or churn < 0:
            raise ValueError("invalid frozen sampling settings")
        self.model, self.normalizer = model, normalizer
        self.steps, self.churn = steps, float(churn)
        config = model.config
        self.identity = {
            "kind": "direct_full_coefficient_conditional_edm", "schema": config.schema,
            "base_decoder_sha256": _sha(base_decoder_sha256, "base decoder identity"),
            "checkpoint_sha256": _sha(checkpoint_sha256, "diffusion checkpoint identity"),
            "collision_nu_range": list(config.collision_nu_range), "temperature_range": list(config.temperature_range),
            "time_domain": list(config.time_domain), "latent_dimension": config.latent_dimension,
            "model_config": config.to_dict(), "normalization_fit": dict(normalizer.fit_metadata),
            "sampling": {"method": "heun", "steps": steps, "churn": self.churn, "noise_space": "complete_active_coefficients",
                         "condition_prior": "uniform_physical_nu_and_temperature", "seed_stream": "sha256_domain_separated_per_case_v1"},
            "learned_generation": True, "pde_solves_per_sample": 0,
            "qualification": "unqualified_research_checkpoint", "experimental_data_validation": False,
            "answer_first": {"primary": "complete_continuous_answer_coefficients", "source": "derive_exact_MMS_after_sampling"},
            "coefficient_fidelity_is_not_physical_fidelity": True,
        }

    def generate(self, seeds):
        seed_values = _seeds(seeds)
        cfg = self.model.config
        conditions, noise_seeds = [], []
        for seed in seed_values:
            condition_seed = int.from_bytes(sha256(f"vpfp-edm-condition:{seed}".encode()).digest()[:8], "little")
            noise_seed = int.from_bytes(sha256(f"vpfp-edm-noise:{seed}".encode()).digest()[:8], "little") % (2 ** 63)
            random = np.random.default_rng(condition_seed)
            conditions.append((random.uniform(*cfg.collision_nu_range), random.uniform(*cfg.temperature_range)))
            noise_seeds.append(noise_seed)
        condition_array = np.asarray(conditions, dtype=np.float32)
        normalized = self.model.sample(condition_array, noise_seeds, steps=self.steps, churn=self.churn)
        latents = self.normalizer.decode(normalized).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(latents).all():
            raise FloatingPointError("sampled VPFP answer coefficients are nonfinite")
        return {"schema": "vpfp_learned_answer_batch_v1", "latents": latents,
                "equation_conditions": condition_array, "equation_seeds": np.asarray(seed_values, dtype=np.int64),
                "diagnostics": {"pde_solves": 0, "learned_generation": True,
                                "scientific_qualification": "not_established_by_sampling"}}


def load_vpfp_diffusion_sampler(checkpoint, expected_sha256, device="cpu", *, steps=64, churn=0.):
    """Load a tensor-only research checkpoint without a checksum pass.

    Corpus provenance is a binding, not a statement that its statistics have
    been qualified. The caller separately verifies any external corpus itself.
    No physical runtime, PDE generation, training or scientific audit occurs.
    """
    expected_sha256 = _sha(expected_sha256, "expected checkpoint identity")
    path = Path(checkpoint).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("diffusion checkpoint must be a regular non-symlink file")
    with path.open("rb") as source:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("schema") == "vpfp_value_control_diffusion_checkpoint_v1":
        from .value_control_diffusion import _sampler_from_value_payload
        return _sampler_from_value_payload(payload, expected_sha256, device, steps=steps, churn=churn)
    if isinstance(payload, dict) and payload.get("schema") == "vpfp_temporal_covariance_diffusion_checkpoint_v1":
        from .temporal_covariance_diffusion import _sampler_from_temporal_payload
        return _sampler_from_temporal_payload(payload, expected_sha256, device, steps=steps, churn=churn)
    if isinstance(payload, dict) and payload.get("schema") in (
            "vpfp_lowmode_coherent_diffusion_checkpoint_v1", "vpfp_branch_identity_control_diffusion_checkpoint_v1"):
        from .structure_prior_diffusion import _sampler_from_structure_payload
        return _sampler_from_structure_payload(payload, expected_sha256, device, steps=steps, churn=churn)
    if isinstance(payload, dict) and payload.get("schema") == "vpfp_initial_conditioned_future_diffusion_checkpoint_v1":
        from .initial_conditioned_diffusion import _sampler_from_initial_conditioned_payload
        return _sampler_from_initial_conditioned_payload(payload, expected_sha256, device, steps=steps, churn=churn)
    expected_keys = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != expected_keys or payload.get("schema") != "vpfp_structured_diffusion_checkpoint_v1":
        raise ValueError("unexpected VPFP diffusion checkpoint schema/fields")
    base_sha = _sha(payload["base_decoder_sha256"], "base decoder identity")
    corpus_sha = _sha(payload["corpus_manifest_sha256"], "corpus manifest identity")
    training = payload["training"]
    if not isinstance(training, dict) or not training:
        raise ValueError("diffusion checkpoint needs nonempty training provenance")
    try:
        json.dumps(training, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("training provenance must be finite JSON data") from error
    config = VPFPStructuredEDMConfig.from_dict(payload["config"])
    normalizer = VPFPCoefficientNormalizer.from_payload(payload["normalizer"])
    if config != normalizer.config:
        raise ValueError("checkpoint model and normalizer configurations differ")
    metadata = normalizer.fit_metadata
    ids = _seeds(metadata.get("sample_ids", ()))
    if (metadata.get("schema") != "vpfp_train_group_rms_normalization_v1"
            or len(ids) < 2 or len(set(ids)) != len(ids)
            or metadata.get("centering") != "none"
            or metadata.get("inactive_policy") != "exact_zero"
            or metadata.get("slope_transform") != "d_d_tau_times_delta_tau"
            or metadata.get("delta_tau") != config.slope_interval):
        raise ValueError("normalization provenance/layout is invalid")
    model = VPFPStructuredEDM(config)
    state = payload["model"]
    expected_state = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected_state):
        raise ValueError("diffusion checkpoint model tensor keys differ")
    for name, expected in expected_state.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != expected.shape or tensor.dtype != expected.dtype
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"invalid diffusion checkpoint tensor: {name}")
        # The Fourier basis is an immutable representation, not learned state.
        if name.startswith("transform.") and not torch.equal(tensor, expected):
            raise ValueError(f"checkpoint modified the fixed Fourier transform: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    normalizer.to(device=device)
    sampler = VPFPDiffusionBatchSampler(model, normalizer, base_decoder_sha256=base_sha,
        checkpoint_sha256=expected_sha256, steps=steps, churn=churn)
    sampler.identity.update(corpus_manifest_sha256=corpus_sha, training=dict(training))
    return sampler
