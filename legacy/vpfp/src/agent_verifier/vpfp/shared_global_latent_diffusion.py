from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from agent_verifier.fusion_simulation_first.axial_latent_diffusion import (
    AxialEDMConfig,
    ConditionalAxialLatentEDM,
)
from agent_verifier.fusion_simulation_first.axial_latent_flow import (
    AxialLatentFlow,
)


CHECKPOINT_SCHEMA = "vpfp_shared_global_solution_latent_edm_checkpoint_v1"


@dataclass(frozen=True)
class VPFPSharedGlobalLatentEDMConfig:
    grid_size: int = 13
    latent_channels: int = 64
    condition_dim: int = 2
    width: int = 256
    depth: int = 8
    attention_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 20.0
    rho: float = 7.0
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.2
    dtype: str = "float32"
    representation: str = "vpfp_shared_global_solution_latent_edm_v1"
    grouped_stems: bool = False
    macro_field_channels: int = 0
    macro_stem_width: int = 0
    macro_loss_fraction: float = 0.5
    sampled_global_dim: int = 0
    global_loss_fraction: float = 0.25

    def validate(self) -> None:
        if min(
            int(self.grid_size),
            int(self.latent_channels),
            int(self.condition_dim),
            int(self.width),
            int(self.depth),
            int(self.attention_heads),
        ) < 1:
            raise ValueError(
                "shared global latent EDM dimensions must be positive"
            )
        if int(self.width) % int(self.attention_heads):
            raise ValueError("EDM width must be divisible by attention heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("EDM MLP ratio must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("EDM dropout must lie in [0,1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("shared latent EDM dtype is unsupported")
        if self.grouped_stems:
            macro = int(self.macro_field_channels)
            macro_width = int(self.macro_stem_width)
            micro = int(self.latent_channels) - macro
            micro_width = int(self.width) - macro_width
            if not 0 < macro < int(self.latent_channels):
                raise ValueError("grouped stems need a strict macro boundary")
            if macro_width < macro or micro_width < micro:
                raise ValueError(
                    "grouped stems must preserve full macro and micro rank"
                )
            if not 0.0 < float(self.macro_loss_fraction) < 1.0:
                raise ValueError("macro loss fraction must lie in (0,1)")
        elif any(
            int(value) != 0
            for value in (self.macro_field_channels, self.macro_stem_width)
        ):
            raise ValueError("ungrouped EDM cannot declare grouped stem sizes")
        if int(self.sampled_global_dim) < 0:
            raise ValueError("sampled global dimension cannot be negative")
        if int(self.sampled_global_dim) > 0:
            if not self.grouped_stems:
                raise ValueError(
                    "sampled global coordinates require grouped field stems"
                )
            if not 0.0 < float(self.global_loss_fraction) < 1.0:
                raise ValueError("global loss fraction must lie in (0,1)")
        self.edm_config()

    def edm_config(self) -> AxialEDMConfig:
        return AxialEDMConfig(
            sigma_data=float(self.sigma_data),
            sigma_min=float(self.sigma_min),
            sigma_max=float(self.sigma_max),
            rho=float(self.rho),
            log_sigma_mean=float(self.log_sigma_mean),
            log_sigma_std=float(self.log_sigma_std),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPSharedGlobalLatentEDMConfig":
        config = cls(**dict(payload))
        config.validate()
        return config


class _GroupedInputProjection(nn.Module):
    def __init__(self, input_sizes: tuple[int, int], widths: tuple[int, int]):
        super().__init__()
        self.input_sizes = input_sizes
        self.out_features = int(sum(widths))
        self.projections = nn.ModuleList(
            nn.Linear(input_size, width)
            for input_size, width in zip(input_sizes, widths, strict=True)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        groups = torch.split(values, self.input_sizes, dim=-1)
        return torch.cat(
            tuple(
                projection(group)
                for projection, group in zip(
                    self.projections, groups, strict=True
                )
            ),
            dim=-1,
        )


class _GroupedOutputProjection(nn.Module):
    def __init__(self, widths: tuple[int, int], output_sizes: tuple[int, int]):
        super().__init__()
        self.widths = widths
        self.projections = nn.ModuleList(
            nn.Linear(width, output_size)
            for width, output_size in zip(widths, output_sizes, strict=True)
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        groups = torch.split(values, self.widths, dim=-1)
        return torch.cat(
            tuple(
                projection(group)
                for projection, group in zip(
                    self.projections, groups, strict=True
                )
            ),
            dim=-1,
        )


class _ConditionalJointAxialLatentEDM(nn.Module):
    """Jointly denoise an answer residual and a sampled shared global code."""

    def __init__(
        self,
        grid_shape,
        latent_channels: int,
        *,
        condition_channels: int,
        sampled_global_channels: int,
        width: int,
        depth: int,
        heads: int,
        config: AxialEDMConfig,
        mlp_ratio: float,
        dropout: float,
        spatial_translation_equivariant: bool,
    ):
        super().__init__()
        self.config = config
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.latent_channels = int(latent_channels)
        self.condition_channels = int(condition_channels)
        self.sampled_global_channels = int(sampled_global_channels)
        self.network = AxialLatentFlow(
            self.grid_shape,
            self.latent_channels,
            global_channels=(
                self.condition_channels + self.sampled_global_channels
            ),
            width=int(width),
            depth=int(depth),
            heads=int(heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            spatial_translation_equivariant=bool(
                spatial_translation_equivariant
            ),
        )

    def _sigma_time(self, sigma: torch.Tensor) -> torch.Tensor:
        lower = math.log(float(self.config.sigma_min))
        width = math.log(float(self.config.sigma_max)) - lower
        return ((torch.log(sigma) - lower) / width).clamp(0.0, 1.0)

    def denoise(
        self,
        noisy_latent: torch.Tensor,
        noisy_global: torch.Tensor,
        condition: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(noisy_latent.shape[0])
        if tuple(noisy_global.shape) != (
            batch,
            self.sampled_global_channels,
        ):
            raise ValueError("sampled global code shape changed")
        if tuple(condition.shape) != (batch, self.condition_channels):
            raise ValueError("joint EDM condition shape changed")
        if tuple(sigma.shape) != (batch,) or bool(torch.any(sigma <= 0.0)):
            raise ValueError("joint EDM sigma is invalid")

        sigma_data = float(self.config.sigma_data)
        denominator = torch.sqrt(sigma.square() + sigma_data**2)
        c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
        c_out = sigma * sigma_data / denominator
        c_in = denominator.reciprocal()
        latent_input = noisy_latent * _edm_broadcast(
            c_in, noisy_latent.ndim
        )
        global_input = noisy_global * c_in[:, None]
        network_global = torch.cat((condition, global_input), dim=1)
        latent_residual, all_global_residual = self.network(
            latent_input,
            network_global,
            self._sigma_time(sigma),
        )
        global_residual = all_global_residual[
            :, self.condition_channels :
        ]
        return (
            noisy_latent * _edm_broadcast(c_skip, noisy_latent.ndim)
            + latent_residual * _edm_broadcast(c_out, noisy_latent.ndim),
            noisy_global * c_skip[:, None]
            + global_residual * c_out[:, None],
        )

    def sampling_sigmas(
        self,
        steps: int,
        *,
        device,
        dtype,
        maximum_sigma: float | None = None,
    ) -> torch.Tensor:
        if int(steps) < 1:
            raise ValueError("joint EDM sampling steps must be positive")
        sigma_max = (
            float(self.config.sigma_max)
            if maximum_sigma is None
            else float(maximum_sigma)
        )
        if not float(self.config.sigma_min) < sigma_max <= float(
            self.config.sigma_max
        ):
            raise ValueError("joint EDM sampling maximum sigma is invalid")
        ramp = torch.linspace(
            0.0, 1.0, int(steps), device=device, dtype=dtype
        )
        inverse_rho = 1.0 / float(self.config.rho)
        maximum = sigma_max**inverse_rho
        minimum = float(self.config.sigma_min) ** inverse_rho
        sigmas = (maximum + ramp * (minimum - maximum)) ** float(
            self.config.rho
        )
        return torch.cat((sigmas, torch.zeros_like(sigmas[:1])))

    def sample(
        self,
        condition,
        *,
        steps: int,
        device,
        generator: torch.Generator | None,
        guidance,
        guidance_scale: float,
        method: str,
        initial_noise_transform,
        latent_projection,
        maximum_sigma: float | None = None,
        churn: float = 0.0,
        churn_min: float = 0.0,
        churn_max: float = math.inf,
        churn_noise: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if method not in {"euler", "heun"}:
            raise ValueError("sampling method must be euler or heun")
        if (
            not math.isfinite(float(churn))
            or float(churn) < 0.0
            or not math.isfinite(float(churn_min))
            or float(churn_min) < 0.0
            or math.isnan(float(churn_max))
            or float(churn_max) < float(churn_min)
            or not math.isfinite(float(churn_noise))
            or float(churn_noise) <= 0.0
        ):
            raise ValueError("joint EDM churn parameters are invalid")
        parameter = next(self.parameters())
        condition = torch.as_tensor(
            condition, dtype=parameter.dtype, device=parameter.device
        )
        if (
            condition.ndim != 2
            or int(condition.shape[1]) != self.condition_channels
        ):
            raise ValueError("joint EDM condition shape changed")
        batch = int(condition.shape[0])

        def apply_transform(values, transform, *, name: str):
            if transform is None:
                return values
            with torch.no_grad():
                transformed = torch.as_tensor(
                    transform(values),
                    dtype=values.dtype,
                    device=values.device,
                )
            if tuple(transformed.shape) != tuple(values.shape):
                raise ValueError(f"{name} must preserve residual shape")
            if not bool(torch.all(torch.isfinite(transformed)).item()):
                raise ValueError(f"{name} must return finite values")
            return transformed

        sigmas = self.sampling_sigmas(
            int(steps), device=device, dtype=parameter.dtype,
            maximum_sigma=maximum_sigma,
        )
        latent = torch.randn(
            (batch, *self.grid_shape, self.latent_channels),
            dtype=parameter.dtype,
            device=device,
            generator=generator,
        )
        latent = apply_transform(
            latent, initial_noise_transform, name="initial noise transform"
        )
        latent = apply_transform(
            latent * sigmas[0],
            latent_projection,
            name="residual projection",
        )
        global_code = (
            torch.randn(
                (batch, self.sampled_global_channels),
                dtype=parameter.dtype,
                device=device,
                generator=generator,
            )
            * sigmas[0]
        )

        def guided_denoise(current, current_global, sigma):
            with torch.no_grad():
                denoised, denoised_global = self.denoise(
                    current, current_global, condition, sigma
                )
            denoised = apply_transform(
                denoised,
                latent_projection,
                name="denoised residual projection",
            )
            if guidance is None or float(guidance_scale) == 0.0:
                return denoised, denoised_global
            latent_gradient, global_gradient = guidance(
                current,
                current_global,
                condition,
                self._sigma_time(sigma),
            )
            if latent_gradient.shape != current.shape:
                raise ValueError("joint guidance residual gradient shape changed")
            if global_gradient.shape != current_global.shape:
                raise ValueError("joint guidance global gradient shape changed")
            variance = sigma.square()
            return (
                denoised
                + float(guidance_scale)
                * _edm_broadcast(variance, current.ndim)
                * latent_gradient,
                denoised_global
                + float(guidance_scale)
                * variance[:, None]
                * global_gradient,
            )

        for index in range(int(steps)):
            sigma_value = sigmas[index]
            next_sigma_value = sigmas[index + 1]
            gamma = _edm_churn_gamma(
                float(sigma_value),
                steps=int(steps),
                churn=float(churn),
                churn_min=float(churn_min),
                churn_max=float(churn_max),
            )
            hat_sigma_value = sigma_value * (1.0 + gamma)
            if gamma > 0.0:
                added_variance = (
                    hat_sigma_value.square() - sigma_value.square()
                ).clamp_min(0.0)
                noise_scale = torch.sqrt(added_variance) * float(churn_noise)
                latent_noise = torch.randn(
                    latent.shape,
                    dtype=latent.dtype,
                    device=latent.device,
                    generator=generator,
                )
                global_noise = torch.randn(
                    global_code.shape,
                    dtype=global_code.dtype,
                    device=global_code.device,
                    generator=generator,
                )
                latent = apply_transform(
                    latent + noise_scale * latent_noise,
                    latent_projection,
                    name="churned residual projection",
                )
                global_code = global_code + noise_scale * global_noise
            sigma = torch.full(
                (batch,),
                hat_sigma_value,
                dtype=parameter.dtype,
                device=device,
            )
            denoised, denoised_global = guided_denoise(
                latent, global_code, sigma
            )
            latent_derivative = (latent - denoised) / hat_sigma_value
            global_derivative = (
                global_code - denoised_global
            ) / hat_sigma_value
            step_size = next_sigma_value - hat_sigma_value
            provisional = apply_transform(
                latent + step_size * latent_derivative,
                latent_projection,
                name="provisional residual projection",
            )
            provisional_global = (
                global_code + step_size * global_derivative
            )
            if method == "heun" and float(next_sigma_value) > 0.0:
                next_sigma = torch.full_like(sigma, next_sigma_value)
                next_denoised, next_denoised_global = guided_denoise(
                    provisional, provisional_global, next_sigma
                )
                next_derivative = (
                    provisional - next_denoised
                ) / next_sigma_value
                next_global_derivative = (
                    provisional_global - next_denoised_global
                ) / next_sigma_value
                latent = latent + 0.5 * step_size * (
                    latent_derivative + next_derivative
                )
                global_code = global_code + 0.5 * step_size * (
                    global_derivative + next_global_derivative
                )
            else:
                latent = provisional
                global_code = provisional_global
            latent = apply_transform(
                latent,
                latent_projection,
                name="updated residual projection",
            ).detach()
            global_code = global_code.detach()
        return latent, global_code



def _edm_churn_gamma(
    sigma_value: float,
    *,
    steps: int,
    churn: float,
    churn_min: float,
    churn_max: float,
) -> float:
    if int(steps) < 1:
        raise ValueError("EDM churn needs a positive step count")
    if float(churn) <= 0.0 or not (
        float(churn_min) <= float(sigma_value) <= float(churn_max)
    ):
        return 0.0
    return min(float(churn) / float(steps), math.sqrt(2.0) - 1.0)

def _edm_broadcast(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.reshape(int(values.shape[0]), *((1,) * (int(ndim) - 1)))


class VPFPSharedGlobalLatentEDM(nn.Module):
    @staticmethod
    def build(
        config: VPFPSharedGlobalLatentEDMConfig | Mapping[str, Any],
    ) -> "VPFPSharedGlobalLatentEDM":
        parsed = (
            config
            if isinstance(config, VPFPSharedGlobalLatentEDMConfig)
            else VPFPSharedGlobalLatentEDMConfig.from_dict(config)
        )
        parsed.validate()
        return VPFPSharedGlobalLatentEDM(parsed)

    def __init__(self, config: VPFPSharedGlobalLatentEDMConfig):
        super().__init__()
        config.validate()
        self.config = config
        if int(config.sampled_global_dim) > 0:
            self.edm = _ConditionalJointAxialLatentEDM(
                (1, 1, 1, int(config.grid_size), int(config.grid_size)),
                int(config.latent_channels),
                condition_channels=int(config.condition_dim),
                sampled_global_channels=int(config.sampled_global_dim),
                width=int(config.width),
                depth=int(config.depth),
                heads=int(config.attention_heads),
                config=config.edm_config(),
                mlp_ratio=float(config.mlp_ratio),
                dropout=float(config.dropout),
                spatial_translation_equivariant=True,
            )
        else:
            self.edm = ConditionalAxialLatentEDM(
                (1, 1, 1, int(config.grid_size), int(config.grid_size)),
                int(config.latent_channels),
                global_channels=int(config.condition_dim),
                width=int(config.width),
                depth=int(config.depth),
                heads=int(config.attention_heads),
                config=config.edm_config(),
                mlp_ratio=float(config.mlp_ratio),
                dropout=float(config.dropout),
                spatial_translation_equivariant=True,
            )
        if config.grouped_stems:
            macro = int(config.macro_field_channels)
            input_sizes = (macro, int(config.latent_channels) - macro)
            widths = (
                int(config.macro_stem_width),
                int(config.width) - int(config.macro_stem_width),
            )
            self.edm.network.input_projection = _GroupedInputProjection(
                input_sizes, widths
            )
            self.edm.network.output_projection = _GroupedOutputProjection(
                widths, input_sizes
            )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def to_edm_grid(self, shared_latent) -> torch.Tensor:
        values = torch.as_tensor(shared_latent)
        expected = (
            int(values.shape[0]),
            int(self.config.latent_channels),
            int(self.config.grid_size),
            int(self.config.grid_size),
        )
        if tuple(values.shape) != expected:
            raise ValueError("shared latent shape does not match EDM")
        return values.permute(0, 2, 3, 1)[:, None, None, None]

    def from_edm_grid(self, edm_grid) -> torch.Tensor:
        values = torch.as_tensor(edm_grid)
        expected = (
            int(values.shape[0]),
            1,
            1,
            1,
            int(self.config.grid_size),
            int(self.config.grid_size),
            int(self.config.latent_channels),
        )
        if tuple(values.shape) != expected:
            raise ValueError("EDM grid shape changed")
        return values[:, 0, 0, 0].permute(0, 3, 1, 2)


def shared_global_latent_edm_loss(
    model: VPFPSharedGlobalLatentEDM,
    clean_shared_latent,
    conditions,
    *,
    clean_global_latent=None,
    generator: torch.Generator | None = None,
    auxiliary_loss_fn: Callable | None = None,
    auxiliary_weight: float = 0.0,
    macro_loss_fraction: float | None = None,
    global_loss_fraction: float | None = None,
    stratified_high_noise_fraction: float = 0.0,
    stratified_high_noise_min: float | None = None,
    subspace_noise_fn: Callable | None = None,
    denoised_projection_fn: Callable | None = None,
    latent_reconstruction_fn: Callable | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = next(model.parameters())
    clean = torch.as_tensor(
        clean_shared_latent, dtype=parameter.dtype, device=parameter.device
    )
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    if tuple(condition.shape) != (
        int(clean.shape[0]),
        int(model.config.condition_dim),
    ):
        raise ValueError("shared latent EDM condition shape is invalid")
    has_sampled_global = int(model.config.sampled_global_dim) > 0
    if has_sampled_global:
        if clean_global_latent is None:
            raise ValueError("joint-global EDM needs clean global coordinates")
        clean_global = torch.as_tensor(
            clean_global_latent,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        if tuple(clean_global.shape) != (
            int(clean.shape[0]),
            int(model.config.sampled_global_dim),
        ):
            raise ValueError("clean sampled-global coordinates have wrong shape")
    else:
        if clean_global_latent is not None:
            raise ValueError("field-only EDM cannot receive sampled globals")
        if global_loss_fraction is not None or latent_reconstruction_fn is not None:
            raise ValueError(
                "global loss/reconstruction callbacks need sampled globals"
            )
        clean_global = None

    if model.config.grouped_stems:
        loss, metrics = _group_balanced_edm_loss(
            model,
            model.to_edm_grid(clean),
            condition,
            clean_global=clean_global,
            generator=generator,
            auxiliary_loss_fn=auxiliary_loss_fn,
            auxiliary_weight=float(auxiliary_weight),
            macro_loss_fraction=macro_loss_fraction,
            global_loss_fraction=global_loss_fraction,
            stratified_high_noise_fraction=float(stratified_high_noise_fraction),
            stratified_high_noise_min=stratified_high_noise_min,
            subspace_noise_fn=subspace_noise_fn,
            denoised_projection_fn=denoised_projection_fn,
            latent_reconstruction_fn=latent_reconstruction_fn,
        )
    else:
        if (
            float(stratified_high_noise_fraction) != 0.0
            or stratified_high_noise_min is not None
        ):
            raise ValueError(
                "stratified high-noise training requires grouped stems"
            )
        if any(
            value is not None
            for value in (
                macro_loss_fraction,
                global_loss_fraction,
                subspace_noise_fn,
                denoised_projection_fn,
                latent_reconstruction_fn,
            )
        ):
            raise ValueError(
                "group-specific loss or projection callbacks require grouped stems"
            )
        loss, metrics = model.edm.training_loss(
            model.to_edm_grid(clean),
            condition,
            generator=generator,
            auxiliary_loss_fn=auxiliary_loss_fn,
            auxiliary_weight=float(auxiliary_weight),
        )
    return loss, {"loss": loss, **metrics}



def _sample_stratified_edm_sigmas(
    config,
    batch_size: int,
    *,
    device,
    dtype,
    generator: torch.Generator | None,
    high_noise_fraction: float = 0.0,
    high_noise_min: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample answer-coordinate noise with an optional log-uniform high tail."""

    count = int(batch_size)
    fraction = float(high_noise_fraction)
    if count < 1:
        raise ValueError("EDM sigma batch size must be positive")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("stratified high-noise fraction must lie in [0,1]")
    normal = torch.randn(
        count, device=device, dtype=dtype, generator=generator
    )
    sigma = torch.exp(
        float(config.log_sigma_mean)
        + float(config.log_sigma_std) * normal
    ).clamp(float(config.sigma_min), float(config.sigma_max))
    high_mask = torch.zeros(count, dtype=torch.bool, device=device)
    if fraction == 0.0:
        return sigma, high_mask
    if high_noise_min is None:
        raise ValueError("positive high-noise fraction needs a lower bound")
    lower = float(high_noise_min)
    upper = float(config.sigma_max)
    if (
        not math.isfinite(lower)
        or not float(config.sigma_min) < lower < upper
    ):
        raise ValueError("stratified high-noise lower bound is invalid")
    high_count = min(count, max(1, int(round(fraction * count))))
    indices = torch.randperm(count, device=device, generator=generator)[
        :high_count
    ]
    uniform = torch.rand(
        high_count, device=device, dtype=dtype, generator=generator
    )
    high_sigma = torch.exp(
        math.log(lower) + uniform * (math.log(upper) - math.log(lower))
    )
    sigma = sigma.clone()
    sigma[indices] = high_sigma
    high_mask[indices] = True
    return sigma, high_mask

def _group_balanced_edm_loss(
    model: VPFPSharedGlobalLatentEDM,
    clean: torch.Tensor,
    condition: torch.Tensor,
    *,
    clean_global: torch.Tensor | None,
    generator: torch.Generator | None,
    auxiliary_loss_fn: Callable | None,
    auxiliary_weight: float,
    macro_loss_fraction: float | None,
    global_loss_fraction: float | None,
    stratified_high_noise_fraction: float,
    stratified_high_noise_min: float | None,
    subspace_noise_fn: Callable | None,
    denoised_projection_fn: Callable | None,
    latent_reconstruction_fn: Callable | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary weight cannot be negative")
    if auxiliary_weight and auxiliary_loss_fn is None:
        raise ValueError("weighted auxiliary loss needs a function")
    if auxiliary_weight and clean_global is not None:
        if latent_reconstruction_fn is None:
            raise ValueError(
                "joint-global auxiliary loss needs exact field reconstruction"
            )
    edm_config = model.edm.config
    sigma, high_noise_mask = _sample_stratified_edm_sigmas(
        edm_config,
        int(clean.shape[0]),
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
        high_noise_fraction=float(stratified_high_noise_fraction),
        high_noise_min=stratified_high_noise_min,
    )
    sigma_view = sigma.reshape(
        int(sigma.shape[0]), *((1,) * (clean.ndim - 1))
    )
    if subspace_noise_fn is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    else:
        noise = torch.as_tensor(
            subspace_noise_fn(clean, generator=generator),
            dtype=clean.dtype,
            device=clean.device,
        )
        if tuple(noise.shape) != tuple(clean.shape) or not bool(
            torch.all(torch.isfinite(noise)).item()
        ):
            raise ValueError(
                "answer-subspace noise must be finite and shape preserving"
            )
    noisy = clean + sigma_view * noise

    if clean_global is None:
        denoised = model.edm.denoise(noisy, condition, sigma)
        denoised_global = None
    else:
        global_noise = torch.randn(
            clean_global.shape,
            device=clean_global.device,
            dtype=clean_global.dtype,
            generator=generator,
        )
        noisy_global = clean_global + sigma[:, None] * global_noise
        denoised, denoised_global = model.edm.denoise(
            noisy, noisy_global, condition, sigma
        )
    if denoised_projection_fn is not None:
        denoised = torch.as_tensor(
            denoised_projection_fn(denoised),
            dtype=clean.dtype,
            device=clean.device,
        )
        if tuple(denoised.shape) != tuple(clean.shape) or not bool(
            torch.all(torch.isfinite(denoised)).item()
        ):
            raise ValueError(
                "denoised answer projection must be finite and shape preserving"
            )

    error = (denoised - clean).square()
    boundary = int(model.config.macro_field_channels)
    macro_error = error[..., :boundary].flatten(1).mean(dim=1)
    micro_error = error[..., boundary:].flatten(1).mean(dim=1)
    weight = (
        sigma.square() + float(edm_config.sigma_data) ** 2
    ) / (sigma * float(edm_config.sigma_data)).square()
    macro_loss = (weight * macro_error).mean()
    micro_loss = (weight * micro_error).mean()
    macro_fraction = (
        float(model.config.macro_loss_fraction)
        if macro_loss_fraction is None
        else float(macro_loss_fraction)
    )
    if not 0.0 < macro_fraction < 1.0:
        raise ValueError("macro loss fraction must lie in (0,1)")
    residual_loss = (
        macro_fraction * macro_loss
        + (1.0 - macro_fraction) * micro_loss
    )

    global_loss = residual_loss.new_zeros(())
    sampled_global_fraction = residual_loss.new_zeros(())
    if clean_global is None:
        latent_loss = residual_loss
    else:
        assert denoised_global is not None
        global_error = (denoised_global - clean_global).square().mean(dim=1)
        global_loss = (weight * global_error).mean()
        sampled_global_fraction_value = (
            float(model.config.global_loss_fraction)
            if global_loss_fraction is None
            else float(global_loss_fraction)
        )
        if not 0.0 < sampled_global_fraction_value < 1.0:
            raise ValueError("sampled global loss fraction must lie in (0,1)")
        sampled_global_fraction = residual_loss.new_tensor(
            sampled_global_fraction_value
        )
        latent_loss = (
            (1.0 - sampled_global_fraction_value) * residual_loss
            + sampled_global_fraction_value * global_loss
        )

    auxiliary_loss = latent_loss.new_zeros(())
    auxiliary_metrics: dict[str, torch.Tensor] = {}
    if auxiliary_weight:
        assert auxiliary_loss_fn is not None
        if clean_global is None:
            auxiliary_predicted = denoised
            auxiliary_clean = clean
        else:
            assert denoised_global is not None
            assert latent_reconstruction_fn is not None
            auxiliary_predicted = latent_reconstruction_fn(
                denoised, denoised_global
            )
            auxiliary_clean = latent_reconstruction_fn(clean, clean_global)
        auxiliary_result = auxiliary_loss_fn(
            auxiliary_predicted, auxiliary_clean
        )
        if isinstance(auxiliary_result, tuple):
            if len(auxiliary_result) != 2:
                raise ValueError("auxiliary loss tuple must contain loss and metrics")
            auxiliary_loss, raw_auxiliary_metrics = auxiliary_result
            if not isinstance(raw_auxiliary_metrics, Mapping):
                raise ValueError("auxiliary metrics must be a mapping")
            for name, value in raw_auxiliary_metrics.items():
                metric = torch.as_tensor(
                    value, dtype=latent_loss.dtype, device=latent_loss.device
                )
                if metric.ndim != 0 or not bool(torch.isfinite(metric).item()):
                    raise ValueError(
                        f"auxiliary metric {name!r} must be a finite scalar"
                    )
                auxiliary_metrics[f"auxiliary_{name}"] = metric
        else:
            auxiliary_loss = auxiliary_result
        if auxiliary_loss.ndim != 0 or not bool(
            torch.isfinite(auxiliary_loss).item()
        ):
            raise ValueError("auxiliary loss must be a finite scalar")
    total = latent_loss + float(auxiliary_weight) * auxiliary_loss
    return total, {
        "latent_edm_loss": latent_loss,
        "residual_edm_loss": residual_loss,
        "macro_edm_loss": macro_loss,
        "micro_edm_loss": micro_loss,
        "global_edm_loss": global_loss,
        "latent_structure_loss": latent_loss.new_zeros(()),
        "auxiliary_loss": auxiliary_loss,
        **auxiliary_metrics,
        "macro_loss_fraction": latent_loss.new_tensor(macro_fraction),
        "sampled_global_loss_fraction": sampled_global_fraction,
        "answer_subspace_enforced": latent_loss.new_tensor(
            float(
                subspace_noise_fn is not None
                and denoised_projection_fn is not None
            )
        ),
        "sigma_mean": sigma.mean(),
        "sigma_max_observed": sigma.max(),
        "stratified_high_noise_fraction": high_noise_mask.float().mean(),
    }


def sample_shared_global_latent_edm(
    model: VPFPSharedGlobalLatentEDM,
    conditions,
    *,
    steps: int = 64,
    generator: torch.Generator | None = None,
    guidance: Callable | None = None,
    guidance_scale: float = 0.0,
    initial_shared_latent=None,
    method: str = "heun",
    initial_noise_transform: Callable | None = None,
    latent_projection: Callable | None = None,
    maximum_sigma: float | None = None,
    churn: float = 0.0,
    churn_min: float = 0.0,
    churn_max: float = math.inf,
    churn_noise: float = 1.0,
    latent_reconstruction_fn: Callable | None = None,
) -> torch.Tensor:
    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    if condition.ndim != 2 or int(condition.shape[1]) != int(
        model.config.condition_dim
    ):
        raise ValueError("shared latent EDM condition shape is invalid")

    if int(model.config.sampled_global_dim) > 0:
        if initial_shared_latent is not None:
            raise ValueError(
                "joint-global sampling does not accept a field-only anchor"
            )
        if latent_reconstruction_fn is None:
            raise ValueError(
                "joint-global sampling needs exact field reconstruction"
            )
        sampled_residual, sampled_global = model.edm.sample(
            condition,
            steps=int(steps),
            device=parameter.device,
            maximum_sigma=maximum_sigma,
            generator=generator,
            guidance=guidance,
            guidance_scale=float(guidance_scale),
            initial_noise_transform=initial_noise_transform,
            latent_projection=latent_projection,
            churn=float(churn),
            churn_min=float(churn_min),
            churn_max=float(churn_max),
            churn_noise=float(churn_noise),
            method=str(method),
        )
        sampled = torch.as_tensor(
            latent_reconstruction_fn(
                sampled_residual, sampled_global
            ),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        if tuple(sampled.shape) != tuple(sampled_residual.shape):
            raise ValueError("joint-global reconstruction changed field shape")
        if not bool(torch.all(torch.isfinite(sampled)).item()):
            raise ValueError("joint-global reconstruction is not finite")
        return model.from_edm_grid(sampled)

    if float(churn) != 0.0:
        raise ValueError("field-only sampling does not support stochastic churn")
    if latent_reconstruction_fn is not None:
        raise ValueError("field-only sampling cannot reconstruct globals")
    initial = (
        None
        if initial_shared_latent is None
        else model.to_edm_grid(
            torch.as_tensor(
                initial_shared_latent,
                dtype=parameter.dtype,
                device=parameter.device,
            )
        )
    )
    sampled, _ = model.edm.sample(
        condition,
        steps=int(steps),
        device=parameter.device,
        sampling_sigma_max=maximum_sigma,
        initial_latent=initial,
        generator=generator,
        guidance=guidance,
        guidance_scale=float(guidance_scale),
        initial_noise_transform=initial_noise_transform,
        latent_projection=latent_projection,
        method=str(method),
    )
    return model.from_edm_grid(sampled)



@torch.no_grad()
def sample_shared_global_components_edm(
    model: VPFPSharedGlobalLatentEDM,
    conditions,
    *,
    steps: int = 64,
    generator: torch.Generator | None = None,
    guidance: Callable | None = None,
    guidance_scale: float = 0.0,
    method: str = "heun",
    initial_noise_transform: Callable | None = None,
    latent_projection: Callable | None = None,
    maximum_sigma: float | None = None,
    churn: float = 0.0,
    churn_min: float = 0.0,
    churn_max: float = math.inf,
    churn_noise: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample coupled local and global coordinates of one hidden answer."""

    if int(model.config.sampled_global_dim) <= 0:
        raise ValueError("component sampling requires a joint-global EDM")
    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    if condition.ndim != 2 or int(condition.shape[1]) != int(
        model.config.condition_dim
    ):
        raise ValueError("joint component condition shape is invalid")
    sampled_residual, sampled_global = model.edm.sample(
        condition,
        steps=int(steps),
        device=parameter.device,
        maximum_sigma=maximum_sigma,
        generator=generator,
        guidance=guidance,
        guidance_scale=float(guidance_scale),
        initial_noise_transform=initial_noise_transform,
        latent_projection=latent_projection,
        churn=float(churn),
        churn_min=float(churn_min),
        churn_max=float(churn_max),
        churn_noise=float(churn_noise),
        method=str(method),
    )
    residual = model.from_edm_grid(sampled_residual)
    if not bool(torch.all(torch.isfinite(residual)).item()) or not bool(
        torch.all(torch.isfinite(sampled_global)).item()
    ):
        raise FloatingPointError("joint answer coordinates are non-finite")
    return residual, sampled_global

def save_shared_global_latent_edm_checkpoint(
    path: str | Path,
    *,
    model: VPFPSharedGlobalLatentEDM,
    shared_latent_normalization: Mapping[str, Any],
    condition_normalization: Mapping[str, Any],
    codec_checkpoint: str | Path,
    codec_checkpoint_sha256: str,
    solution_reliability_report: str | Path,
    solution_reliability_report_sha256: str,
    train_config: Mapping[str, Any],
    step: int,
    optimizer_state: Mapping[str, Any] | None = None,
    ema_state_dict: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "shared_latent_normalization": dict(shared_latent_normalization),
            "condition_normalization": dict(condition_normalization),
            "codec_checkpoint": str(Path(codec_checkpoint).resolve()),
            "codec_checkpoint_sha256": str(codec_checkpoint_sha256),
            "solution_reliability_report": str(
                Path(solution_reliability_report).resolve()
            ),
            "solution_reliability_report_sha256": str(
                solution_reliability_report_sha256
            ),
            "answer_first_contract": {
                "hidden_continuous_solution_is_primary": True,
                "public_source_is_operator_image_of_hidden_answer": True,
                "equation_first_source_is_training_target": False,
                "equation_first_dynamics_are_generation_constraints": False,
            },
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_shared_global_latent_edm_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[VPFPSharedGlobalLatentEDM, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected shared global latent EDM checkpoint schema")
    contract = dict(payload.get("answer_first_contract") or {})
    if not bool(
        contract.get("hidden_continuous_solution_is_primary")
        and contract.get("public_source_is_operator_image_of_hidden_answer")
        and not contract.get("equation_first_source_is_training_target")
        and not contract.get("equation_first_dynamics_are_generation_constraints")
    ):
        raise ValueError("shared latent checkpoint violates answer-first contract")
    model = VPFPSharedGlobalLatentEDM.build(payload["model_config"]).to(
        device=device
    )
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload
