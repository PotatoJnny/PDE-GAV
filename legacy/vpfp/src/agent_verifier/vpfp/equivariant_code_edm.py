from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from agent_verifier.fusion_simulation_first.axial_latent_diffusion import (
    AxialEDMConfig,
    ConditionalAxialLatentEDM,
)
from agent_verifier.vpfp.structured_diffusion import _physical_fourier_basis


CHECKPOINT_SCHEMA = "vpfp_macro_pca_equivariant_micro_edm_checkpoint_v1"


@dataclass(frozen=True)
class VPFPEquivariantCodeEDMConfig:
    macro_dim: int
    micro_channels: int
    spatial_feature_count: int
    condition_dim: int = 2
    time_embedding_dim: int = 128
    macro_width: int = 384
    macro_layers: int = 8
    micro_width: int = 512
    micro_layers: int = 8
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    axial_mlp_ratio: float = 2.0
    dropout: float = 0.0
    timesteps: int = 200
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 20.0
    rho: float = 7.0
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.2
    dtype: str = "float32"

    @property
    def micro_dim(self) -> int:
        return int(self.micro_channels) * int(self.spatial_feature_count)

    @property
    def spatial_side(self) -> int:
        return int(round(np.sqrt(float(self.spatial_feature_count))))

    @property
    def spatial_mode(self) -> int:
        return (self.spatial_side - 1) // 2

    @property
    def micro_residual_output_rank_cap(self) -> int:
        return min(int(self.micro_width), int(self.micro_channels))

    @property
    def micro_output_rank_cap(self) -> int:
        # EDM preconditioning includes a full-dimensional identity skip.
        return int(self.micro_channels)

    @property
    def has_full_rank_micro_output(self) -> bool:
        return True

    def edm_config(self) -> AxialEDMConfig:
        return AxialEDMConfig(
            sigma_data=float(self.sigma_data),
            sigma_min=float(self.sigma_min),
            sigma_max=float(self.sigma_max),
            rho=float(self.rho),
            log_sigma_mean=float(self.log_sigma_mean),
            log_sigma_std=float(self.log_sigma_std),
        )

    def validate(self) -> None:
        dimensions = (
            self.macro_dim,
            self.micro_channels,
            self.spatial_feature_count,
            self.condition_dim,
            self.macro_width,
            self.macro_layers,
            self.micro_width,
            self.micro_layers,
            self.attention_heads,
            self.timesteps,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("equivariant code EDM dimensions must be positive")
        if self.spatial_side * self.spatial_side != int(self.spatial_feature_count):
            raise ValueError("micro Fourier feature count must be an odd square")
        if self.spatial_side % 2 != 1:
            raise ValueError("micro Fourier grid side must be odd")
        if int(self.macro_width) % int(self.attention_heads):
            raise ValueError("macro width must be divisible by attention heads")
        if int(self.micro_width) % int(self.attention_heads):
            raise ValueError("micro width must be divisible by attention heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if (
            not np.isfinite(float(self.axial_mlp_ratio))
            or float(self.axial_mlp_ratio) <= 0.0
        ):
            raise ValueError("axial MLP ratio must be positive and finite")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("unsupported EDM dtype")
        self.edm_config()

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPEquivariantCodeEDMConfig":
        config = cls(**dict(data))
        config.validate()
        return config


class VPFPEquivariantCodeEDM(nn.Module):
    @staticmethod
    def build(config: VPFPEquivariantCodeEDMConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPEquivariantCodeEDMConfig)
            else VPFPEquivariantCodeEDMConfig.from_dict(dict(config))
        )
        cfg.validate()
        return VPFPEquivariantCodeEDM(cfg)

    def __init__(self, config: VPFPEquivariantCodeEDMConfig):
        super().__init__()
        config.validate()
        self.config = config
        edm = config.edm_config()
        self.macro_model = ConditionalAxialLatentEDM(
            (1, 1, 1, 1, 1),
            int(config.macro_dim),
            global_channels=int(config.condition_dim),
            width=int(config.macro_width),
            depth=int(config.macro_layers),
            heads=int(config.attention_heads),
            config=edm,
            mlp_ratio=float(config.axial_mlp_ratio),
            dropout=float(config.dropout),
        )
        self.micro_model = ConditionalAxialLatentEDM(
            (1, 1, 1, int(config.spatial_side), int(config.spatial_side)),
            int(config.micro_channels),
            global_channels=int(config.macro_dim) + int(config.condition_dim),
            width=int(config.micro_width),
            depth=int(config.micro_layers),
            heads=int(config.attention_heads),
            config=edm,
            mlp_ratio=float(config.axial_mlp_ratio),
            dropout=float(config.dropout),
            spatial_translation_equivariant=True,
        )
        basis = _physical_fourier_basis(
            torch,
            max_mode=int(config.spatial_mode),
            grid_size=int(config.spatial_side),
            include_constant=True,
        )
        if tuple(basis.shape) != (
            int(config.spatial_feature_count),
            int(config.spatial_feature_count),
        ):
            raise ValueError("micro Fourier basis shape changed")
        self.register_buffer("fourier_basis", basis.to(dtype=torch.float32))
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def macro_to_grid(self, macro):
        values = torch.as_tensor(macro)
        expected = (int(values.shape[0]), int(self.config.macro_dim))
        if values.ndim != 2 or tuple(values.shape) != expected:
            raise ValueError("macro code shape does not match EDM config")
        return values.reshape(int(values.shape[0]), 1, 1, 1, 1, 1, -1)

    def macro_from_grid(self, macro_grid):
        values = torch.as_tensor(macro_grid)
        expected = (
            int(values.shape[0]),
            1,
            1,
            1,
            1,
            1,
            int(self.config.macro_dim),
        )
        if tuple(values.shape) != expected:
            raise ValueError("macro EDM grid shape changed")
        return values.reshape(int(values.shape[0]), int(self.config.macro_dim))

    def micro_to_grid(self, micro):
        values = torch.as_tensor(micro)
        if values.ndim != 2 or int(values.shape[1]) != int(self.config.micro_dim):
            raise ValueError("micro code shape does not match EDM config")
        spectral = values.reshape(
            int(values.shape[0]),
            int(self.config.micro_channels),
            int(self.config.spatial_feature_count),
        )
        basis = self.fourier_basis.to(spectral)
        grid = torch.einsum("bcf,gf->bgc", spectral, basis)
        return grid.reshape(
            int(values.shape[0]),
            1,
            1,
            1,
            int(self.config.spatial_side),
            int(self.config.spatial_side),
            int(self.config.micro_channels),
        )

    def micro_from_grid(self, micro_grid):
        values = torch.as_tensor(micro_grid)
        expected = (
            int(values.shape[0]),
            1,
            1,
            1,
            int(self.config.spatial_side),
            int(self.config.spatial_side),
            int(self.config.micro_channels),
        )
        if tuple(values.shape) != expected:
            raise ValueError("micro EDM grid shape changed")
        grid = values.reshape(
            int(values.shape[0]),
            int(self.config.spatial_feature_count),
            int(self.config.micro_channels),
        )
        spectral = torch.einsum("bgc,gf->bcf", grid, self.fourier_basis.to(grid))
        return spectral.reshape(int(values.shape[0]), int(self.config.micro_dim))


def equivariant_code_edm_loss(
    model,
    clean_macro,
    clean_micro,
    conditions,
    *,
    macro_timesteps=None,
    micro_timesteps=None,
    macro_sigmas=None,
    micro_sigmas=None,
    macro_noise=None,
    micro_noise=None,
    macro_weight: float = 1.0,
    micro_weight: float = 1.0,
    macro_condition_augmentation_std: float = 0.0,
):
    parameter = next(model.parameters())
    macro = torch.as_tensor(clean_macro, dtype=parameter.dtype, device=parameter.device)
    micro = torch.as_tensor(clean_micro, dtype=parameter.dtype, device=parameter.device)
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(macro.shape[0])
    if tuple(macro.shape) != (batch, int(model.config.macro_dim)):
        raise ValueError("clean macro code shape does not match EDM config")
    if tuple(micro.shape) != (batch, int(model.config.micro_dim)):
        raise ValueError("clean micro code shape does not match EDM config")
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("condition shape does not match EDM config")
    weights = (float(macro_weight), float(micro_weight))
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("EDM loss weights must be finite and nonnegative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one EDM factor loss weight must be positive")
    augmentation = float(macro_condition_augmentation_std)
    if not np.isfinite(augmentation) or augmentation < 0.0:
        raise ValueError("macro condition augmentation must be nonnegative")

    zero = torch.zeros((), dtype=parameter.dtype, device=parameter.device)
    macro_loss = zero
    micro_loss = zero
    macro_sigma = zero
    micro_sigma = zero
    if weights[0] > 0.0:
        macro_loss, macro_sigma = _edm_block_loss(
            model.macro_model,
            model.macro_to_grid(macro),
            condition,
            timesteps=macro_timesteps,
            fixed_sigmas=macro_sigmas,
            fixed_noise=(
                model.macro_to_grid(
                    torch.as_tensor(
                        macro_noise,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                )
                if macro_noise is not None
                else None
            ),
            total_timesteps=int(model.config.timesteps),
        )
    if weights[1] > 0.0:
        macro_context = macro
        if augmentation:
            macro_context = macro + augmentation * torch.randn_like(macro)
        micro_condition = torch.cat((macro_context, condition), dim=1)
        micro_loss, micro_sigma = _edm_block_loss(
            model.micro_model,
            model.micro_to_grid(micro),
            micro_condition,
            timesteps=micro_timesteps,
            fixed_sigmas=micro_sigmas,
            fixed_noise=(
                model.micro_to_grid(
                    torch.as_tensor(
                        micro_noise,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                )
                if micro_noise is not None
                else None
            ),
            total_timesteps=int(model.config.timesteps),
        )
    loss = (weights[0] * macro_loss + weights[1] * micro_loss) / sum(weights)
    return loss, {
        "loss": loss,
        "macro_noise_mse": macro_loss,
        "micro_noise_mse": micro_loss,
        "macro_sigma_mean": macro_sigma,
        "micro_sigma_mean": micro_sigma,
    }


def sample_equivariant_codes_edm(
    model,
    conditions,
    *,
    macro_condition=None,
    sample_steps: int | None = None,
    macro_sample_steps: int | None = None,
    micro_sample_steps: int | None = None,
    eta: float = 0.0,
    macro_guidance: Callable | None = None,
    micro_guidance: Callable | None = None,
    macro_guidance_scale: float = 0.0,
    micro_guidance_scale: float = 0.0,
    macro_churn: float = 0.0,
    micro_churn: float = 0.0,
    churn_min: float = 0.0,
    churn_max: float = float("inf"),
    churn_noise: float = 1.0,
    generator: torch.Generator | None = None,
):
    if float(eta) != 0.0:
        raise ValueError("EDM Heun sampling does not use DDIM eta")
    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(condition.shape[0])
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("condition shape does not match EDM config")
    sampler_values = {
        "macro_guidance_scale": float(macro_guidance_scale),
        "micro_guidance_scale": float(micro_guidance_scale),
        "macro_churn": float(macro_churn),
        "micro_churn": float(micro_churn),
        "churn_min": float(churn_min),
        "churn_noise": float(churn_noise),
    }
    if any(not np.isfinite(value) for value in sampler_values.values()):
        raise ValueError("EDM sampler controls must be finite")
    if min(float(macro_churn), float(micro_churn), float(churn_min)) < 0.0:
        raise ValueError("EDM churn controls must be nonnegative")
    if float(churn_noise) <= 0.0:
        raise ValueError("EDM churn noise scale must be positive")
    if math.isnan(float(churn_max)) or float(churn_max) < float(churn_min):
        raise ValueError("EDM churn interval is invalid")
    default_steps = 32 if sample_steps is None else int(sample_steps)
    if macro_condition is None:
        macro_grid = _sample_edm_block(
            model.macro_model,
            condition,
            steps=(
                default_steps
                if macro_sample_steps is None
                else int(macro_sample_steps)
            ),
            device=parameter.device,
            generator=generator,
            guidance=macro_guidance,
            guidance_scale=float(macro_guidance_scale),
            churn=float(macro_churn),
            churn_min=float(churn_min),
            churn_max=float(churn_max),
            churn_noise=float(churn_noise),
        )
        macro = model.macro_from_grid(macro_grid)
    else:
        macro = torch.as_tensor(
            macro_condition, dtype=parameter.dtype, device=parameter.device
        )
        if tuple(macro.shape) != (batch, int(model.config.macro_dim)):
            raise ValueError("fixed macro condition is invalid")
    micro_condition = torch.cat((macro, condition), dim=1)
    micro_grid = _sample_edm_block(
        model.micro_model,
        micro_condition,
        steps=(
            default_steps if micro_sample_steps is None else int(micro_sample_steps)
        ),
        device=parameter.device,
        generator=generator,
        guidance=micro_guidance,
        guidance_scale=float(micro_guidance_scale),
        churn=float(micro_churn),
        churn_min=float(churn_min),
        churn_max=float(churn_max),
        churn_noise=float(churn_noise),
    )
    return macro, model.micro_from_grid(micro_grid)


def _sample_edm_block(
    block,
    condition,
    *,
    steps: int,
    device: torch.device,
    generator: torch.Generator | None,
    guidance: Callable | None,
    guidance_scale: float,
    churn: float,
    churn_min: float,
    churn_max: float,
    churn_noise: float,
    context_latent=None,
):
    if churn == 0.0:
        latent, _ = block.sample(
            condition,
            steps=int(steps),
            device=device,
            generator=generator,
            guidance=guidance,
            context_latent=context_latent,
            guidance_scale=float(guidance_scale),
            method="heun",
        )
        return latent

    if int(steps) < 1:
        raise ValueError("EDM sampling steps must be positive")
    batch = int(condition.shape[0])
    sigmas = block.sampling_sigmas(
        int(steps),
        device=device,
        dtype=condition.dtype,
    )
    latent = torch.randn(
        (batch,) + block.grid_shape + (block.latent_channels,),
        dtype=condition.dtype,
        device=device,
        generator=generator,
    ) * sigmas[0]
    maximum_gamma = math.sqrt(2.0) - 1.0
    for index in range(int(steps)):
        sigma_value = sigmas[index]
        sigma_scalar = float(sigma_value)
        gamma = (
            min(float(churn) / float(steps), maximum_gamma)
            if float(churn_min) <= sigma_scalar <= float(churn_max)
            else 0.0
        )
        sigma_hat_value = sigma_value * (1.0 + gamma)
        if gamma:
            noise = torch.randn(
                latent.shape,
                dtype=latent.dtype,
                device=device,
                generator=generator,
            )
            noise_amplitude = torch.sqrt(
                torch.clamp(
                    sigma_hat_value.square() - sigma_value.square(),
                    min=0.0,
                )
            )
            latent = latent + float(churn_noise) * noise_amplitude * noise

        sigma_hat = torch.ones(
            batch, dtype=latent.dtype, device=device
        ) * sigma_hat_value
        denoised = block._guided_conditional_denoise(
            latent,
            condition,
            sigma_hat,
            context_latent=context_latent,
            guidance=guidance,
            guidance_scale=float(guidance_scale),
        )
        derivative = (latent - denoised) / sigma_hat_value
        next_sigma_value = sigmas[index + 1]
        step_size = next_sigma_value - sigma_hat_value
        provisional = latent + step_size * derivative
        if float(next_sigma_value) > 0.0:
            next_sigma = torch.ones_like(sigma_hat) * next_sigma_value
            next_denoised = block._guided_conditional_denoise(
                provisional,
                condition,
                next_sigma,
                context_latent=context_latent,
                guidance=guidance,
                guidance_scale=float(guidance_scale),
            )
            next_derivative = (
                provisional - next_denoised
            ) / next_sigma_value
            latent = latent + 0.5 * step_size * (
                derivative + next_derivative
            )
        else:
            latent = provisional
        latent = latent.detach()
    return latent


def save_equivariant_code_edm_checkpoint(
    path: str | Path,
    *,
    model,
    condition_normalization: dict[str, Any],
    micro_code_normalization: dict[str, Any],
    macro_basis: str | Path,
    macro_basis_sha256: str,
    codec_checkpoint: str | Path,
    codec_checkpoint_sha256: str,
    train_config: dict[str, Any],
    step: int,
    optimizer_state: dict[str, Any] | None = None,
    ema_state_dict: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "condition_normalization": dict(condition_normalization),
            "micro_code_normalization": dict(micro_code_normalization),
            "macro_basis": str(Path(macro_basis).resolve()),
            "macro_basis_sha256": str(macro_basis_sha256),
            "codec_checkpoint": str(Path(codec_checkpoint).resolve()),
            "codec_checkpoint_sha256": str(codec_checkpoint_sha256),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_equivariant_code_edm_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected equivariant code EDM checkpoint schema")
    model = VPFPEquivariantCodeEDM.build(payload["model_config"]).to(device=device)
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _edm_block_loss(
    block,
    clean,
    condition,
    *,
    timesteps,
    fixed_sigmas,
    fixed_noise,
    total_timesteps: int,
    context_latent=None,
):
    batch = int(clean.shape[0])
    if fixed_sigmas is not None and timesteps is not None:
        raise ValueError("provide either EDM timesteps or sigmas, not both")
    if fixed_sigmas is not None:
        sigma = torch.as_tensor(
            fixed_sigmas, dtype=clean.dtype, device=clean.device
        ).reshape(-1)
        if tuple(sigma.shape) != (batch,):
            raise ValueError("fixed EDM sigma shape changed")
        if not bool(torch.all(torch.isfinite(sigma)).item()):
            raise ValueError("fixed EDM sigmas must be finite")
        if bool(
            torch.any(
                (sigma < float(block.config.sigma_min))
                | (sigma > float(block.config.sigma_max))
            ).item()
        ):
            raise ValueError("fixed EDM sigma is outside the model range")
    elif timesteps is None:
        normal = torch.randn(batch, dtype=clean.dtype, device=clean.device)
        sigma = torch.exp(
            float(block.config.log_sigma_mean)
            + float(block.config.log_sigma_std) * normal
        ).clamp(float(block.config.sigma_min), float(block.config.sigma_max))
    else:
        timestep = torch.as_tensor(
            timesteps, dtype=torch.long, device=clean.device
        ).reshape(-1)
        if tuple(timestep.shape) != (batch,):
            raise ValueError("fixed EDM timestep shape changed")
        if bool(torch.any((timestep < 0) | (timestep >= int(total_timesteps))).item()):
            raise ValueError("fixed EDM timestep is outside calibration range")
        quantile = (
            timestep.to(dtype=clean.dtype) + 0.5
        ) / float(total_timesteps)
        normal = np.sqrt(2.0) * torch.erfinv(2.0 * quantile - 1.0)
        sigma = torch.exp(
            float(block.config.log_sigma_mean)
            + float(block.config.log_sigma_std) * normal
        ).clamp(float(block.config.sigma_min), float(block.config.sigma_max))
    noise = torch.randn_like(clean) if fixed_noise is None else fixed_noise
    if tuple(noise.shape) != tuple(clean.shape):
        raise ValueError("fixed EDM noise shape changed")
    sigma_view = sigma.reshape(batch, *((1,) * (clean.ndim - 1)))
    noisy = clean + sigma_view * noise
    denoised = block.denoise(
        noisy, condition, sigma, context_latent=context_latent
    )
    weight = (
        sigma.square() + float(block.config.sigma_data) ** 2
    ) / (sigma * float(block.config.sigma_data)).square()
    error = (denoised - clean).square().flatten(1).mean(dim=1)
    return (weight * error).mean(), sigma.mean()
