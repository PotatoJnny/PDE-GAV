from __future__ import annotations

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
from agent_verifier.vpfp.continuous_v2 import _real_fourier_half_modes
from agent_verifier.vpfp.equivariant_code_edm import (
    _edm_block_loss,
    _sample_edm_block,
)
from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
    _physical_fourier_basis,
    _rho_max_mode,
    join_active_latent,
    split_active_latent,
)


CHECKPOINT_SCHEMA = "vpfp_direct_macro_micro_edm_checkpoint_v1"


@dataclass(frozen=True)
class VPFPDirectMacroMicroEDMConfig:
    latent_config: VPFPStructuredDiffusionConfig
    condition_dim: int = 2
    macro_width: int = 256
    macro_layers: int = 6
    micro_width: int = 384
    micro_layers: int = 8
    attention_heads: int = 8
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
    def rho_side(self) -> int:
        return 2 * _rho_max_mode(self.latent_config) + 1

    @property
    def micro_side(self) -> int:
        return 2 * int(self.latent_config.local_spatial_mode) + 1

    @property
    def macro_dim(self) -> int:
        return int(self.latent_config.rho_latent_dim)

    @property
    def micro_dim(self) -> int:
        return int(self.latent_config.local_latent_dim)

    @property
    def micro_residual_output_rank_cap(self) -> int:
        return min(int(self.micro_width), int(self.latent_config.local_channels))

    @property
    def has_lossless_representation(self) -> bool:
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
        self.latent_config.validate()
        dimensions = (
            self.condition_dim,
            self.macro_width,
            self.macro_layers,
            self.micro_width,
            self.micro_layers,
            self.attention_heads,
            self.timesteps,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("direct macro/micro EDM dimensions must be positive")
        if int(self.condition_dim) != int(self.latent_config.condition_dim):
            raise ValueError("EDM and structured condition dimensions disagree")
        if int(self.macro_width) % int(self.attention_heads):
            raise ValueError("macro width must be divisible by attention heads")
        if int(self.micro_width) % int(self.attention_heads):
            raise ValueError("micro width must be divisible by attention heads")
        if (
            not np.isfinite(float(self.axial_mlp_ratio))
            or float(self.axial_mlp_ratio) <= 0.0
        ):
            raise ValueError("axial MLP ratio must be positive and finite")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("unsupported EDM dtype")
        if (
            int(self.latent_config.rho_time_count)
            != int(self.latent_config.local_time_count)
        ):
            raise ValueError(
                "v1 direct macro/micro context requires matching time features"
            )
        self.edm_config()

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_config": self.latent_config.to_dict(),
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "latent_config"
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPDirectMacroMicroEDMConfig":
        payload = dict(data)
        payload["latent_config"] = VPFPStructuredDiffusionConfig.from_dict(
            dict(payload["latent_config"])
        )
        config = cls(**payload)
        config.validate()
        return config


class VPFPDirectMacroMicroEDM(nn.Module):
    @staticmethod
    def build(config: VPFPDirectMacroMicroEDMConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPDirectMacroMicroEDMConfig)
            else VPFPDirectMacroMicroEDMConfig.from_dict(dict(config))
        )
        cfg.validate()
        return VPFPDirectMacroMicroEDM(cfg)

    def __init__(self, config: VPFPDirectMacroMicroEDMConfig):
        super().__init__()
        config.validate()
        self.config = config
        latent = config.latent_config
        edm = config.edm_config()
        self.macro_model = ConditionalAxialLatentEDM(
            (
                int(latent.rho_time_count),
                1,
                1,
                config.rho_side,
                config.rho_side,
            ),
            1,
            global_channels=int(config.condition_dim),
            width=int(config.macro_width),
            depth=int(config.macro_layers),
            heads=int(config.attention_heads),
            config=edm,
            mlp_ratio=float(config.axial_mlp_ratio),
            dropout=float(config.dropout),
            spatial_translation_equivariant=True,
        )
        self.micro_model = ConditionalAxialLatentEDM(
            (
                1,
                int(latent.local_time_count),
                1,
                config.micro_side,
                config.micro_side,
            ),
            int(latent.local_channels),
            global_channels=int(config.condition_dim),
            width=int(config.micro_width),
            depth=int(config.micro_layers),
            heads=int(config.attention_heads),
            config=edm,
            context_latent_channels=1,
            mlp_ratio=float(config.axial_mlp_ratio),
            dropout=float(config.dropout),
            spatial_translation_equivariant=True,
        )
        rho_basis = _physical_fourier_basis(
            torch,
            max_mode=_rho_max_mode(latent),
            grid_size=config.rho_side,
            include_constant=False,
        )
        micro_basis = _physical_fourier_basis(
            torch,
            max_mode=int(latent.local_spatial_mode),
            grid_size=config.micro_side,
            include_constant=True,
        )
        self.register_buffer("rho_basis", rho_basis.to(dtype=torch.float32))
        self.register_buffer("micro_basis", micro_basis.to(dtype=torch.float32))
        self.register_buffer(
            "rho_to_micro_context_basis",
            _rho_to_micro_context_basis(config).to(dtype=torch.float32),
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def rho_to_grid(self, rho):
        values = torch.as_tensor(rho)
        latent = self.config.latent_config
        expected = (int(values.shape[0]), int(latent.rho_latent_dim))
        if values.ndim != 2 or tuple(values.shape) != expected:
            raise ValueError("rho code shape does not match direct EDM config")
        spectral = values.reshape(
            int(values.shape[0]),
            int(latent.rho_time_count),
            int(latent.rho_feature_count),
        )
        grid = _orthogonal_einsum(
            "btf,gf->btg", spectral, self.rho_basis.to(spectral)
        )
        return grid.reshape(
            int(values.shape[0]),
            int(latent.rho_time_count),
            1,
            1,
            self.config.rho_side,
            self.config.rho_side,
            1,
        )

    def rho_from_grid(self, rho_grid):
        values = torch.as_tensor(rho_grid)
        latent = self.config.latent_config
        expected = (
            int(values.shape[0]),
            int(latent.rho_time_count),
            1,
            1,
            self.config.rho_side,
            self.config.rho_side,
            1,
        )
        if tuple(values.shape) != expected:
            raise ValueError("rho physical grid shape changed")
        grid = values.reshape(
            int(values.shape[0]),
            int(latent.rho_time_count),
            self.config.rho_side**2,
        )
        spectral = _orthogonal_einsum(
            "btg,gf->btf", grid, self.rho_basis.to(grid)
        )
        return spectral.reshape(
            int(values.shape[0]), int(latent.rho_latent_dim)
        )

    def micro_to_grid(self, micro):
        values = torch.as_tensor(micro)
        latent = self.config.latent_config
        expected = (int(values.shape[0]), int(latent.local_latent_dim))
        if values.ndim != 2 or tuple(values.shape) != expected:
            raise ValueError("micro code shape does not match direct EDM config")
        spectral = values.reshape(
            int(values.shape[0]),
            int(latent.local_channels),
            int(latent.local_time_count),
            int(latent.local_spatial_feature_count),
        )
        grid = _orthogonal_einsum(
            "bctf,gf->btgc", spectral, self.micro_basis.to(spectral)
        )
        return grid.reshape(
            int(values.shape[0]),
            1,
            int(latent.local_time_count),
            1,
            self.config.micro_side,
            self.config.micro_side,
            int(latent.local_channels),
        )

    def micro_from_grid(self, micro_grid):
        values = torch.as_tensor(micro_grid)
        latent = self.config.latent_config
        expected = (
            int(values.shape[0]),
            1,
            int(latent.local_time_count),
            1,
            self.config.micro_side,
            self.config.micro_side,
            int(latent.local_channels),
        )
        if tuple(values.shape) != expected:
            raise ValueError("micro physical grid shape changed")
        grid = values.reshape(
            int(values.shape[0]),
            int(latent.local_time_count),
            self.config.micro_side**2,
            int(latent.local_channels),
        )
        spectral = _orthogonal_einsum(
            "btgc,gf->bctf", grid, self.micro_basis.to(grid)
        )
        return spectral.reshape(
            int(values.shape[0]), int(latent.local_latent_dim)
        )

    def rho_to_micro_context(self, rho):
        values = torch.as_tensor(rho)
        latent = self.config.latent_config
        if (
            values.ndim != 2
            or int(values.shape[1]) != int(latent.rho_latent_dim)
        ):
            raise ValueError("rho context shape does not match direct EDM config")
        spectral = values.reshape(
            int(values.shape[0]),
            int(latent.rho_time_count),
            int(latent.rho_feature_count),
        )
        grid = _orthogonal_einsum(
            "btf,gf->btg",
            spectral,
            self.rho_to_micro_context_basis.to(spectral),
        )
        return grid.reshape(
            int(values.shape[0]),
            1,
            int(latent.local_time_count),
            1,
            self.config.micro_side,
            self.config.micro_side,
            1,
        )


def direct_macro_micro_edm_loss(
    model,
    clean_active_latent,
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
    clean = torch.as_tensor(
        clean_active_latent, dtype=parameter.dtype, device=parameter.device
    )
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    latent = model.config.latent_config
    batch = int(clean.shape[0])
    if tuple(clean.shape) != (batch, int(latent.active_latent_dim)):
        raise ValueError("clean active latent shape does not match direct EDM")
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("condition shape does not match direct EDM")
    weights = (float(macro_weight), float(micro_weight))
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("EDM loss weights must be finite and nonnegative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one EDM factor loss weight must be positive")
    augmentation = float(macro_condition_augmentation_std)
    if not np.isfinite(augmentation) or augmentation < 0.0:
        raise ValueError("macro condition augmentation must be nonnegative")

    rho, local = split_active_latent(clean, latent)
    rho = rho.reshape(batch, -1)
    local = local.reshape(batch, -1)
    zero = torch.zeros((), dtype=parameter.dtype, device=parameter.device)
    macro_loss = zero
    micro_loss = zero
    macro_sigma = zero
    micro_sigma = zero
    if weights[0] > 0.0:
        macro_loss, macro_sigma = _edm_block_loss(
            model.macro_model,
            model.rho_to_grid(rho),
            condition,
            timesteps=macro_timesteps,
            fixed_sigmas=macro_sigmas,
            fixed_noise=(
                model.rho_to_grid(
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
        rho_context = rho
        if augmentation:
            rho_context = rho + augmentation * torch.randn_like(rho)
        micro_loss, micro_sigma = _edm_block_loss(
            model.micro_model,
            model.micro_to_grid(local),
            condition,
            context_latent=model.rho_to_micro_context(rho_context),
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


def sample_direct_macro_micro_edm(
    model,
    conditions,
    *,
    macro_condition=None,
    sample_steps: int = 32,
    macro_sample_steps: int | None = None,
    micro_sample_steps: int | None = None,
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
    macro_generator: torch.Generator | None = None,
    micro_generator: torch.Generator | None = None,
    micro_output_scale: float = 1.0,
):
    parameter = next(model.parameters())
    macro_random = generator if macro_generator is None else macro_generator
    micro_random = generator if micro_generator is None else micro_generator
    micro_scale = float(micro_output_scale)
    if not np.isfinite(micro_scale) or micro_scale <= 0.0:
        raise ValueError("micro output scale must be positive and finite")
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(condition.shape[0])
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("condition shape does not match direct EDM")
    macro_steps = int(
        sample_steps if macro_sample_steps is None else macro_sample_steps
    )
    micro_steps = int(
        sample_steps if micro_sample_steps is None else micro_sample_steps
    )
    if macro_condition is None:
        rho_grid = _sample_edm_block(
            model.macro_model,
            condition,
            steps=macro_steps,
            device=parameter.device,
            generator=macro_random,
            guidance=macro_guidance,
            guidance_scale=float(macro_guidance_scale),
            churn=float(macro_churn),
            churn_min=float(churn_min),
            churn_max=float(churn_max),
            churn_noise=float(churn_noise),
        )
        rho = model.rho_from_grid(rho_grid)
    else:
        rho = torch.as_tensor(
            macro_condition, dtype=parameter.dtype, device=parameter.device
        )
        if tuple(rho.shape) != (
            batch,
            int(model.config.latent_config.rho_latent_dim),
        ):
            raise ValueError("fixed macro condition is invalid")
    micro_grid = _sample_edm_block(
        model.micro_model,
        condition,
        context_latent=model.rho_to_micro_context(rho),
        steps=micro_steps,
        device=parameter.device,
        generator=micro_random,
        guidance=micro_guidance,
        guidance_scale=float(micro_guidance_scale),
        churn=float(micro_churn),
        churn_min=float(churn_min),
        churn_max=float(churn_max),
        churn_noise=float(churn_noise),
    )
    latent = model.config.latent_config
    rho_tensor = rho.reshape(
        batch, int(latent.rho_time_count), int(latent.rho_feature_count)
    )
    local = model.micro_from_grid(micro_grid).reshape(
        batch,
        int(latent.local_channels),
        int(latent.local_time_count),
        int(latent.local_spatial_feature_count),
    )
    local = local * micro_scale
    return join_active_latent(rho_tensor, local, latent)


def sample_direct_macro_rho(
    model,
    conditions,
    *,
    sample_steps: int = 32,
    guidance: Callable | None = None,
    guidance_scale: float = 0.0,
    churn: float = 0.0,
    churn_min: float = 0.0,
    churn_max: float = float("inf"),
    churn_noise: float = 1.0,
    generator: torch.Generator | None = None,
):
    """Sample only the lossless macro factor without evaluating the micro EDM."""

    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(condition.shape[0])
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("condition shape does not match direct EDM")
    rho_grid = _sample_edm_block(
        model.macro_model,
        condition,
        steps=int(sample_steps),
        device=parameter.device,
        generator=generator,
        guidance=guidance,
        guidance_scale=float(guidance_scale),
        churn=float(churn),
        churn_min=float(churn_min),
        churn_max=float(churn_max),
        churn_noise=float(churn_noise),
    )
    return model.rho_from_grid(rho_grid)


def save_direct_macro_micro_edm_checkpoint(
    path: str | Path,
    *,
    model,
    active_latent_normalization: dict[str, Any],
    condition_normalization: dict[str, Any],
    answer_first_build_report: str | Path,
    answer_first_build_report_sha256: str,
    decoder_checkpoint: str | Path,
    decoder_checkpoint_sha256: str,
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
            "active_latent_normalization": dict(active_latent_normalization),
            "condition_normalization": dict(condition_normalization),
            "answer_first_build_report": str(
                Path(answer_first_build_report).resolve()
            ),
            "answer_first_build_report_sha256": str(
                answer_first_build_report_sha256
            ),
            "decoder_checkpoint": str(Path(decoder_checkpoint).resolve()),
            "decoder_checkpoint_sha256": str(decoder_checkpoint_sha256),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_direct_macro_micro_edm_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected direct macro/micro EDM checkpoint schema")
    model = VPFPDirectMacroMicroEDM.build(payload["model_config"]).to(
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


def _orthogonal_einsum(equation: str, *operands):
    use_ieee_fp32 = any(
        value.device.type == "cuda" and value.dtype == torch.float32
        for value in operands
    )
    if not use_ieee_fp32:
        return torch.einsum(equation, *operands)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.einsum(equation, *operands)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _rho_to_micro_context_basis(config: VPFPDirectMacroMicroEDMConfig):
    latent = config.latent_config
    rho_max_mode = _rho_max_mode(latent)
    rho_modes = _real_fourier_half_modes(rho_max_mode, rho_max_mode)
    micro_mode = int(latent.local_spatial_mode)
    micro_modes = _real_fourier_half_modes(micro_mode, micro_mode)
    rho_indices = {mode: index for index, mode in enumerate(rho_modes)}
    micro_basis = _physical_fourier_basis(
        torch,
        max_mode=micro_mode,
        grid_size=config.micro_side,
        include_constant=True,
    )
    result = torch.zeros(
        config.micro_side**2,
        int(latent.rho_feature_count),
        dtype=micro_basis.dtype,
    )
    for micro_index, mode in enumerate(micro_modes):
        rho_index = rho_indices.get(mode)
        if rho_index is None:
            raise ValueError("micro spatial mode is unavailable in rho context")
        result[:, 2 * rho_index : 2 * rho_index + 2] = micro_basis[
            :, 1 + 2 * micro_index : 1 + 2 * micro_index + 2
        ]
    return result
