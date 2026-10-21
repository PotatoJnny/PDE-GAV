from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
    _physical_fourier_basis,
    _rho_max_mode,
    join_active_latent,
    split_active_latent,
)


SHARED_LATENT_NORMALIZATION_SCHEMA = (
    "vpfp_shared_global_latent_channel_normalization_v1"
)
SHARED_CODEC_CHECKPOINT_SCHEMA = "vpfp_shared_global_solution_codec_checkpoint_v1"


@dataclass(frozen=True)
class VPFPSharedGlobalLatentCodecConfig:
    """Joint periodic solution codec for density and kinetic microstructure.

    ``latent_config.local_channels`` may be the output channels of the existing
    nonlinear equivariant micro codec. The shared codec therefore coordinates
    macro and micro fields without returning to a fixed small velocity basis.
    """

    latent_config: VPFPStructuredDiffusionConfig
    grid_size: int = 13
    hidden_channels: int = 128
    latent_channels: int = 64
    encoder_depth: int = 3
    decoder_depth: int = 3
    dtype: str = "float32"
    representation: str = "vpfp_shared_periodic_solution_latent_v1"

    @property
    def physical_input_channels(self) -> int:
        latent = self.latent_config
        return int(latent.rho_time_count) + int(
            latent.local_channels * latent.local_time_count
        )

    def validate(self) -> None:
        self.latent_config.validate()
        if min(
            int(self.grid_size),
            int(self.hidden_channels),
            int(self.latent_channels),
            int(self.encoder_depth),
            int(self.decoder_depth),
        ) < 1:
            raise ValueError("shared latent codec dimensions must be positive")
        maximum_mode = max(
            _rho_max_mode(self.latent_config),
            int(self.latent_config.local_spatial_mode),
        )
        if int(self.grid_size) < 2 * maximum_mode + 1:
            raise ValueError("shared physical grid aliases represented modes")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("shared latent codec dtype is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_config": self.latent_config.to_dict(),
            "grid_size": int(self.grid_size),
            "hidden_channels": int(self.hidden_channels),
            "latent_channels": int(self.latent_channels),
            "encoder_depth": int(self.encoder_depth),
            "decoder_depth": int(self.decoder_depth),
            "dtype": str(self.dtype),
            "representation": str(self.representation),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPSharedGlobalLatentCodecConfig":
        values = dict(payload)
        values["latent_config"] = VPFPStructuredDiffusionConfig.from_dict(
            dict(values["latent_config"])
        )
        config = cls(**values)
        config.validate()
        return config


class VPFPSharedGlobalLatentCodec(nn.Module):
    @staticmethod
    def build(
        config: VPFPSharedGlobalLatentCodecConfig | Mapping[str, Any],
    ) -> "VPFPSharedGlobalLatentCodec":
        parsed = (
            config
            if isinstance(config, VPFPSharedGlobalLatentCodecConfig)
            else VPFPSharedGlobalLatentCodecConfig.from_dict(config)
        )
        parsed.validate()
        return VPFPSharedGlobalLatentCodec(parsed)

    def __init__(self, config: VPFPSharedGlobalLatentCodecConfig):
        super().__init__()
        config.validate()
        self.config = config
        latent = config.latent_config
        rho_basis = _physical_fourier_basis(
            torch,
            max_mode=_rho_max_mode(latent),
            grid_size=int(config.grid_size),
            include_constant=False,
        )
        local_basis = _physical_fourier_basis(
            torch,
            max_mode=int(latent.local_spatial_mode),
            grid_size=int(config.grid_size),
            include_constant=True,
        )
        self.register_buffer("rho_basis", rho_basis.to(dtype=torch.float32))
        self.register_buffer("local_basis", local_basis.to(dtype=torch.float32))

        hidden = int(config.hidden_channels)
        self.input_projection = nn.Conv2d(
            int(config.physical_input_channels), hidden, kernel_size=1
        )
        self.encoder = nn.Sequential(
            *[_PeriodicResidualBlock(hidden) for _ in range(int(config.encoder_depth))]
        )
        self.to_latent = nn.Conv2d(hidden, int(config.latent_channels), kernel_size=1)
        self.from_latent = nn.Conv2d(int(config.latent_channels), hidden, kernel_size=1)
        self.decoder = nn.Sequential(
            *[_PeriodicResidualBlock(hidden) for _ in range(int(config.decoder_depth))]
        )
        self.output_projection = nn.Conv2d(
            hidden, int(config.physical_input_channels), kernel_size=1
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def active_to_physical_grid(self, active_latent) -> torch.Tensor:
        values = torch.as_tensor(active_latent)
        latent = self.config.latent_config
        rho, local = split_active_latent(values, latent)
        rho_grid = torch.einsum(
            "btf,gf->btg", rho, self.rho_basis.to(rho)
        )
        local_grid = torch.einsum(
            "bctf,gf->bctg", local, self.local_basis.to(local)
        )
        batch = int(values.shape[0])
        side = int(self.config.grid_size)
        return torch.cat(
            (
                rho_grid.reshape(batch, int(latent.rho_time_count), side, side),
                local_grid.reshape(
                    batch,
                    int(latent.local_channels * latent.local_time_count),
                    side,
                    side,
                ),
            ),
            dim=1,
        )

    def physical_grid_to_active(self, physical_grid) -> torch.Tensor:
        values = torch.as_tensor(physical_grid)
        latent = self.config.latent_config
        side = int(self.config.grid_size)
        expected = (
            int(values.shape[0]),
            int(self.config.physical_input_channels),
            side,
            side,
        )
        if tuple(values.shape) != expected:
            raise ValueError("shared physical solution grid has the wrong shape")
        batch = int(values.shape[0])
        flat = values.reshape(batch, int(values.shape[1]), side * side)
        rho_grid = flat[:, : int(latent.rho_time_count)]
        local_grid = flat[:, int(latent.rho_time_count) :].reshape(
            batch,
            int(latent.local_channels),
            int(latent.local_time_count),
            side * side,
        )
        rho = torch.einsum(
            "btg,gf->btf", rho_grid, self.rho_basis.to(rho_grid)
        )
        local = torch.einsum(
            "bctg,gf->bctf", local_grid, self.local_basis.to(local_grid)
        )
        return join_active_latent(rho, local, latent)

    def encode_grid(self, physical_grid) -> torch.Tensor:
        values = torch.as_tensor(physical_grid)
        hidden = self.input_projection(values)
        hidden = self.encoder(hidden)
        return self.to_latent(F.silu(hidden))

    def decode_grid(self, shared_latent) -> torch.Tensor:
        values = torch.as_tensor(shared_latent)
        expected = (
            int(values.shape[0]),
            int(self.config.latent_channels),
            int(self.config.grid_size),
            int(self.config.grid_size),
        )
        if tuple(values.shape) != expected:
            raise ValueError("shared global latent field has the wrong shape")
        hidden = self.from_latent(values)
        hidden = self.decoder(hidden)
        return self.output_projection(F.silu(hidden))

    def encode(self, active_latent) -> torch.Tensor:
        return self.encode_grid(self.active_to_physical_grid(active_latent))

    def decode(self, shared_latent) -> torch.Tensor:
        return self.physical_grid_to_active(self.decode_grid(shared_latent))

    def forward(self, active_latent) -> torch.Tensor:
        return self.decode(self.encode(active_latent))


def shared_global_latent_reconstruction_loss(
    model: VPFPSharedGlobalLatentCodec,
    active_latent,
    *,
    coordinate_reliability=None,
    unresolved_weight: float = 0.0,
    rho_weight: float = 1.0,
    micro_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reliability-aware shape loss; unresolved coarse frequencies are optional."""

    clean = torch.as_tensor(active_latent)
    reconstructed = model(clean)
    latent = model.config.latent_config
    if coordinate_reliability is None:
        weights = torch.ones(
            int(latent.active_latent_dim),
            dtype=clean.dtype,
            device=clean.device,
        )
    else:
        reliability = torch.as_tensor(
            coordinate_reliability, dtype=clean.dtype, device=clean.device
        )
        if tuple(reliability.shape) != (int(latent.active_latent_dim),):
            raise ValueError("coordinate reliability has the wrong shape")
        if bool(torch.any((reliability < 0.0) | (reliability > 1.0)).item()):
            raise ValueError("coordinate reliability must lie in [0,1]")
        weights = reliability + float(unresolved_weight) * (1.0 - reliability)
    rho_weights = weights[: int(latent.rho_latent_dim)]
    micro_weights = weights[int(latent.rho_latent_dim) :]
    rho_error = (reconstructed[:, : int(latent.rho_latent_dim)] - clean[:, : int(latent.rho_latent_dim)]).square()
    micro_error = (reconstructed[:, int(latent.rho_latent_dim) :] - clean[:, int(latent.rho_latent_dim) :]).square()
    rho_loss = torch.sum(rho_error * rho_weights) / (
        torch.sum(rho_weights).clamp_min(torch.finfo(clean.dtype).eps)
        * int(clean.shape[0])
    )
    micro_loss = torch.sum(micro_error * micro_weights) / (
        torch.sum(micro_weights).clamp_min(torch.finfo(clean.dtype).eps)
        * int(clean.shape[0])
    )
    loss = float(rho_weight) * rho_loss + float(micro_weight) * micro_loss
    return loss, {
        "loss": loss,
        "rho_reconstruction_loss": rho_loss,
        "micro_reconstruction_loss": micro_loss,
        "mean_coordinate_reliability": weights.mean(),
    }


def fit_shared_latent_normalization(
    shared_latents,
    *,
    minimum_std: float = 1.0e-5,
) -> dict[str, Any]:
    values = torch.as_tensor(shared_latents).detach()
    if values.ndim != 4:
        raise ValueError("shared latents must have shape [batch, channel, x, y]")
    reduce_dimensions = (0, 2, 3)
    mean = torch.mean(values, dim=reduce_dimensions)
    std = torch.std(values, dim=reduce_dimensions, unbiased=False).clamp_min(
        float(minimum_std)
    )
    return {
        "schema": SHARED_LATENT_NORMALIZATION_SCHEMA,
        "mean": mean.cpu().to(dtype=torch.float64).tolist(),
        "std": std.cpu().to(dtype=torch.float64).tolist(),
        "translation_invariant_fit": True,
    }


def normalize_shared_latent(shared_latents, normalization: Mapping[str, Any]):
    values = torch.as_tensor(shared_latents)
    mean, std = _shared_normalization_tensors(values, normalization)
    return (values - mean) / std


def denormalize_shared_latent(shared_latents, normalization: Mapping[str, Any]):
    values = torch.as_tensor(shared_latents)
    mean, std = _shared_normalization_tensors(values, normalization)
    return values * std + mean


def save_shared_global_latent_codec_checkpoint(
    path: str | Path,
    *,
    model: VPFPSharedGlobalLatentCodec,
    train_config: Mapping[str, Any],
    step: int,
    optimizer_state: Mapping[str, Any] | None = None,
    ema_state_dict: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SHARED_CODEC_CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "answer_first_contract": {
                "hidden_continuous_solution_is_primary": True,
                "equation_first_observations_only_calibrate_solution_shape": True,
                "equation_first_source_is_training_target": False,
                "equation_first_dynamics_are_generation_constraints": False,
            },
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_shared_global_latent_codec_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[VPFPSharedGlobalLatentCodec, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != SHARED_CODEC_CHECKPOINT_SCHEMA:
        raise ValueError(
            "unexpected shared global solution codec checkpoint schema"
        )
    contract = dict(payload.get("answer_first_contract") or {})
    if not bool(
        contract.get("hidden_continuous_solution_is_primary")
        and contract.get(
            "equation_first_observations_only_calibrate_solution_shape"
        )
        and not contract.get("equation_first_source_is_training_target")
        and not contract.get("equation_first_dynamics_are_generation_constraints")
    ):
        raise ValueError("shared solution codec violates answer-first contract")
    model = VPFPSharedGlobalLatentCodec.build(payload["model_config"]).to(
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


class _PeriodicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(int(channels))
        self.norm1 = nn.GroupNorm(groups, int(channels))
        self.conv1 = nn.Conv2d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm2 = nn.GroupNorm(groups, int(channels))
        self.conv2 = nn.Conv2d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, values):
        residual = self.conv1(F.silu(self.norm1(values)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return values + residual / np.sqrt(2.0)


def _shared_normalization_tensors(values, normalization):
    if normalization.get("schema") != SHARED_LATENT_NORMALIZATION_SCHEMA:
        raise ValueError("unexpected shared latent normalization schema")
    mean = torch.as_tensor(
        normalization["mean"], dtype=values.dtype, device=values.device
    )[None, :, None, None]
    std = torch.as_tensor(
        normalization["std"], dtype=values.dtype, device=values.device
    )[None, :, None, None]
    if int(mean.shape[1]) != int(values.shape[1]) or bool(torch.any(std <= 0.0)):
        raise ValueError("shared latent normalization shape is invalid")
    return mean, std


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if int(channels) % groups == 0:
            return groups
    return 1
