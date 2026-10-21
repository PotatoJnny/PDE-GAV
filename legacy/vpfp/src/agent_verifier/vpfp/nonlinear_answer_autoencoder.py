from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


NONLINEAR_ANSWER_AE_CHECKPOINT_SCHEMA = (
    "vpfp_nonlinear_joint_answer_autoencoder_checkpoint_v1"
)
NONLINEAR_ANSWER_LATENT_NORMALIZATION_SCHEMA = (
    "vpfp_nonlinear_joint_answer_latent_normalization_v1"
)
NONLINEAR_ANSWER_LATENT_EDM_CHECKPOINT_SCHEMA = (
    "vpfp_nonlinear_joint_answer_latent_edm_checkpoint_v1"
)


@dataclass(frozen=True)
class VPFPNonlinearAnswerAutoencoderConfig:
    """Grouped periodic autoencoder for a complete VPFP answer field.

    The spatial latent field preserves torus phase and locality. The global
    vector is translation invariant and is decoded jointly into both macro and
    micro channels. Both codes are functions of the complete answer.
    """

    grid_size: int = 13
    macro_input_channels: int = 34
    micro_input_channels: int = 512
    answer_coordinate_dim: int = 0
    macro_latent_channels: int = 48
    micro_latent_channels: int = 144
    global_latent_dim: int = 128
    hidden_channels: int = 512
    macro_stem_channels: int = 128
    bottleneck_channels: int = 128
    encoder_depth: int = 4
    decoder_depth: int = 4
    dtype: str = "float32"
    representation: str = "vpfp_nonlinear_joint_answer_latent_v1"

    @property
    def input_channels(self) -> int:
        return int(self.macro_input_channels + self.micro_input_channels)

    @property
    def local_latent_channels(self) -> int:
        return int(self.macro_latent_channels + self.micro_latent_channels)

    @property
    def micro_stem_channels(self) -> int:
        return int(self.hidden_channels - self.macro_stem_channels)

    @property
    def input_dimension(self) -> int:
        if int(self.answer_coordinate_dim) > 0:
            return int(self.answer_coordinate_dim)
        side = int(self.grid_size)
        return int(self.input_channels) * side * side

    @property
    def compression_ratio(self) -> float:
        side = int(self.grid_size)
        latent_dim = int(self.local_latent_channels) * side * side + int(
            self.global_latent_dim
        )
        return float(self.input_dimension / latent_dim)

    def validate(self) -> None:
        dimensions = (
            self.grid_size,
            self.macro_input_channels,
            self.micro_input_channels,
            self.macro_latent_channels,
            self.micro_latent_channels,
            self.global_latent_dim,
            self.hidden_channels,
            self.macro_stem_channels,
            self.bottleneck_channels,
            self.encoder_depth,
            self.decoder_depth,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("answer autoencoder dimensions must be positive")
        if int(self.answer_coordinate_dim) < 0:
            raise ValueError("answer coordinate dimension cannot be negative")
        if not 0 < int(self.macro_stem_channels) < int(self.hidden_channels):
            raise ValueError("answer autoencoder needs strict grouped stems")
        if int(self.macro_stem_channels) < int(self.macro_latent_channels):
            raise ValueError("macro stem is narrower than the macro latent")
        if int(self.micro_stem_channels) < int(self.micro_latent_channels):
            raise ValueError("micro stem is narrower than the micro latent")
        if int(self.bottleneck_channels) > int(self.hidden_channels):
            raise ValueError("residual bottleneck cannot exceed hidden width")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("answer autoencoder dtype is unsupported")
        if float(self.compression_ratio) <= 1.0:
            raise ValueError("answer autoencoder must be a strict bottleneck")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPNonlinearAnswerAutoencoderConfig":
        config = cls(**dict(payload))
        config.validate()
        return config


class VPFPNonlinearAnswerAutoencoder(nn.Module):
    @staticmethod
    def build(
        config: VPFPNonlinearAnswerAutoencoderConfig | Mapping[str, Any],
    ) -> "VPFPNonlinearAnswerAutoencoder":
        parsed = (
            config
            if isinstance(config, VPFPNonlinearAnswerAutoencoderConfig)
            else VPFPNonlinearAnswerAutoencoderConfig.from_dict(config)
        )
        parsed.validate()
        return VPFPNonlinearAnswerAutoencoder(parsed)

    def __init__(self, config: VPFPNonlinearAnswerAutoencoderConfig):
        super().__init__()
        config.validate()
        self.config = config
        macro_width = int(config.macro_stem_channels)
        micro_width = int(config.micro_stem_channels)
        hidden = int(config.hidden_channels)

        self.macro_encoder_stem = nn.Conv2d(
            int(config.macro_input_channels), macro_width, kernel_size=1
        )
        self.micro_encoder_stem = nn.Conv2d(
            int(config.micro_input_channels), micro_width, kernel_size=1
        )
        self.encoder = nn.Sequential(
            *[
                _PeriodicContextResidualBlock(
                    hidden, int(config.bottleneck_channels)
                )
                for _ in range(int(config.encoder_depth))
            ]
        )
        self.to_macro_latent = nn.Conv2d(
            hidden, int(config.macro_latent_channels), kernel_size=1
        )
        self.to_micro_latent = nn.Conv2d(
            hidden, int(config.micro_latent_channels), kernel_size=1
        )
        self.to_global_latent = nn.Sequential(
            nn.LayerNorm(2 * hidden, eps=1.0e-6),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(config.global_latent_dim)),
        )

        self.from_macro_latent = nn.Conv2d(
            int(config.macro_latent_channels), macro_width, kernel_size=1
        )
        self.from_micro_latent = nn.Conv2d(
            int(config.micro_latent_channels), micro_width, kernel_size=1
        )
        self.from_global_latent = nn.Sequential(
            nn.LayerNorm(int(config.global_latent_dim), eps=1.0e-6),
            nn.Linear(int(config.global_latent_dim), hidden),
        )
        self.decoder = nn.Sequential(
            *[
                _PeriodicContextResidualBlock(
                    hidden, int(config.bottleneck_channels)
                )
                for _ in range(int(config.decoder_depth))
            ]
        )
        self.macro_output = nn.Conv2d(
            hidden, int(config.macro_input_channels), kernel_size=1
        )
        self.micro_output = nn.Conv2d(
            hidden, int(config.micro_input_channels), kernel_size=1
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def encode_field(self, answer_field) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.as_tensor(answer_field)
        self._validate_field(values)
        boundary = int(self.config.macro_input_channels)
        macro = self.macro_encoder_stem(values[:, :boundary])
        micro = self.micro_encoder_stem(values[:, boundary:])
        hidden = self.encoder(torch.cat((macro, micro), dim=1))
        local = torch.cat(
            (
                self.to_macro_latent(F.silu(hidden)),
                self.to_micro_latent(F.silu(hidden)),
            ),
            dim=1,
        )
        local = local - torch.mean(local, dim=(2, 3), keepdim=True)
        mean = torch.mean(hidden, dim=(2, 3))
        root_mean_square = torch.sqrt(
            torch.mean(hidden.square(), dim=(2, 3)) + 1.0e-8
        )
        global_code = self.to_global_latent(
            torch.cat((mean, root_mean_square), dim=1)
        )
        return local, global_code

    def decode_codes(self, local_latent, global_latent) -> torch.Tensor:
        local = torch.as_tensor(local_latent)
        global_code = torch.as_tensor(
            global_latent, dtype=local.dtype, device=local.device
        )
        self._validate_codes(local, global_code)
        local = local - torch.mean(local, dim=(2, 3), keepdim=True)
        boundary = int(self.config.macro_latent_channels)
        macro = self.from_macro_latent(local[:, :boundary])
        micro = self.from_micro_latent(local[:, boundary:])
        hidden = torch.cat((macro, micro), dim=1)
        hidden = hidden + self.from_global_latent(global_code)[:, :, None, None]
        hidden = self.decoder(hidden)
        hidden = F.silu(hidden)
        return torch.cat(
            (self.macro_output(hidden), self.micro_output(hidden)), dim=1
        )

    def forward(self, answer_field) -> torch.Tensor:
        local, global_code = self.encode_field(answer_field)
        return self.decode_codes(local, global_code)

    def _validate_field(self, values: torch.Tensor) -> None:
        expected = (
            int(values.shape[0]),
            int(self.config.input_channels),
            int(self.config.grid_size),
            int(self.config.grid_size),
        )
        if tuple(values.shape) != expected:
            raise ValueError("answer field shape does not match autoencoder")

    def _validate_codes(
        self, local: torch.Tensor, global_code: torch.Tensor
    ) -> None:
        batch = int(local.shape[0])
        expected_local = (
            batch,
            int(self.config.local_latent_channels),
            int(self.config.grid_size),
            int(self.config.grid_size),
        )
        expected_global = (batch, int(self.config.global_latent_dim))
        if tuple(local.shape) != expected_local:
            raise ValueError("local answer latent has the wrong shape")
        if tuple(global_code.shape) != expected_global:
            raise ValueError("global answer latent has the wrong shape")


def nonlinear_answer_reconstruction_loss(
    model: VPFPNonlinearAnswerAutoencoder,
    clean_field,
    *,
    macro_fraction: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 < float(macro_fraction) < 1.0:
        raise ValueError("macro reconstruction fraction must lie in (0,1)")
    clean = torch.as_tensor(clean_field)
    reconstructed = model(clean)
    boundary = int(model.config.macro_input_channels)
    macro_loss = torch.mean(
        (reconstructed[:, :boundary] - clean[:, :boundary]).square()
    )
    micro_loss = torch.mean(
        (reconstructed[:, boundary:] - clean[:, boundary:]).square()
    )
    fraction = float(macro_fraction)
    loss = fraction * macro_loss + (1.0 - fraction) * micro_loss
    return loss, {
        "loss": loss,
        "macro_reconstruction_loss": macro_loss,
        "micro_reconstruction_loss": micro_loss,
    }


def fit_nonlinear_answer_latent_normalization(
    local_latents,
    global_latents,
    *,
    minimum_std: float = 1.0e-5,
) -> dict[str, Any]:
    local = torch.as_tensor(local_latents).detach()
    global_code = torch.as_tensor(global_latents).detach()
    if local.ndim != 4 or global_code.ndim != 2:
        raise ValueError("answer latent normalization shapes are invalid")
    if int(local.shape[0]) != int(global_code.shape[0]):
        raise ValueError("local and global answer latent batches disagree")
    local_mean = torch.mean(local, dim=(0, 2, 3))
    local_std = torch.std(local, dim=(0, 2, 3), unbiased=False).clamp_min(
        float(minimum_std)
    )
    global_mean = torch.mean(global_code, dim=0)
    global_std = torch.std(global_code, dim=0, unbiased=False).clamp_min(
        float(minimum_std)
    )
    return {
        "schema": NONLINEAR_ANSWER_LATENT_NORMALIZATION_SCHEMA,
        "local_mean": local_mean.cpu().to(dtype=torch.float64).tolist(),
        "local_std": local_std.cpu().to(dtype=torch.float64).tolist(),
        "global_mean": global_mean.cpu().to(dtype=torch.float64).tolist(),
        "global_std": global_std.cpu().to(dtype=torch.float64).tolist(),
        "translation_invariant_local_fit": True,
        "generated_samples_used_for_fit": False,
    }


def normalize_nonlinear_answer_latents(
    local_latents, global_latents, normalization: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.as_tensor(local_latents)
    global_code = torch.as_tensor(
        global_latents, dtype=local.dtype, device=local.device
    )
    local_mean, local_std, global_mean, global_std = (
        _latent_normalization_tensors(local, global_code, normalization)
    )
    return (
        (local - local_mean) / local_std,
        (global_code - global_mean) / global_std,
    )


def denormalize_nonlinear_answer_latents(
    local_latents, global_latents, normalization: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.as_tensor(local_latents)
    global_code = torch.as_tensor(
        global_latents, dtype=local.dtype, device=local.device
    )
    local_mean, local_std, global_mean, global_std = (
        _latent_normalization_tensors(local, global_code, normalization)
    )
    return (
        local * local_std + local_mean,
        global_code * global_std + global_mean,
    )


def save_nonlinear_answer_autoencoder_checkpoint(
    path: str | Path,
    *,
    model: VPFPNonlinearAnswerAutoencoder,
    field_config: Mapping[str, Any],
    field_normalization: Mapping[str, Any],
    latent_normalization: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    train_config: Mapping[str, Any],
    step: int,
    optimizer_state: Mapping[str, Any] | None = None,
    ema_state_dict: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": NONLINEAR_ANSWER_AE_CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "field_config": dict(field_config),
            "field_normalization": dict(field_normalization),
            "latent_normalization": (
                None
                if latent_normalization is None
                else dict(latent_normalization)
            ),
            "provenance": dict(provenance),
            "answer_first_contract": {
                "primary_object": "continuous_hidden_solution_function",
                "reconstructed_object": "complete_answer_not_equation",
                "training_target": "answer_coordinates_only",
                "equation_first_source_used": False,
                "induced_source_used": False,
                "equation_dynamics_used": False,
                "unreliable_coarse_detail_treated_as_truth": False,
                "public_source_stage": "after_complete_answer_sampling_only",
            },
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_nonlinear_answer_autoencoder_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[VPFPNonlinearAnswerAutoencoder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != NONLINEAR_ANSWER_AE_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected nonlinear answer autoencoder schema")
    contract = dict(payload.get("answer_first_contract") or {})
    if not bool(
        contract.get("primary_object")
        == "continuous_hidden_solution_function"
        and contract.get("reconstructed_object")
        == "complete_answer_not_equation"
        and contract.get("training_target") == "answer_coordinates_only"
        and not contract.get("equation_first_source_used")
        and not contract.get("induced_source_used")
        and not contract.get("equation_dynamics_used")
        and not contract.get("unreliable_coarse_detail_treated_as_truth")
    ):
        raise ValueError("nonlinear answer autoencoder violates contract")
    model = VPFPNonlinearAnswerAutoencoder.build(payload["model_config"]).to(
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


class _PeriodicContextResidualBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        groups = _group_count(int(channels))
        inner_groups = _group_count(int(bottleneck))
        self.norm1 = nn.GroupNorm(groups, int(channels), eps=1.0e-6)
        self.reduce = nn.Conv2d(int(channels), int(bottleneck), kernel_size=1)
        self.norm2 = nn.GroupNorm(inner_groups, int(bottleneck), eps=1.0e-6)
        self.spatial = nn.Conv2d(
            int(bottleneck),
            int(bottleneck),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        context_width = max(16, int(bottleneck) // 2)
        self.context_gate = nn.Sequential(
            nn.Linear(int(bottleneck), context_width),
            nn.SiLU(),
            nn.Linear(context_width, int(bottleneck)),
        )
        self.expand = nn.Conv2d(int(bottleneck), int(channels), kernel_size=1)
        nn.init.zeros_(self.expand.weight)
        if self.expand.bias is not None:
            nn.init.zeros_(self.expand.bias)

    def forward(self, values):
        hidden = self.reduce(F.silu(self.norm1(values)))
        hidden = self.spatial(F.silu(self.norm2(hidden)))
        context = torch.mean(hidden, dim=(2, 3))
        gate = 2.0 * torch.sigmoid(self.context_gate(context))
        residual = self.expand(hidden * gate[:, :, None, None])
        return values + residual / math.sqrt(2.0)


def _latent_normalization_tensors(local, global_code, normalization):
    if normalization.get("schema") != (
        NONLINEAR_ANSWER_LATENT_NORMALIZATION_SCHEMA
    ):
        raise ValueError("unexpected nonlinear answer latent normalization")
    local_mean = torch.as_tensor(
        normalization["local_mean"], dtype=local.dtype, device=local.device
    )[None, :, None, None]
    local_std = torch.as_tensor(
        normalization["local_std"], dtype=local.dtype, device=local.device
    )[None, :, None, None]
    global_mean = torch.as_tensor(
        normalization["global_mean"],
        dtype=global_code.dtype,
        device=global_code.device,
    )[None, :]
    global_std = torch.as_tensor(
        normalization["global_std"],
        dtype=global_code.dtype,
        device=global_code.device,
    )[None, :]
    if (
        int(local_mean.shape[1]) != int(local.shape[1])
        or int(global_mean.shape[1]) != int(global_code.shape[1])
        or bool(torch.any(local_std <= 0.0).item())
        or bool(torch.any(global_std <= 0.0).item())
    ):
        raise ValueError("nonlinear answer latent normalization changed")
    return local_mean, local_std, global_mean, global_std


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if int(channels) % groups == 0:
            return groups
    return 1
