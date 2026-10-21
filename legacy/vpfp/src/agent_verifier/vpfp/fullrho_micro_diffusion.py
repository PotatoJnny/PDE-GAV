from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent_verifier.vpfp.continuous_diffusion import (
    diffusion_model_predictions,
    diffusion_schedule,
    diffusion_training_target,
    q_sample,
)
from agent_verifier.vpfp.equivariant_code_diffusion import (
    _AdaptiveTransformerBlock,
    _ConditionContext,
    _zero_output,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


CHECKPOINT_SCHEMA = "vpfp_fullrho_equivariant_micro_diffusion_checkpoint_v1"


@dataclass(frozen=True)
class VPFPFullRhoMicroDiffusionConfig:
    rho_time_count: int
    rho_feature_count: int
    micro_channels: int
    spatial_feature_count: int
    condition_dim: int = 2
    time_embedding_dim: int = 128
    width: int = 512
    layers: int = 8
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    dropout: float = 0.0
    timesteps: int = 200
    noise_schedule: str = "cosine"
    prediction_type: str = "v"
    dtype: str = "float32"

    @property
    def rho_dim(self) -> int:
        return int(self.rho_time_count) * int(self.rho_feature_count)

    @property
    def micro_dim(self) -> int:
        return int(self.micro_channels) * int(self.spatial_feature_count)

    @property
    def micro_output_rank_cap(self) -> int:
        return min(int(self.width), int(self.micro_channels))

    @property
    def has_full_rank_micro_output(self) -> bool:
        return self.micro_output_rank_cap == int(self.micro_channels)

    def validate(self) -> None:
        dimensions = (
            self.rho_time_count,
            self.rho_feature_count,
            self.micro_channels,
            self.spatial_feature_count,
            self.condition_dim,
            self.time_embedding_dim,
            self.width,
            self.layers,
            self.attention_heads,
            self.feedforward_multiplier,
            self.timesteps,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("full-rho micro diffusion dimensions must be positive")
        if int(self.spatial_feature_count) % 2 != 1:
            raise ValueError("micro Fourier layout must contain 1+2M features")
        if int(self.width) % int(self.attention_heads):
            raise ValueError("model width must be divisible by attention heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.noise_schedule not in {"linear", "cosine"}:
            raise ValueError("unsupported diffusion noise schedule")
        if self.prediction_type not in {"epsilon", "v"}:
            raise ValueError("unsupported diffusion prediction type")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("unsupported diffusion dtype")

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "VPFPFullRhoMicroDiffusionConfig":
        config = cls(**dict(data))
        config.validate()
        return config


class VPFPFullRhoMicroDenoiser:
    @staticmethod
    def build(config: VPFPFullRhoMicroDiffusionConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPFullRhoMicroDiffusionConfig)
            else VPFPFullRhoMicroDiffusionConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPFullRhoMicroDenoiser(_torch(), cfg)


class _VPFPFullRhoMicroDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    """Denoise nonlinear micro codes while attending to every rho coefficient."""

    def __init__(self, torch, config: VPFPFullRhoMicroDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        width = int(config.width)
        self.rho_input = torch.nn.Linear(int(config.rho_feature_count), width)
        self.micro_input = torch.nn.Linear(int(config.micro_channels), width)
        self.rho_position = torch.nn.Parameter(
            torch.randn(int(config.rho_time_count), width) / np.sqrt(float(width))
        )
        self.micro_position = torch.nn.Parameter(
            torch.randn(int(config.spatial_feature_count), width)
            / np.sqrt(float(width))
        )
        self.token_type = torch.nn.Parameter(
            torch.randn(2, width) / np.sqrt(float(width))
        )
        self.context = _ConditionContext(torch, config, width)
        self.blocks = torch.nn.ModuleList(
            [
                _AdaptiveTransformerBlock(
                    torch,
                    width=width,
                    heads=int(config.attention_heads),
                    multiplier=int(config.feedforward_multiplier),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.layers))
            ]
        )
        self.final_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        self.final_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(width, 2 * width)
        )
        self.output = torch.nn.Linear(width, int(config.micro_channels))
        _zero_output(torch, self.final_modulation[-1], self.output)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy_micro, rho_condition, timesteps, conditions):
        parameter = next(self.parameters())
        micro = self.torch.as_tensor(
            noisy_micro, dtype=parameter.dtype, device=parameter.device
        )
        rho = self.torch.as_tensor(
            rho_condition, dtype=parameter.dtype, device=parameter.device
        )
        if micro.ndim != 2 or int(micro.shape[1]) != int(self.config.micro_dim):
            raise ValueError("micro code shape does not match diffusion config")
        expected_rho = (int(micro.shape[0]), int(self.config.rho_dim))
        if tuple(rho.shape) != expected_rho:
            raise ValueError("full rho condition shape does not match micro batch")
        time, context = self.context(timesteps, conditions, reference=micro)
        if int(time.shape[0]) != int(micro.shape[0]):
            raise ValueError("diffusion timestep batch does not match micro batch")

        rho_tokens = self.rho_input(
            rho.reshape(
                int(rho.shape[0]),
                int(self.config.rho_time_count),
                int(self.config.rho_feature_count),
            )
        )
        rho_tokens = (
            rho_tokens
            + self.rho_position[None]
            + self.token_type[None, :1]
        )
        micro_tokens = micro.reshape(
            int(micro.shape[0]),
            int(self.config.micro_channels),
            int(self.config.spatial_feature_count),
        ).transpose(1, 2)
        micro_tokens = (
            self.micro_input(micro_tokens)
            + self.micro_position[None]
            + self.token_type[None, 1:2]
        )
        tokens = self.torch.cat((rho_tokens, micro_tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens, context)
        micro_tokens = tokens[:, int(self.config.rho_time_count) :]
        shift, scale = self.final_modulation(context).chunk(2, dim=-1)
        micro_tokens = self.final_norm(micro_tokens)
        micro_tokens = micro_tokens * (1.0 + scale[:, None]) + shift[:, None]
        prediction = self.output(micro_tokens).transpose(1, 2)
        return prediction.reshape(int(micro.shape[0]), -1)


def fullrho_micro_diffusion_loss(
    model,
    clean_micro,
    rho_condition,
    conditions,
    *,
    timesteps=None,
    noise=None,
    rho_condition_augmentation_std: float = 0.0,
    rho_condition_dropout_probability: float = 0.0,
    rho_dropout_mask=None,
):
    torch = _torch()
    parameter = next(model.parameters())
    micro = torch.as_tensor(
        clean_micro, dtype=parameter.dtype, device=parameter.device
    )
    rho = torch.as_tensor(
        rho_condition, dtype=parameter.dtype, device=parameter.device
    )
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(micro.shape[0])
    if tuple(micro.shape) != (batch, int(model.config.micro_dim)):
        raise ValueError("clean micro code shape does not match diffusion config")
    if tuple(rho.shape) != (batch, int(model.config.rho_dim)):
        raise ValueError("rho condition shape does not match diffusion config")
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("equation condition shape does not match diffusion config")

    timestep = _resolve_training_timesteps(
        torch,
        timesteps,
        batch=batch,
        total=int(model.config.timesteps),
        device=parameter.device,
    )
    epsilon = _resolve_noise(torch, noise, micro, parameter)
    schedule = diffusion_schedule(
        model.config, device=parameter.device, dtype=parameter.dtype
    )
    noisy_micro = q_sample(micro, timestep, epsilon, schedule)
    augmentation = float(rho_condition_augmentation_std)
    if not np.isfinite(augmentation) or augmentation < 0.0:
        raise ValueError("rho condition augmentation must be nonnegative")
    dropout_probability = float(rho_condition_dropout_probability)
    if (
        not np.isfinite(dropout_probability)
        or dropout_probability < 0.0
        or dropout_probability > 1.0
    ):
        raise ValueError("rho condition dropout probability must lie in [0,1]")
    rho_input = rho
    if augmentation:
        rho_input = rho + augmentation * torch.randn_like(rho)
    if rho_dropout_mask is None:
        dropout_mask = torch.rand(
            (batch, 1), device=parameter.device
        ) < dropout_probability
    else:
        dropout_mask = torch.as_tensor(
            rho_dropout_mask, dtype=torch.bool, device=parameter.device
        ).reshape(-1, 1)
        if tuple(dropout_mask.shape) != (batch, 1):
            raise ValueError("rho dropout mask must contain one value per case")
    rho_input = torch.where(dropout_mask, torch.zeros_like(rho_input), rho_input)
    prediction = model(noisy_micro, rho_input, timestep, condition)
    target = diffusion_training_target(
        micro,
        epsilon,
        timestep,
        schedule,
        str(model.config.prediction_type),
    )
    loss = torch.mean((prediction - target).square())
    return loss, {
        "loss": loss,
        "micro_noise_mse": loss,
        "micro_timestep_mean": timestep.to(dtype=parameter.dtype).mean(),
        "rho_condition_dropout_fraction": dropout_mask.to(
            dtype=parameter.dtype
        ).mean(),
    }


def sample_fullrho_micro_codes(
    model,
    rho_condition,
    conditions,
    *,
    initial_noise=None,
    sample_steps: int | None = None,
    eta: float = 0.0,
    rho_cfg_scale: float = 1.0,
    guidance: Callable | None = None,
    generator=None,
):
    torch = _torch()
    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    rho = torch.as_tensor(
        rho_condition, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(condition.shape[0])
    if tuple(condition.shape) != (batch, int(model.config.condition_dim)):
        raise ValueError("sample condition shape does not match diffusion config")
    if tuple(rho.shape) != (batch, int(model.config.rho_dim)):
        raise ValueError("sample rho shape does not match diffusion config")
    if initial_noise is None:
        current = torch.randn(
            (batch, int(model.config.micro_dim)),
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
    else:
        current = torch.as_tensor(
            initial_noise, dtype=parameter.dtype, device=parameter.device
        ).clone()
    if tuple(current.shape) != (batch, int(model.config.micro_dim)):
        raise ValueError("initial micro noise shape does not match diffusion config")
    if not np.isfinite(float(eta)) or float(eta) < 0.0:
        raise ValueError("DDIM eta must be finite and nonnegative")
    if not np.isfinite(float(rho_cfg_scale)) or float(rho_cfg_scale) < 0.0:
        raise ValueError("rho classifier-free guidance scale must be nonnegative")

    schedule = diffusion_schedule(
        model.config, device=parameter.device, dtype=parameter.dtype
    )
    timestep_values = _sample_timestep_values(
        int(model.config.timesteps), sample_steps
    )
    for index, timestep_value in enumerate(timestep_values):
        timestep = torch.full(
            (batch,),
            int(timestep_value),
            dtype=torch.long,
            device=parameter.device,
        )
        with torch.no_grad():
            conditional_output = model(current, rho, timestep, condition)
            if float(rho_cfg_scale) == 1.0:
                output = conditional_output
            else:
                unconditional_output = model(
                    current, torch.zeros_like(rho), timestep, condition
                )
                output = unconditional_output + float(rho_cfg_scale) * (
                    conditional_output - unconditional_output
                )
            predicted_noise, predicted_clean = diffusion_model_predictions(
                current,
                output,
                timestep,
                schedule,
                str(model.config.prediction_type),
            )
        if guidance is not None:
            predicted_noise = guidance(
                current,
                timestep,
                condition,
                rho,
                predicted_noise,
                schedule,
            )
            if tuple(predicted_noise.shape) != tuple(current.shape):
                raise ValueError("micro guidance returned the wrong shape")
            alpha_bar = schedule["alpha_bar"][int(timestep_value)]
            predicted_clean = (
                current - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
        if index + 1 == len(timestep_values):
            current = predicted_clean
            continue
        alpha_bar = schedule["alpha_bar"][int(timestep_value)]
        previous_alpha_bar = schedule["alpha_bar"][
            int(timestep_values[index + 1])
        ]
        sigma = float(eta) * torch.sqrt(
            (1.0 - previous_alpha_bar)
            / (1.0 - alpha_bar)
            * (1.0 - alpha_bar / previous_alpha_bar)
        )
        direction = torch.sqrt(
            (1.0 - previous_alpha_bar - sigma.square()).clamp_min(0.0)
        )
        if float(eta) > 0.0:
            random_noise = torch.randn(
                current.shape,
                dtype=current.dtype,
                device=current.device,
                generator=generator,
            )
        else:
            random_noise = 0.0
        current = (
            torch.sqrt(previous_alpha_bar) * predicted_clean
            + direction * predicted_noise
            + sigma * random_noise
        )
    return current


def save_fullrho_micro_diffusion_checkpoint(
    path: str | Path,
    *,
    model,
    rho_normalization: dict[str, Any],
    condition_normalization: dict[str, Any],
    micro_code_normalization: dict[str, Any],
    direct_macro_checkpoint: str | Path,
    direct_macro_checkpoint_sha256: str,
    codec_checkpoint: str | Path,
    codec_checkpoint_sha256: str,
    local_basis: str | Path,
    local_basis_sha256: str,
    decoder_checkpoint: str | Path,
    decoder_checkpoint_sha256: str,
    train_config: dict[str, Any],
    step: int,
    optimizer_state: dict[str, Any] | None = None,
    ema_state_dict: dict[str, Any] | None = None,
) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "rho_normalization": dict(rho_normalization),
            "condition_normalization": dict(condition_normalization),
            "micro_code_normalization": dict(micro_code_normalization),
            "direct_macro_checkpoint": str(
                Path(direct_macro_checkpoint).resolve()
            ),
            "direct_macro_checkpoint_sha256": str(
                direct_macro_checkpoint_sha256
            ),
            "codec_checkpoint": str(Path(codec_checkpoint).resolve()),
            "codec_checkpoint_sha256": str(codec_checkpoint_sha256),
            "local_basis": str(Path(local_basis).resolve()),
            "local_basis_sha256": str(local_basis_sha256),
            "decoder_checkpoint": str(Path(decoder_checkpoint).resolve()),
            "decoder_checkpoint_sha256": str(decoder_checkpoint_sha256),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_fullrho_micro_diffusion_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected full-rho micro diffusion checkpoint schema")
    model = VPFPFullRhoMicroDenoiser.build(payload["model_config"]).to(
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


def _sample_timestep_values(total: int, requested: int | None) -> list[int]:
    count = int(total) if requested is None else min(int(requested), int(total))
    if count < 1:
        raise ValueError("sample step count must be positive")
    values = np.rint(np.linspace(int(total) - 1, 0, count)).astype(np.int64)
    return [int(value) for value in np.unique(values)[::-1]]


def _resolve_training_timesteps(torch, values, *, batch: int, total: int, device):
    if values is None:
        return torch.randint(0, int(total), (int(batch),), device=device)
    timesteps = torch.as_tensor(values, dtype=torch.long, device=device).reshape(-1)
    if tuple(timesteps.shape) != (int(batch),):
        raise ValueError("training timestep shape does not match micro batch")
    if bool(torch.any((timesteps < 0) | (timesteps >= int(total))).item()):
        raise ValueError("training timestep lies outside diffusion schedule")
    return timesteps


def _resolve_noise(torch, values, reference, parameter):
    if values is None:
        return torch.randn_like(reference)
    noise = torch.as_tensor(
        values, dtype=parameter.dtype, device=parameter.device
    )
    if noise.shape != reference.shape:
        raise ValueError("training noise shape does not match micro target")
    return noise


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
