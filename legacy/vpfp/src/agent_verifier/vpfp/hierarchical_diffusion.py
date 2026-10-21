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
    timestep_embedding,
)
from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
    join_active_latent,
    split_active_latent,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


CONDITION_TRANSFORM_SCHEMA = "vpfp_log_collision_temperature_condition_v1"
CHECKPOINT_SCHEMA = "vpfp_hierarchical_macro_micro_diffusion_checkpoint_v1"


@dataclass(frozen=True)
class VPFPHierarchicalDiffusionConfig:
    latent_config: VPFPStructuredDiffusionConfig
    macro_width: int = 256
    macro_layers: int = 6
    micro_width: int = 384
    micro_layers: int = 8
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        self.latent_config.validate()
        dimensions = (
            self.macro_width,
            self.macro_layers,
            self.micro_width,
            self.micro_layers,
            self.attention_heads,
            self.feedforward_multiplier,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("hierarchical diffusion dimensions must be positive")
        if int(self.macro_width) % int(self.attention_heads):
            raise ValueError("macro width must be divisible by attention heads")
        if int(self.micro_width) % int(self.attention_heads):
            raise ValueError("micro width must be divisible by attention heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_config": self.latent_config.to_dict(),
            "macro_width": int(self.macro_width),
            "macro_layers": int(self.macro_layers),
            "micro_width": int(self.micro_width),
            "micro_layers": int(self.micro_layers),
            "attention_heads": int(self.attention_heads),
            "feedforward_multiplier": int(self.feedforward_multiplier),
            "dropout": float(self.dropout),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPHierarchicalDiffusionConfig":
        payload = dict(data)
        payload["latent_config"] = VPFPStructuredDiffusionConfig.from_dict(
            dict(payload["latent_config"])
        )
        return cls(**payload)


class VPFPHierarchicalDenoiser:
    @staticmethod
    def build(config: VPFPHierarchicalDiffusionConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPHierarchicalDiffusionConfig)
            else VPFPHierarchicalDiffusionConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPHierarchicalDenoiser(_torch(), cfg)


class _AdaptiveTransformerBlock(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(
        self,
        torch,
        *,
        width: int,
        attention_heads: int,
        feedforward_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        self.attention = torch.nn.MultiheadAttention(
            width,
            attention_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        hidden = int(feedforward_multiplier) * int(width)
        self.mlp_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden, width),
            torch.nn.Dropout(float(dropout)),
        )
        self.modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(width, 6 * width),
        )
        torch.nn.init.zeros_(self.modulation[-1].weight)
        torch.nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens, context):
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.modulation(context).chunk(6, dim=-1)
        attention_input = self.attention_norm(tokens)
        attention_input = attention_input * (
            1.0 + attention_scale[:, None]
        ) + attention_shift[:, None]
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + attention_gate[:, None] * attention_output
        mlp_input = self.mlp_norm(tokens)
        mlp_input = mlp_input * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        return tokens + mlp_gate[:, None] * self.mlp(mlp_input)


class _ConditionedTokenDenoiserBase(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def _build_context(self, torch, *, width: int, config):
        self.diffusion_time = torch.nn.Sequential(
            torch.nn.Linear(config.latent_config.time_embedding_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.equation_condition = torch.nn.Sequential(
            torch.nn.Linear(config.latent_config.condition_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )

    def _context(self, timesteps, conditions, *, width: int):
        torch = self.torch
        parameter = next(self.parameters())
        time = torch.as_tensor(timesteps, device=parameter.device).reshape(-1)
        condition = torch.as_tensor(
            conditions, dtype=parameter.dtype, device=parameter.device
        )
        if (
            condition.ndim != 2
            or int(condition.shape[1])
            != int(self.config.latent_config.condition_dim)
            or int(condition.shape[0]) != int(time.shape[0])
        ):
            raise ValueError("hierarchical diffusion condition shape mismatch")
        embedding = timestep_embedding(
            torch,
            time,
            self.config.latent_config.time_embedding_dim,
            dtype=parameter.dtype,
        )
        context = self.diffusion_time(embedding) + self.equation_condition(
            condition
        )
        if int(context.shape[1]) != int(width):
            raise RuntimeError("hierarchical context width changed")
        return time, context

    def _finalize(self, tokens, context):
        shift, scale = self.final_modulation(context).chunk(2, dim=-1)
        return self.final_norm(tokens) * (1.0 + scale[:, None]) + shift[:, None]


class _VPFPMacroDenoiser(_ConditionedTokenDenoiserBase):
    def __init__(self, torch, config: VPFPHierarchicalDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        latent = config.latent_config
        width = int(config.macro_width)
        self.input = torch.nn.Linear(latent.rho_feature_count, width)
        self.position = torch.nn.Parameter(
            torch.randn(latent.rho_time_count, width) / np.sqrt(float(width))
        )
        self._build_context(torch, width=width, config=config)
        self.blocks = torch.nn.ModuleList(
            [
                _AdaptiveTransformerBlock(
                    torch,
                    width=width,
                    attention_heads=int(config.attention_heads),
                    feedforward_multiplier=int(config.feedforward_multiplier),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.macro_layers))
            ]
        )
        self.final_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        self.final_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(width, 2 * width)
        )
        self.output = torch.nn.Linear(width, latent.rho_feature_count)
        torch.nn.init.zeros_(self.final_modulation[-1].weight)
        torch.nn.init.zeros_(self.final_modulation[-1].bias)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, noisy_rho, timesteps, conditions):
        torch = self.torch
        parameter = next(self.parameters())
        rho = torch.as_tensor(
            noisy_rho, dtype=parameter.dtype, device=parameter.device
        )
        latent = self.config.latent_config
        if rho.ndim != 2 or int(rho.shape[1]) != int(latent.rho_latent_dim):
            raise ValueError("macro latent shape does not match config")
        batch = int(rho.shape[0])
        time, context = self._context(
            timesteps, conditions, width=int(self.config.macro_width)
        )
        if int(time.shape[0]) != batch:
            raise ValueError("macro diffusion batch dimensions do not match")
        tokens = self.input(
            rho.reshape(batch, latent.rho_time_count, latent.rho_feature_count)
        ) + self.position[None]
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.output(self._finalize(tokens, context)).reshape(batch, -1)


class _VPFPMicroConditionalDenoiser(_ConditionedTokenDenoiserBase):
    def __init__(self, torch, config: VPFPHierarchicalDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        latent = config.latent_config
        width = int(config.micro_width)
        self.rho_input = torch.nn.Linear(latent.rho_feature_count, width)
        self.local_input = torch.nn.Linear(latent.local_channels, width)
        self.rho_position = torch.nn.Parameter(
            torch.randn(latent.rho_time_count, width) / np.sqrt(float(width))
        )
        self.local_time_position = torch.nn.Parameter(
            torch.randn(latent.local_time_count, width) / np.sqrt(float(width))
        )
        self.local_spatial_position = torch.nn.Parameter(
            torch.randn(latent.local_spatial_feature_count, width)
            / np.sqrt(float(width))
        )
        self.token_type = torch.nn.Parameter(
            torch.randn(2, width) / np.sqrt(float(width))
        )
        self._build_context(torch, width=width, config=config)
        self.blocks = torch.nn.ModuleList(
            [
                _AdaptiveTransformerBlock(
                    torch,
                    width=width,
                    attention_heads=int(config.attention_heads),
                    feedforward_multiplier=int(config.feedforward_multiplier),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.micro_layers))
            ]
        )
        self.final_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        self.final_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(width, 2 * width)
        )
        self.output = torch.nn.Linear(width, latent.local_channels)
        torch.nn.init.zeros_(self.final_modulation[-1].weight)
        torch.nn.init.zeros_(self.final_modulation[-1].bias)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, noisy_local, rho_condition, timesteps, conditions):
        torch = self.torch
        parameter = next(self.parameters())
        local = torch.as_tensor(
            noisy_local, dtype=parameter.dtype, device=parameter.device
        )
        rho = torch.as_tensor(
            rho_condition, dtype=parameter.dtype, device=parameter.device
        )
        latent = self.config.latent_config
        if local.ndim != 2 or int(local.shape[1]) != int(latent.local_latent_dim):
            raise ValueError("micro latent shape does not match config")
        if rho.ndim != 2 or int(rho.shape[1]) != int(latent.rho_latent_dim):
            raise ValueError("micro rho-condition shape does not match config")
        batch = int(local.shape[0])
        if int(rho.shape[0]) != batch:
            raise ValueError("micro and rho batches do not match")
        time, context = self._context(
            timesteps, conditions, width=int(self.config.micro_width)
        )
        if int(time.shape[0]) != batch:
            raise ValueError("micro diffusion batch dimensions do not match")

        rho_tokens = self.rho_input(
            rho.reshape(batch, latent.rho_time_count, latent.rho_feature_count)
        ) + (self.rho_position[None] + self.token_type[None, :1])
        local_tensor = local.reshape(
            batch,
            latent.local_channels,
            latent.local_time_count,
            latent.local_spatial_feature_count,
        )
        local_tokens = self.local_input(
            local_tensor.permute(0, 2, 3, 1).reshape(
                batch,
                latent.local_time_count * latent.local_spatial_feature_count,
                latent.local_channels,
            )
        )
        local_position = (
            self.local_time_position[:, None]
            + self.local_spatial_position[None]
        ).reshape(1, -1, int(self.config.micro_width))
        local_tokens = local_tokens + local_position + self.token_type[None, 1:2]
        tokens = torch.cat((rho_tokens, local_tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens, context)
        tokens = self._finalize(tokens, context)
        local_tokens = tokens[:, latent.rho_time_count :]
        prediction = self.output(local_tokens).reshape(
            batch,
            latent.local_time_count,
            latent.local_spatial_feature_count,
            latent.local_channels,
        ).permute(0, 3, 1, 2)
        return prediction.reshape(batch, -1)


class _VPFPHierarchicalDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config: VPFPHierarchicalDiffusionConfig):
        super().__init__()
        self.config = config
        self.macro_model = _VPFPMacroDenoiser(torch, config)
        self.micro_model = _VPFPMicroConditionalDenoiser(torch, config)
        dtype = (
            torch.float64
            if config.latent_config.dtype == "float64"
            else torch.float32
        )
        self.to(dtype=dtype)


def transform_equation_conditions(
    values,
    *,
    schema: str = CONDITION_TRANSFORM_SCHEMA,
):
    """Map public `(collision_nu, temperature)` to stable model conditions."""

    if schema != CONDITION_TRANSFORM_SCHEMA:
        raise ValueError(f"unsupported VPFP condition transform: {schema}")
    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or int(tensor.shape[1]) != 2:
        raise ValueError("VPFP equation conditions must have shape [batch,2]")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        raise ValueError("VPFP equation conditions must be finite")
    if bool(torch.any(tensor[:, 0] <= 0.0).item()):
        raise ValueError("collision frequency must be positive before log transform")
    return torch.stack((torch.log(tensor[:, 0]), tensor[:, 1]), dim=-1)


def hierarchical_diffusion_loss(
    model,
    clean_active_latent,
    conditions,
    *,
    macro_timesteps=None,
    micro_timesteps=None,
    macro_noise=None,
    micro_noise=None,
    macro_weight: float = 1.0,
    micro_weight: float = 1.0,
    rho_condition_augmentation_std: float = 0.0,
):
    torch = _torch()
    parameter = next(model.parameters())
    config = model.config.latent_config
    clean = torch.as_tensor(
        clean_active_latent, dtype=parameter.dtype, device=parameter.device
    )
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    clean_rho, clean_local = split_active_latent(clean, config)
    clean_rho = clean_rho.reshape(int(clean.shape[0]), -1)
    clean_local = clean_local.reshape(int(clean.shape[0]), -1)
    macro_t = _resolve_training_timesteps(
        torch,
        macro_timesteps,
        batch=int(clean.shape[0]),
        total=int(config.timesteps),
        device=parameter.device,
    )
    micro_t = _resolve_training_timesteps(
        torch,
        micro_timesteps,
        batch=int(clean.shape[0]),
        total=int(config.timesteps),
        device=parameter.device,
    )
    macro_epsilon = _resolve_noise(
        torch, macro_noise, clean_rho, dtype=parameter.dtype, device=parameter.device
    )
    micro_epsilon = _resolve_noise(
        torch,
        micro_noise,
        clean_local,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    schedule = diffusion_schedule(
        config, device=parameter.device, dtype=parameter.dtype
    )
    noisy_rho = q_sample(clean_rho, macro_t, macro_epsilon, schedule)
    noisy_local = q_sample(clean_local, micro_t, micro_epsilon, schedule)
    rho_condition = clean_rho
    augmentation_std = float(rho_condition_augmentation_std)
    if not np.isfinite(augmentation_std) or augmentation_std < 0.0:
        raise ValueError("rho condition augmentation must be finite and nonnegative")
    if augmentation_std:
        rho_condition = rho_condition + augmentation_std * torch.randn_like(
            rho_condition
        )
    macro_prediction = model.macro_model(noisy_rho, macro_t, condition)
    micro_prediction = model.micro_model(
        noisy_local, rho_condition, micro_t, condition
    )
    prediction_type = str(config.prediction_type)
    macro_target = diffusion_training_target(
        clean_rho, macro_epsilon, macro_t, schedule, prediction_type
    )
    micro_target = diffusion_training_target(
        clean_local, micro_epsilon, micro_t, schedule, prediction_type
    )
    macro_mse = torch.mean((macro_prediction - macro_target).square())
    micro_mse = torch.mean((micro_prediction - micro_target).square())
    weights = (float(macro_weight), float(micro_weight))
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("macro and micro weights must be finite and nonnegative")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one hierarchical loss weight must be positive")
    loss = (weights[0] * macro_mse + weights[1] * micro_mse) / weight_sum
    return loss, {
        "loss": loss,
        "macro_noise_mse": macro_mse,
        "micro_noise_mse": micro_mse,
        "macro_timestep_mean": macro_t.to(dtype=parameter.dtype).mean(),
        "micro_timestep_mean": micro_t.to(dtype=parameter.dtype).mean(),
    }


def sample_hierarchical_latents(
    model,
    conditions,
    *,
    macro_initial_noise=None,
    micro_initial_noise=None,
    rho_condition=None,
    sample_steps: int | None = None,
    macro_sample_steps: int | None = None,
    micro_sample_steps: int | None = None,
    eta: float = 0.0,
    macro_guidance: Callable | None = None,
    micro_guidance: Callable | None = None,
):
    """Sample `rho` first, then sample micro coefficients conditioned on it."""

    config = model.config.latent_config
    if rho_condition is None:
        rho = _sample_block(
            model.macro_model,
            conditions,
            latent_dim=int(config.rho_latent_dim),
            diffusion_config=config,
            initial_noise=macro_initial_noise,
            sample_steps=(
                macro_sample_steps
                if macro_sample_steps is not None
                else sample_steps
            ),
            eta=float(eta),
            guidance=macro_guidance,
        )
    else:
        torch = _torch()
        parameter = next(model.micro_model.parameters())
        rho = torch.as_tensor(
            rho_condition, dtype=parameter.dtype, device=parameter.device
        )
        expected_shape = (
            int(torch.as_tensor(conditions).shape[0]),
            int(config.rho_latent_dim),
        )
        if tuple(rho.shape) != expected_shape:
            raise ValueError(
                "rho_condition shape must match condition batch and rho latent dim"
            )
        if not torch.all(torch.isfinite(rho)):
            raise ValueError("rho_condition must be finite")

    def micro_forward(current, timestep, condition):
        return model.micro_model(current, rho, timestep, condition)

    local = _sample_block(
        micro_forward,
        conditions,
        latent_dim=int(config.local_latent_dim),
        diffusion_config=config,
        initial_noise=micro_initial_noise,
        sample_steps=(
            micro_sample_steps if micro_sample_steps is not None else sample_steps
        ),
        eta=float(eta),
        guidance=micro_guidance,
        parameter=next(model.micro_model.parameters()),
    )
    return join_active_latent(
        rho.reshape(-1, config.rho_time_count, config.rho_feature_count),
        local.reshape(
            -1,
            config.local_channels,
            config.local_time_count,
            config.local_spatial_feature_count,
        ),
        config,
    )


def save_hierarchical_diffusion_checkpoint(
    path: str | Path,
    *,
    model,
    active_latent_normalization: dict[str, Any],
    condition_normalization: dict[str, Any],
    decoder_checkpoint: str | Path,
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
            "active_latent_normalization": dict(active_latent_normalization),
            "condition_normalization": dict(condition_normalization),
            "condition_transform_schema": CONDITION_TRANSFORM_SCHEMA,
            "decoder_checkpoint": str(Path(decoder_checkpoint).resolve()),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_hierarchical_diffusion_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected hierarchical VPFP diffusion checkpoint schema")
    if payload.get("condition_transform_schema") != CONDITION_TRANSFORM_SCHEMA:
        raise ValueError("hierarchical VPFP condition transform changed")
    model = VPFPHierarchicalDenoiser.build(
        VPFPHierarchicalDiffusionConfig.from_dict(payload["model_config"])
    ).to(device=device)
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _sample_block(
    model,
    conditions,
    *,
    latent_dim: int,
    diffusion_config,
    initial_noise,
    sample_steps: int | None,
    eta: float,
    guidance,
    parameter=None,
):
    torch = _torch()
    if parameter is None:
        parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(condition.shape[0])
    if initial_noise is None:
        current = torch.randn(
            (batch, int(latent_dim)),
            dtype=parameter.dtype,
            device=parameter.device,
        )
    else:
        current = torch.as_tensor(
            initial_noise, dtype=parameter.dtype, device=parameter.device
        ).clone()
    if tuple(current.shape) != (batch, int(latent_dim)):
        raise ValueError("hierarchical initial noise shape mismatch")
    if not np.isfinite(float(eta)) or float(eta) < 0.0:
        raise ValueError("DDIM eta must be finite and nonnegative")
    schedule = diffusion_schedule(
        diffusion_config, device=parameter.device, dtype=parameter.dtype
    )
    timestep_values = _sample_timestep_values(
        int(diffusion_config.timesteps), sample_steps
    )
    for index, timestep_value in enumerate(timestep_values):
        timestep = torch.full(
            (batch,), int(timestep_value), dtype=torch.long, device=parameter.device
        )
        with torch.no_grad():
            model_output = model(current, timestep, condition)
            predicted_noise, clean_prediction = diffusion_model_predictions(
                current,
                model_output,
                timestep,
                schedule,
                str(diffusion_config.prediction_type),
            )
        if guidance is not None:
            predicted_noise = guidance(
                current, timestep, condition, predicted_noise, schedule
            )
            alpha_bar = schedule["alpha_bar"][int(timestep_value)]
            clean_prediction = (
                current - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
        if index + 1 == len(timestep_values):
            current = clean_prediction
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
        direction_scale = torch.sqrt(
            (1.0 - previous_alpha_bar - sigma.square()).clamp_min(0.0)
        )
        random_noise = torch.randn_like(current) if float(eta) > 0.0 else 0.0
        current = (
            torch.sqrt(previous_alpha_bar) * clean_prediction
            + direction_scale * predicted_noise
            + sigma * random_noise
        )
    return current


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
        raise ValueError("hierarchical training timestep shape mismatch")
    if bool(torch.any((timesteps < 0) | (timesteps >= int(total))).item()):
        raise ValueError("hierarchical training timestep is outside the schedule")
    return timesteps


def _resolve_noise(torch, values, reference, *, dtype, device):
    if values is None:
        return torch.randn_like(reference)
    noise = torch.as_tensor(values, dtype=dtype, device=device)
    if noise.shape != reference.shape:
        raise ValueError("hierarchical training noise shape mismatch")
    return noise


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError(
            "PyTorch is required for VPFP diffusion"
        ) from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
