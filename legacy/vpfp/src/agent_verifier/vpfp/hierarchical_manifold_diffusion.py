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


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


MANIFOLD_DIFFUSION_CHECKPOINT_SCHEMA = (
    "vpfp_hierarchical_manifold_diffusion_checkpoint_v1"
)


@dataclass(frozen=True)
class VPFPHierarchicalManifoldDiffusionConfig:
    macro_dim: int
    micro_dim: int
    condition_dim: int = 2
    macro_width: int = 768
    macro_layers: int = 8
    micro_width: int = 1024
    micro_layers: int = 10
    feedforward_multiplier: int = 2
    time_embedding_dim: int = 128
    timesteps: int = 1000
    noise_schedule: str = "cosine"
    prediction_type: str = "v"
    dropout: float = 0.0
    dtype: str = "float32"

    def validate(self) -> None:
        dimensions = (
            self.macro_dim,
            self.micro_dim,
            self.condition_dim,
            self.macro_width,
            self.macro_layers,
            self.micro_width,
            self.micro_layers,
            self.feedforward_multiplier,
            self.time_embedding_dim,
            self.timesteps,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("manifold diffusion dimensions must be positive")
        if self.noise_schedule not in {"linear", "cosine"}:
            raise ValueError("unsupported manifold diffusion noise schedule")
        if self.prediction_type not in {"epsilon", "v"}:
            raise ValueError("unsupported manifold diffusion prediction type")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("manifold diffusion dropout must lie in [0,1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("manifold diffusion dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "VPFPHierarchicalManifoldDiffusionConfig":
        config = cls(**dict(data))
        config.validate()
        return config


class VPFPHierarchicalManifoldDenoiser:
    @staticmethod
    def build(
        config: VPFPHierarchicalManifoldDiffusionConfig | dict[str, Any],
    ):
        cfg = (
            config
            if isinstance(config, VPFPHierarchicalManifoldDiffusionConfig)
            else VPFPHierarchicalManifoldDiffusionConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPHierarchicalManifoldDenoiser(_torch(), cfg)


class _AdaptiveResidualBlock(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(
        self,
        torch,
        *,
        width: int,
        feedforward_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        hidden = int(width) * int(feedforward_multiplier)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden, width),
            torch.nn.Dropout(float(dropout)),
        )
        self.modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(width, 3 * width)
        )
        torch.nn.init.zeros_(self.modulation[-1].weight)
        torch.nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values, context):
        shift, scale, gate = self.modulation(context).chunk(3, dim=-1)
        hidden = self.norm(values) * (1.0 + scale) + shift
        return values + gate * self.mlp(hidden)


class _ConditionedVectorDenoiserBase(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def _build_context(self, torch, *, width: int, config):
        self.diffusion_time = torch.nn.Sequential(
            torch.nn.Linear(config.time_embedding_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.equation_condition = torch.nn.Sequential(
            torch.nn.Linear(config.condition_dim, width),
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
            or int(condition.shape[0]) != int(time.shape[0])
            or int(condition.shape[1]) != int(self.config.condition_dim)
        ):
            raise ValueError("manifold diffusion condition shape mismatch")
        embedding = timestep_embedding(
            torch,
            time,
            int(self.config.time_embedding_dim),
            dtype=parameter.dtype,
        )
        context = self.diffusion_time(embedding) + self.equation_condition(
            condition
        )
        if int(context.shape[1]) != int(width):
            raise RuntimeError("manifold diffusion context width changed")
        return context

    def _finalize(self, values, context):
        shift, scale = self.final_modulation(context).chunk(2, dim=-1)
        return self.final_norm(values) * (1.0 + scale) + shift


class _MacroCodeDenoiser(_ConditionedVectorDenoiserBase):
    def __init__(self, torch, config):
        super().__init__()
        self.torch = torch
        self.config = config
        width = int(config.macro_width)
        self.input = torch.nn.Linear(int(config.macro_dim), width)
        self._build_context(torch, width=width, config=config)
        self.blocks = torch.nn.ModuleList(
            [
                _AdaptiveResidualBlock(
                    torch,
                    width=width,
                    feedforward_multiplier=int(
                        config.feedforward_multiplier
                    ),
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
        self.output = torch.nn.Linear(width, int(config.macro_dim))
        _zero_output(torch, self.final_modulation[-1], self.output)

    def forward(self, noisy_macro, timesteps, conditions):
        parameter = next(self.parameters())
        values = self.torch.as_tensor(
            noisy_macro, dtype=parameter.dtype, device=parameter.device
        )
        if values.ndim != 2 or int(values.shape[1]) != int(
            self.config.macro_dim
        ):
            raise ValueError("macro code shape does not match config")
        context = self._context(
            timesteps, conditions, width=int(self.config.macro_width)
        )
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(self._finalize(hidden, context))


class _MicroCodeDenoiser(_ConditionedVectorDenoiserBase):
    def __init__(self, torch, config):
        super().__init__()
        self.torch = torch
        self.config = config
        width = int(config.micro_width)
        self.input = torch.nn.Linear(int(config.micro_dim), width)
        self.macro_condition = torch.nn.Sequential(
            torch.nn.Linear(int(config.macro_dim), width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self._build_context(torch, width=width, config=config)
        self.blocks = torch.nn.ModuleList(
            [
                _AdaptiveResidualBlock(
                    torch,
                    width=width,
                    feedforward_multiplier=int(
                        config.feedforward_multiplier
                    ),
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
        self.output = torch.nn.Linear(width, int(config.micro_dim))
        _zero_output(torch, self.final_modulation[-1], self.output)

    def forward(self, noisy_micro, macro_condition, timesteps, conditions):
        parameter = next(self.parameters())
        micro = self.torch.as_tensor(
            noisy_micro, dtype=parameter.dtype, device=parameter.device
        )
        macro = self.torch.as_tensor(
            macro_condition, dtype=parameter.dtype, device=parameter.device
        )
        expected = (
            micro.ndim == 2
            and macro.ndim == 2
            and int(micro.shape[1]) == int(self.config.micro_dim)
            and int(macro.shape[1]) == int(self.config.macro_dim)
            and int(micro.shape[0]) == int(macro.shape[0])
        )
        if not expected:
            raise ValueError("conditional micro code shapes do not match config")
        context = self._context(
            timesteps, conditions, width=int(self.config.micro_width)
        )
        context = context + self.macro_condition(macro)
        hidden = self.input(micro)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(self._finalize(hidden, context))


class _VPFPHierarchicalManifoldDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config):
        super().__init__()
        self.config = config
        self.macro_model = _MacroCodeDenoiser(torch, config)
        self.micro_model = _MicroCodeDenoiser(torch, config)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)


def hierarchical_manifold_diffusion_loss(
    model,
    clean_macro,
    clean_micro,
    conditions,
    *,
    macro_timesteps=None,
    micro_timesteps=None,
    macro_noise=None,
    micro_noise=None,
    macro_weight: float = 1.0,
    micro_weight: float = 1.0,
    macro_condition_augmentation_std: float = 0.0,
):
    torch = _torch()
    parameter = next(model.parameters())
    macro = torch.as_tensor(
        clean_macro, dtype=parameter.dtype, device=parameter.device
    )
    micro = torch.as_tensor(
        clean_micro, dtype=parameter.dtype, device=parameter.device
    )
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    if (
        macro.ndim != 2
        or micro.ndim != 2
        or int(macro.shape[0]) != int(micro.shape[0])
        or int(macro.shape[1]) != int(model.config.macro_dim)
        or int(micro.shape[1]) != int(model.config.micro_dim)
    ):
        raise ValueError("clean manifold code shapes do not match model")
    batch = int(macro.shape[0])
    macro_t = _resolve_training_timesteps(
        torch, macro_timesteps, batch=batch, total=model.config.timesteps,
        device=parameter.device,
    )
    micro_t = _resolve_training_timesteps(
        torch, micro_timesteps, batch=batch, total=model.config.timesteps,
        device=parameter.device,
    )
    macro_noise = _resolve_noise(
        torch, macro_noise, macro, dtype=parameter.dtype, device=parameter.device
    )
    micro_noise = _resolve_noise(
        torch, micro_noise, micro, dtype=parameter.dtype, device=parameter.device
    )
    schedule = diffusion_schedule(
        model.config, device=parameter.device, dtype=parameter.dtype
    )
    noisy_macro = q_sample(macro, macro_t, macro_noise, schedule)
    noisy_micro = q_sample(micro, micro_t, micro_noise, schedule)
    macro_condition = macro
    augmentation = float(macro_condition_augmentation_std)
    if augmentation < 0.0 or not np.isfinite(augmentation):
        raise ValueError("macro condition augmentation must be nonnegative")
    if augmentation:
        macro_condition = macro_condition + augmentation * torch.randn_like(
            macro_condition
        )
    macro_prediction = model.macro_model(noisy_macro, macro_t, condition)
    micro_prediction = model.micro_model(
        noisy_micro, macro_condition, micro_t, condition
    )
    prediction_type = str(model.config.prediction_type)
    macro_target = diffusion_training_target(
        macro, macro_noise, macro_t, schedule, prediction_type
    )
    micro_target = diffusion_training_target(
        micro, micro_noise, micro_t, schedule, prediction_type
    )
    macro_mse = torch.mean((macro_prediction - macro_target).square())
    micro_mse = torch.mean((micro_prediction - micro_target).square())
    weights = (float(macro_weight), float(micro_weight))
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("manifold diffusion loss weights must be nonnegative")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("at least one manifold loss weight must be positive")
    loss = (weights[0] * macro_mse + weights[1] * micro_mse) / total_weight
    return loss, {
        "loss": loss,
        "macro_noise_mse": macro_mse,
        "micro_noise_mse": micro_mse,
        "macro_timestep_mean": macro_t.to(dtype=parameter.dtype).mean(),
        "micro_timestep_mean": micro_t.to(dtype=parameter.dtype).mean(),
    }


def sample_hierarchical_manifold_codes(
    model,
    conditions,
    *,
    macro_initial_noise=None,
    micro_initial_noise=None,
    macro_condition=None,
    sample_steps: int | None = None,
    macro_sample_steps: int | None = None,
    micro_sample_steps: int | None = None,
    eta: float = 0.0,
    macro_guidance: Callable | None = None,
    micro_guidance: Callable | None = None,
):
    config = model.config
    if macro_condition is None:
        macro = _sample_block(
            model.macro_model,
            conditions,
            latent_dim=int(config.macro_dim),
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
        macro = torch.as_tensor(
            macro_condition, dtype=parameter.dtype, device=parameter.device
        )
        expected = (
            int(torch.as_tensor(conditions).shape[0]),
            int(config.macro_dim),
        )
        if tuple(macro.shape) != expected or not bool(
            torch.all(torch.isfinite(macro)).item()
        ):
            raise ValueError("fixed macro condition is invalid")

    def micro_forward(current, timestep, condition):
        return model.micro_model(current, macro, timestep, condition)

    micro = _sample_block(
        micro_forward,
        conditions,
        latent_dim=int(config.micro_dim),
        diffusion_config=config,
        initial_noise=micro_initial_noise,
        sample_steps=(
            micro_sample_steps
            if micro_sample_steps is not None
            else sample_steps
        ),
        eta=float(eta),
        guidance=micro_guidance,
        parameter=next(model.micro_model.parameters()),
    )
    return macro, micro


def save_hierarchical_manifold_diffusion_checkpoint(
    path: str | Path,
    *,
    model,
    condition_normalization: dict[str, Any],
    manifold_basis: str | Path,
    manifold_basis_sha256: str,
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
            "schema": MANIFOLD_DIFFUSION_CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "condition_normalization": dict(condition_normalization),
            "manifold_basis": str(Path(manifold_basis).resolve()),
            "manifold_basis_sha256": str(manifold_basis_sha256),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_hierarchical_manifold_diffusion_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != MANIFOLD_DIFFUSION_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected manifold diffusion checkpoint schema")
    model = VPFPHierarchicalManifoldDenoiser.build(
        payload["model_config"]
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
        raise ValueError("manifold diffusion initial noise shape changed")
    schedule = diffusion_schedule(
        diffusion_config,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    timesteps = _sample_timestep_values(
        int(diffusion_config.timesteps), sample_steps
    )
    for index, timestep_value in enumerate(timesteps):
        timestep = torch.full(
            (batch,),
            int(timestep_value),
            dtype=torch.long,
            device=parameter.device,
        )
        with torch.no_grad():
            output = model(current, timestep, condition)
            predicted_noise, predicted_clean = diffusion_model_predictions(
                current,
                output,
                timestep,
                schedule,
                str(diffusion_config.prediction_type),
            )
        if guidance is not None:
            predicted_noise = guidance(
                current, timestep, condition, predicted_noise, schedule
            )
            alpha_bar = schedule["alpha_bar"][int(timestep_value)]
            predicted_clean = (
                current - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
        alpha_bar = schedule["alpha_bar"][int(timestep_value)]
        if index + 1 == len(timesteps):
            current = predicted_clean
            continue
        previous = int(timesteps[index + 1])
        previous_alpha_bar = schedule["alpha_bar"][previous]
        sigma = float(eta) * torch.sqrt(
            (1.0 - previous_alpha_bar)
            / (1.0 - alpha_bar)
            * (1.0 - alpha_bar / previous_alpha_bar)
        )
        direction = torch.sqrt(
            (1.0 - previous_alpha_bar - sigma.square()).clamp_min(0.0)
        )
        random_noise = torch.randn_like(current) if float(eta) > 0.0 else 0.0
        current = (
            torch.sqrt(previous_alpha_bar) * predicted_clean
            + direction * predicted_noise
            + sigma * random_noise
        )
    return current


def _sample_timestep_values(
    total: int, requested: int | None
) -> list[int]:
    count = int(total) if requested is None else min(int(total), int(requested))
    if count < 1:
        raise ValueError("sample steps must be positive")
    values = np.linspace(int(total) - 1, 0, count, dtype=int)
    return list(dict.fromkeys(int(value) for value in values))


def _resolve_training_timesteps(
    torch, values, *, batch: int, total: int, device
):
    if values is None:
        return torch.randint(0, int(total), (int(batch),), device=device)
    timesteps = torch.as_tensor(
        values, dtype=torch.long, device=device
    ).reshape(-1)
    if tuple(timesteps.shape) != (int(batch),):
        raise ValueError("manifold training timestep shape mismatch")
    if bool(torch.any((timesteps < 0) | (timesteps >= int(total))).item()):
        raise ValueError("manifold training timestep is outside the schedule")
    return timesteps


def _resolve_noise(torch, values, reference, *, dtype, device):
    if values is None:
        return torch.randn_like(reference)
    noise = torch.as_tensor(values, dtype=dtype, device=device)
    if noise.shape != reference.shape:
        raise ValueError("manifold training noise shape mismatch")
    return noise



def _zero_output(torch, modulation, output) -> None:
    torch.nn.init.zeros_(modulation.weight)
    torch.nn.init.zeros_(modulation.bias)
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
