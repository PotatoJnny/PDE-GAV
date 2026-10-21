from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPContinuousDiffusionConfig:
    latent_dim: int = 12
    condition_dim: int = 4
    hidden_width: int = 192
    hidden_layers: int = 5
    time_embedding_dim: int = 64
    timesteps: int = 200
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    noise_schedule: str = "linear"
    prediction_type: str = "epsilon"
    dtype: str = "float32"
    representation: str = "vpfp_continuous_vector_latent_diffusion_v1"

    def validate(self) -> None:
        if min(
            self.latent_dim,
            self.condition_dim,
            self.hidden_width,
            self.hidden_layers,
            self.time_embedding_dim,
            self.timesteps,
        ) < 1:
            raise ValueError("continuous diffusion dimensions must be positive")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("continuous diffusion beta schedule is invalid")
        if self.noise_schedule not in {"linear", "cosine"}:
            raise ValueError("noise_schedule must be linear or cosine")
        if self.prediction_type not in {"epsilon", "v"}:
            raise ValueError("prediction_type must be epsilon or v")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("continuous diffusion dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPContinuousDiffusionConfig":
        return cls(**dict(data))


class VPFPContinuousDenoiser:
    @staticmethod
    def build(config: VPFPContinuousDiffusionConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPContinuousDiffusionConfig)
            else VPFPContinuousDiffusionConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPContinuousDenoiser(_torch(), cfg)


class _ResidualBlock((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch, width: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(width, 2 * width),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * width, width),
        )

    def forward(self, values):
        return values + self.network(self.norm(values))


class _VPFPContinuousDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config: VPFPContinuousDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        input_dim = (
            int(config.latent_dim)
            + int(config.condition_dim)
            + int(config.time_embedding_dim)
        )
        self.input = torch.nn.Linear(input_dim, config.hidden_width)
        self.blocks = torch.nn.ModuleList(
            [
                _ResidualBlock(torch, config.hidden_width)
                for _ in range(int(config.hidden_layers))
            ]
        )
        self.output_norm = torch.nn.LayerNorm(config.hidden_width)
        self.output = torch.nn.Linear(config.hidden_width, config.latent_dim)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy_latent, timesteps, conditions):
        torch = self.torch
        parameter = next(self.parameters())
        latent = torch.as_tensor(
            noisy_latent, dtype=parameter.dtype, device=parameter.device
        )
        condition = torch.as_tensor(
            conditions, dtype=parameter.dtype, device=parameter.device
        )
        time = torch.as_tensor(timesteps, device=parameter.device).reshape(-1)
        if latent.ndim != 2 or int(latent.shape[1]) != self.config.latent_dim:
            raise ValueError("noisy latent shape does not match diffusion config")
        if condition.ndim != 2 or int(condition.shape[1]) != self.config.condition_dim:
            raise ValueError("condition shape does not match diffusion config")
        if int(time.shape[0]) != int(latent.shape[0]) or int(condition.shape[0]) != int(
            latent.shape[0]
        ):
            raise ValueError("diffusion batch dimensions do not match")
        embedding = timestep_embedding(
            torch,
            time,
            self.config.time_embedding_dim,
            dtype=parameter.dtype,
        )
        hidden = torch.nn.functional.silu(
            self.input(torch.cat((latent, condition, embedding), dim=-1))
        )
        block_scale = float(max(1, len(self.blocks))) ** -0.5
        for block in self.blocks:
            hidden = hidden + block_scale * (block(hidden) - hidden)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))


def timestep_embedding(torch, timesteps, dimension: int, *, dtype):
    half = int(dimension) // 2
    if half == 0:
        return timesteps.to(dtype=dtype)[:, None]
    frequencies = torch.exp(
        -np.log(10000.0)
        * torch.arange(half, dtype=dtype, device=timesteps.device)
        / float(max(half - 1, 1))
    )
    angles = timesteps.to(dtype=dtype)[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if int(embedding.shape[1]) < int(dimension):
        embedding = torch.nn.functional.pad(
            embedding, (0, int(dimension) - int(embedding.shape[1]))
        )
    return embedding


def diffusion_schedule(config: VPFPContinuousDiffusionConfig, *, device, dtype):
    torch = _torch()
    schedule_name = str(getattr(config, "noise_schedule", "linear"))
    if schedule_name == "linear":
        beta = torch.linspace(
            float(config.beta_start),
            float(config.beta_end),
            int(config.timesteps),
            dtype=dtype,
            device=device,
        )
    elif schedule_name == "cosine":
        step = torch.arange(
            int(config.timesteps) + 1,
            dtype=dtype,
            device=device,
        )
        offset = 0.008
        angle = (
            (step / float(config.timesteps) + offset)
            / (1.0 + offset)
            * (np.pi / 2.0)
        )
        cumulative = torch.cos(angle).square()
        cumulative = cumulative / cumulative[0]
        beta = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(
            min=1.0e-8, max=0.999
        )
    else:
        raise ValueError(f"unsupported noise schedule: {schedule_name}")
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar}


def q_sample(clean_latent, timesteps, noise, schedule):
    alpha_bar = schedule["alpha_bar"].index_select(0, timesteps)
    return (
        torch_sqrt(alpha_bar)[:, None] * clean_latent
        + torch_sqrt(1.0 - alpha_bar)[:, None] * noise
    )


def diffusion_training_target(
    clean_latent,
    noise,
    timesteps,
    schedule,
    prediction_type: str,
):
    alpha_bar = schedule["alpha_bar"].index_select(0, timesteps)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v":
        return (
            torch_sqrt(alpha_bar)[:, None] * noise
            - torch_sqrt(1.0 - alpha_bar)[:, None] * clean_latent
        )
    raise ValueError(f"unsupported prediction type: {prediction_type}")


def diffusion_model_predictions(
    noisy_latent,
    model_output,
    timesteps,
    schedule,
    prediction_type: str,
):
    alpha_bar = schedule["alpha_bar"].index_select(0, timesteps)
    signal = torch_sqrt(alpha_bar)[:, None]
    noise_scale = torch_sqrt(1.0 - alpha_bar)[:, None]
    if prediction_type == "epsilon":
        predicted_noise = model_output
        predicted_clean = (
            noisy_latent - noise_scale * predicted_noise
        ) / signal
    elif prediction_type == "v":
        predicted_clean = signal * noisy_latent - noise_scale * model_output
        predicted_noise = noise_scale * noisy_latent + signal * model_output
    else:
        raise ValueError(f"unsupported prediction type: {prediction_type}")
    return predicted_noise, predicted_clean


def continuous_diffusion_loss(model, clean_latent, conditions):
    torch = _torch()
    parameter = next(model.parameters())
    clean = torch.as_tensor(
        clean_latent, dtype=parameter.dtype, device=parameter.device
    )
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    timesteps = torch.randint(
        0,
        int(model.config.timesteps),
        (int(clean.shape[0]),),
        device=parameter.device,
    )
    noise = torch.randn_like(clean)
    schedule = diffusion_schedule(
        model.config, device=parameter.device, dtype=parameter.dtype
    )
    noisy = q_sample(clean, timesteps, noise, schedule)
    prediction = model(noisy, timesteps, condition)
    prediction_type = str(getattr(model.config, "prediction_type", "epsilon"))
    target = diffusion_training_target(
        clean,
        noise,
        timesteps,
        schedule,
        prediction_type,
    )
    loss = torch.mean((prediction - target).square())
    return loss, {
        "loss": loss,
        "noise_mse": loss,
        "timestep_mean": timesteps.to(dtype=parameter.dtype).mean(),
    }


def sample_continuous_latents(
    model,
    conditions,
    *,
    initial_noise=None,
    sample_steps: int | None = None,
    start_timestep: int | None = None,
    eta: float = 0.0,
    guidance: Callable | None = None,
):
    """DDIM sample normalized vector latents, optionally changing predicted noise."""

    torch = _torch()
    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(condition.shape[0])
    if initial_noise is None:
        current = torch.randn(
            (batch, int(model.config.latent_dim)),
            dtype=parameter.dtype,
            device=parameter.device,
        )
    else:
        current = torch.as_tensor(
            initial_noise, dtype=parameter.dtype, device=parameter.device
        ).clone()
    schedule = diffusion_schedule(
        model.config, device=parameter.device, dtype=parameter.dtype
    )
    maximum_timestep = (
        int(model.config.timesteps) - 1
        if start_timestep is None
        else int(start_timestep)
    )
    if not 0 <= maximum_timestep < int(model.config.timesteps):
        raise ValueError("start timestep is outside the diffusion schedule")
    timestep_values = _sample_timestep_values(
        maximum_timestep + 1, sample_steps
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
                str(getattr(model.config, "prediction_type", "epsilon")),
            )
        if guidance is not None:
            predicted_noise = guidance(
                current, timestep, condition, predicted_noise, schedule
            )
            alpha_bar = schedule["alpha_bar"][int(timestep_value)]
            clean_prediction = (
                current - torch_sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch_sqrt(alpha_bar)
        alpha_bar = schedule["alpha_bar"][int(timestep_value)]
        if index + 1 == len(timestep_values):
            current = clean_prediction
            continue
        previous_timestep = int(timestep_values[index + 1])
        previous_alpha_bar = schedule["alpha_bar"][previous_timestep]
        sigma = float(eta) * torch_sqrt(
            (1.0 - previous_alpha_bar)
            / (1.0 - alpha_bar)
            * (1.0 - alpha_bar / previous_alpha_bar)
        )
        direction_scale = torch_sqrt(
            (1.0 - previous_alpha_bar - sigma.square()).clamp_min(0.0)
        )
        random_noise = torch.randn_like(current) if float(eta) > 0.0 else 0.0
        current = (
            torch_sqrt(previous_alpha_bar) * clean_prediction
            + direction_scale * predicted_noise
            + sigma * random_noise
        )
    return current


def normalize_vectors(values, normalization: dict[str, Any]):
    torch = _torch()
    tensor = torch.as_tensor(values)
    mean = torch.as_tensor(
        normalization["mean"], dtype=tensor.dtype, device=tensor.device
    )
    std = torch.as_tensor(
        normalization["std"], dtype=tensor.dtype, device=tensor.device
    )
    return (tensor - mean) / std


def denormalize_vectors(values, normalization: dict[str, Any]):
    torch = _torch()
    tensor = torch.as_tensor(values)
    mean = torch.as_tensor(
        normalization["mean"], dtype=tensor.dtype, device=tensor.device
    )
    std = torch.as_tensor(
        normalization["std"], dtype=tensor.dtype, device=tensor.device
    )
    return tensor * std + mean


def fit_vector_normalization(values, *, min_std: float = 1.0e-4):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("vector normalization expects a rank-two array")
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.maximum(np.std(array, axis=0), float(min_std)).tolist(),
        "min_std": float(min_std),
    }


def save_continuous_diffusion_checkpoint(
    path: str | Path,
    *,
    model,
    latent_normalization: dict[str, Any],
    condition_normalization: dict[str, Any],
    train_config: dict[str, Any],
    step: int,
    optimizer=None,
) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_continuous_diffusion_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "latent_normalization": latent_normalization,
            "condition_normalization": condition_normalization,
            "train_config": train_config,
            "step": int(step),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
        },
        destination,
    )


def load_continuous_diffusion_checkpoint(path: str | Path, *, device="cpu"):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_continuous_diffusion_checkpoint_v1":
        raise ValueError("unexpected continuous diffusion checkpoint schema")
    model = VPFPContinuousDenoiser.build(payload["model_config"]).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _sample_timestep_values(total: int, requested: int | None) -> list[int]:
    count = int(total) if requested is None else min(int(total), int(requested))
    if count < 1:
        raise ValueError("sample_steps must be positive")
    values = np.linspace(int(total) - 1, 0, count, dtype=int)
    return list(dict.fromkeys(int(value) for value in values))


def torch_sqrt(value):
    return _torch().sqrt(value)


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for VPFP continuous diffusion") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
