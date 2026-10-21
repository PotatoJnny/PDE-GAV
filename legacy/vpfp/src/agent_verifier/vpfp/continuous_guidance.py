from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_diffusion import timestep_embedding


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPContinuousRewardConfig:
    point_count: int = 512
    broad_velocity_fraction: float = 0.25
    seed: int = 20260712
    solver: str = "hold_initial_public_rho0_g0"
    reward: str = "f_relative_l2"

    def validate(self) -> None:
        if self.point_count < 8:
            raise ValueError("reward evaluation requires at least eight points")
        if not 0.0 <= self.broad_velocity_fraction <= 1.0:
            raise ValueError("broad_velocity_fraction must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VPFPContinuousRewardModelConfig:
    latent_dim: int = 12
    condition_dim: int = 4
    hidden_width: int = 192
    hidden_layers: int = 4
    time_embedding_dim: int = 64
    dtype: str = "float32"
    representation: str = "vpfp_continuous_dual_reward_guidance_v1"

    def validate(self) -> None:
        if min(
            self.latent_dim,
            self.condition_dim,
            self.hidden_width,
            self.hidden_layers,
            self.time_embedding_dim,
        ) < 1:
            raise ValueError("reward model dimensions must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("reward model dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**dict(data))


class VPFPContinuousRewardModel:
    @staticmethod
    def build(config: VPFPContinuousRewardModelConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPContinuousRewardModelConfig)
            else VPFPContinuousRewardModelConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPContinuousRewardModel(_torch(), cfg)


class _VPFPContinuousRewardModel(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config: VPFPContinuousRewardModelConfig):
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
                torch.nn.Sequential(
                    torch.nn.LayerNorm(config.hidden_width),
                    torch.nn.Linear(config.hidden_width, 2 * config.hidden_width),
                    torch.nn.SiLU(),
                    torch.nn.Linear(2 * config.hidden_width, config.hidden_width),
                )
                for _ in range(int(config.hidden_layers))
            ]
        )
        self.output_norm = torch.nn.LayerNorm(config.hidden_width)
        self.absolute_head = torch.nn.Linear(config.hidden_width, 1)
        self.centered_head = torch.nn.Linear(config.hidden_width, 1)
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
            hidden = hidden + block_scale * block(hidden)
        hidden = torch.nn.functional.silu(self.output_norm(hidden))
        return {
            "absolute": self.absolute_head(hidden)[:, 0],
            "centered": self.centered_head(hidden)[:, 0],
        }


def make_reward_phase_points(model, config: VPFPContinuousRewardConfig, *, device):
    config.validate()
    torch = _torch()
    parameter = next(model.parameters())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed))
    count = int(config.point_count)
    t0, t1 = model.config.time_domain
    lx, ly = model.config.equation.x_domain
    t = float(t0) + 0.1 * float(t1 - t0) + 0.9 * float(t1 - t0) * torch.rand(
        count, generator=generator, device=device, dtype=parameter.dtype
    )
    x = float(lx) * torch.rand(
        count, generator=generator, device=device, dtype=parameter.dtype
    )
    y = float(ly) * torch.rand(
        count, generator=generator, device=device, dtype=parameter.dtype
    )
    theta = float(model.config.equation.temperature)
    velocity = theta**0.5 * torch.randn(
        (count, 2), generator=generator, device=device, dtype=parameter.dtype
    )
    broad_count = int(round(float(config.broad_velocity_fraction) * count))
    if broad_count:
        velocity[-broad_count:] = 2.5 * theta**0.5 * torch.randn(
            (broad_count, 2),
            generator=generator,
            device=device,
            dtype=parameter.dtype,
        )
    return torch.stack((t, x, y, velocity[:, 0], velocity[:, 1]), dim=-1)


def hold_initial_solver_reward(model, latents, phase_points):
    """Evaluate a fixed public-data solver that returns rho0,g0 at every time."""

    torch = _torch()
    parameter = next(model.parameters())
    latent = torch.as_tensor(
        latents, dtype=parameter.dtype, device=parameter.device
    )
    if latent.ndim == 1:
        latent = latent[None, :]
    points = torch.as_tensor(
        phase_points, dtype=parameter.dtype, device=parameter.device
    )
    batch = int(latent.shape[0])
    point_count = int(points.shape[0])
    query = points[None, :, :].expand(batch, -1, -1).reshape(-1, 5)
    initial_query = query.clone()
    initial_query[:, 0] = float(model.config.time_domain[0])
    latent_rows = latent.repeat_interleave(point_count, dim=0)
    truth = model.fields(query, latent_rows)
    prediction = model.fields(initial_query, latent_rows)
    epsilon = torch.finfo(parameter.dtype).eps

    def relative(name):
        target = truth[name].reshape(batch, point_count)
        estimate = prediction[name].reshape(batch, point_count)
        return torch.sqrt(
            torch.mean((estimate - target).square(), dim=1)
            / (torch.mean(target.square(), dim=1) + epsilon)
        )

    f_relative = relative("f")
    rho_relative = relative("rho")
    g_relative = relative("g")
    return {
        "reward": f_relative,
        "f_relative_l2": f_relative,
        "rho_relative_l2": rho_relative,
        "g_relative_l2": g_relative,
    }


def grouped_centered_rewards(rewards, group_ids):
    torch = _torch()
    values = torch.as_tensor(rewards)
    groups = torch.as_tensor(group_ids, device=values.device)
    centered = torch.empty_like(values)
    for group in torch.unique(groups):
        mask = groups == group
        centered[mask] = values[mask] - torch.mean(values[mask])
    return centered


def reward_regression_loss(
    model,
    noisy_latent,
    timesteps,
    conditions,
    *,
    absolute_target,
    centered_target,
    absolute_weight: float = 1.0,
    centered_weight: float = 1.0,
):
    torch = _torch()
    prediction = model(noisy_latent, timesteps, conditions)
    absolute = torch.as_tensor(
        absolute_target,
        dtype=prediction["absolute"].dtype,
        device=prediction["absolute"].device,
    )
    centered = torch.as_tensor(
        centered_target,
        dtype=prediction["centered"].dtype,
        device=prediction["centered"].device,
    )
    absolute_loss = torch.mean((prediction["absolute"] - absolute).square())
    centered_loss = torch.mean((prediction["centered"] - centered).square())
    total = (
        float(absolute_weight) * absolute_loss
        + float(centered_weight) * centered_loss
    )
    return total, {
        "loss": total,
        "absolute_mse": absolute_loss,
        "centered_mse": centered_loss,
    }


def make_reward_guidance(
    reward_model,
    *,
    absolute_scale: float,
    centered_scale: float,
    max_gradient_norm: float = 5.0,
):
    torch = _torch()

    def guidance(current, timestep, condition, predicted_noise, schedule):
        latent = current.detach().requires_grad_(True)
        prediction = reward_model(latent, timestep, condition)
        objective = (
            float(absolute_scale) * prediction["absolute"]
            + float(centered_scale) * prediction["centered"]
        ).sum()
        gradient = torch.autograd.grad(objective, latent, create_graph=False)[0]
        norm = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        scale = torch.clamp(float(max_gradient_norm) / norm, max=1.0)
        gradient = (gradient * scale).detach()
        sigma = torch.sqrt(
            1.0 - schedule["alpha_bar"].index_select(0, timestep)
        )[:, None]
        return predicted_noise.detach() - sigma * gradient

    return guidance


def save_continuous_reward_checkpoint(
    path: str | Path,
    *,
    model,
    reward_normalization: dict[str, Any],
    train_config: dict[str, Any],
    step: int,
    optimizer=None,
) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_continuous_reward_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "reward_normalization": reward_normalization,
            "train_config": train_config,
            "step": int(step),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
        },
        destination,
    )


def load_continuous_reward_checkpoint(path: str | Path, *, device="cpu"):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_continuous_reward_checkpoint_v1":
        raise ValueError("unexpected continuous reward checkpoint schema")
    model = VPFPContinuousRewardModel.build(payload["model_config"]).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for VPFP continuous guidance") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
