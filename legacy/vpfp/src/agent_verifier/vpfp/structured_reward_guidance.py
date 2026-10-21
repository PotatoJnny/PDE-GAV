from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_diffusion import timestep_embedding
from agent_verifier.vpfp.structured_diffusion import split_active_latent


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPStructuredRewardModelConfig:
    rho_time_count: int
    rho_feature_count: int
    local_channels: int
    local_time_count: int
    local_spatial_feature_count: int
    condition_dim: int
    rho_sketch_dim: int = 256
    local_sketch_dim: int = 1024
    hidden_width: int = 256
    hidden_layers: int = 3
    time_embedding_dim: int = 64
    sketch_seed: int = 20260730
    dtype: str = "float32"
    representation: str = "vpfp_structured_dual_reward_countsketch_v1"

    @property
    def rho_latent_dim(self) -> int:
        return int(self.rho_time_count) * int(self.rho_feature_count)

    @property
    def local_latent_dim(self) -> int:
        return (
            int(self.local_channels)
            * int(self.local_time_count)
            * int(self.local_spatial_feature_count)
        )

    @property
    def latent_dim(self) -> int:
        return self.rho_latent_dim + self.local_latent_dim

    @property
    def active_latent_dim(self) -> int:
        return self.latent_dim

    @property
    def feature_dim(self) -> int:
        return (
            int(self.rho_sketch_dim)
            + int(self.local_sketch_dim)
            + int(self.rho_time_count)
            + int(self.local_channels)
            + int(self.local_spatial_feature_count)
            + int(self.local_channels) * int(self.local_time_count)
            + int(self.condition_dim)
            + int(self.time_embedding_dim)
        )

    def validate(self) -> None:
        dimensions = (
            self.rho_time_count,
            self.rho_feature_count,
            self.local_channels,
            self.local_time_count,
            self.local_spatial_feature_count,
            self.condition_dim,
            self.rho_sketch_dim,
            self.local_sketch_dim,
            self.hidden_width,
            self.hidden_layers,
            self.time_embedding_dim,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("structured reward dimensions must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("structured reward dtype must be float32 or float64")
        if self.representation != "vpfp_structured_dual_reward_countsketch_v1":
            raise ValueError("unsupported structured reward representation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**dict(data))

    @classmethod
    def from_structured_config(cls, config, **overrides):
        return cls(
            rho_time_count=int(config.rho_time_count),
            rho_feature_count=int(config.rho_feature_count),
            local_channels=int(config.local_channels),
            local_time_count=int(config.local_time_count),
            local_spatial_feature_count=int(
                config.local_spatial_feature_count
            ),
            condition_dim=int(config.condition_dim),
            **overrides,
        )


class VPFPStructuredRewardModel:
    @staticmethod
    def build(config: VPFPStructuredRewardModelConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPStructuredRewardModelConfig)
            else VPFPStructuredRewardModelConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPStructuredRewardModel(_torch(), cfg)


class _VPFPStructuredRewardModel(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    """Low-capacity differentiable reward model over the active PCA latent.

    CountSketch preserves signed phase information without a dense
    34k-by-hidden input layer. Explicit block energies retain interpretable
    kinetic and spatial scale information needed by a black-box solver reward.
    """

    def __init__(self, torch, config: VPFPStructuredRewardModelConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.sketch_seed))
        self.register_buffer(
            "rho_bucket",
            torch.randint(
                int(config.rho_sketch_dim),
                (int(config.rho_latent_dim),),
                generator=generator,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "rho_sign",
            2.0
            * torch.randint(
                2,
                (int(config.rho_latent_dim),),
                generator=generator,
                dtype=torch.float32,
            )
            - 1.0,
        )
        self.register_buffer(
            "local_bucket",
            torch.randint(
                int(config.local_sketch_dim),
                (int(config.local_latent_dim),),
                generator=generator,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "local_sign",
            2.0
            * torch.randint(
                2,
                (int(config.local_latent_dim),),
                generator=generator,
                dtype=torch.float32,
            )
            - 1.0,
        )
        self.feature_norm = torch.nn.LayerNorm(
            int(config.feature_dim), elementwise_affine=False
        )
        self.input = torch.nn.Linear(
            int(config.feature_dim), int(config.hidden_width)
        )
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(int(config.hidden_width)),
                    torch.nn.Linear(
                        int(config.hidden_width), 2 * int(config.hidden_width)
                    ),
                    torch.nn.SiLU(),
                    torch.nn.Linear(
                        2 * int(config.hidden_width), int(config.hidden_width)
                    ),
                )
                for _ in range(int(config.hidden_layers))
            ]
        )
        self.output_norm = torch.nn.LayerNorm(int(config.hidden_width))
        self.absolute_head = torch.nn.Linear(int(config.hidden_width), 1)
        self.centered_head = torch.nn.Linear(int(config.hidden_width), 1)
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
        if latent.ndim != 2 or int(latent.shape[1]) != int(
            self.config.latent_dim
        ):
            raise ValueError("structured reward latent shape mismatch")
        if condition.ndim != 2 or int(condition.shape[1]) != int(
            self.config.condition_dim
        ):
            raise ValueError("structured reward condition shape mismatch")
        if not (
            int(latent.shape[0]) == int(condition.shape[0]) == int(time.shape[0])
        ):
            raise ValueError("structured reward batch dimensions disagree")

        rho, local = split_active_latent(latent, self.config)
        rho_flat = rho.reshape(int(latent.shape[0]), -1)
        local_flat = local.reshape(int(latent.shape[0]), -1)
        epsilon = torch.finfo(latent.dtype).eps
        features = torch.cat(
            (
                self._count_sketch(
                    rho_flat,
                    self.rho_bucket,
                    self.rho_sign,
                    int(self.config.rho_sketch_dim),
                ),
                self._count_sketch(
                    local_flat,
                    self.local_bucket,
                    self.local_sign,
                    int(self.config.local_sketch_dim),
                ),
                torch.sqrt(torch.mean(rho.square(), dim=-1) + epsilon),
                torch.sqrt(
                    torch.mean(local.square(), dim=(2, 3)) + epsilon
                ),
                torch.sqrt(
                    torch.mean(local.square(), dim=(1, 2)) + epsilon
                ),
                local[..., 0].reshape(int(latent.shape[0]), -1),
                condition,
                timestep_embedding(
                    torch,
                    time,
                    int(self.config.time_embedding_dim),
                    dtype=parameter.dtype,
                ),
            ),
            dim=-1,
        )
        hidden = torch.nn.functional.silu(
            self.input(self.feature_norm(features))
        )
        scale = float(max(1, len(self.blocks))) ** -0.5
        for block in self.blocks:
            hidden = hidden + scale * block(hidden)
        hidden = torch.nn.functional.silu(self.output_norm(hidden))
        return {
            "absolute": self.absolute_head(hidden)[:, 0],
            "centered": self.centered_head(hidden)[:, 0],
        }

    def _count_sketch(self, values, buckets, signs, output_dim: int):
        torch = self.torch
        output = torch.zeros(
            (int(values.shape[0]), int(output_dim)),
            dtype=values.dtype,
            device=values.device,
        )
        indices = buckets[None, :].expand(int(values.shape[0]), -1)
        signed = values * signs.to(dtype=values.dtype)[None, :]
        output.scatter_add_(1, indices, signed)
        return output * np.sqrt(float(output_dim) / float(values.shape[1]))


def condition_centered_log_rewards(
    rewards: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rewards, dtype=np.float64)
    groups = np.asarray(group_ids)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("reward groups must align with one-dimensional rewards")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("solver rewards must be finite and nonnegative")
    transformed = np.log1p(values)
    centered = np.empty_like(transformed)
    for group in np.unique(groups):
        selected = groups == group
        if int(np.count_nonzero(selected)) < 2:
            raise ValueError("centered reward needs two samples per condition")
        centered[selected] = transformed[selected] - float(
            np.mean(transformed[selected])
        )
    return transformed, centered


def structured_reward_regression_loss(
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
    loss = (
        float(absolute_weight) * absolute_loss
        + float(centered_weight) * centered_loss
    )
    return loss, {
        "loss": loss,
        "absolute_mse": absolute_loss,
        "centered_mse": centered_loss,
    }


def make_structured_reward_guidance(
    reward_model,
    *,
    absolute_scale: float,
    centered_scale: float,
    rho_latent_dim: int,
    max_gradient_norm: float = 1.0,
    freeze_rho: bool = True,
):
    torch = _torch()
    if not np.isfinite(float(max_gradient_norm)) or float(max_gradient_norm) <= 0:
        raise ValueError("guidance gradient norm must be finite and positive")

    def guidance(current, timestep, condition, predicted_noise, schedule):
        latent = current.detach().requires_grad_(True)
        prediction = reward_model(latent, timestep, condition)
        objective = (
            float(absolute_scale) * prediction["absolute"]
            + float(centered_scale) * prediction["centered"]
        ).sum()
        gradient = torch.autograd.grad(objective, latent, create_graph=False)[0]
        if freeze_rho:
            gradient = gradient.clone()
            gradient[:, : int(rho_latent_dim)] = 0.0
        norm = torch.linalg.vector_norm(
            gradient, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        gradient = gradient * torch.clamp(
            float(max_gradient_norm) / norm, max=1.0
        )
        sigma = torch.sqrt(
            1.0 - schedule["alpha_bar"].index_select(0, timestep)
        )[:, None]
        return predicted_noise.detach() - sigma * gradient.detach()

    return guidance


def project_guided_local_trust_region(
    guided,
    baseline,
    *,
    rho_latent_dim: int,
    maximum_local_rms_delta: float,
    maximum_coordinate_delta: float,
):
    torch = _torch()
    candidate = torch.as_tensor(guided)
    reference = torch.as_tensor(
        baseline, dtype=candidate.dtype, device=candidate.device
    )
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("guided and baseline latents must be matching matrices")
    if not 0 < int(rho_latent_dim) < int(candidate.shape[1]):
        raise ValueError("rho latent boundary is outside the latent")
    if min(float(maximum_local_rms_delta), float(maximum_coordinate_delta)) <= 0:
        raise ValueError("trust-region bounds must be positive")
    output = candidate.clone()
    output[:, : int(rho_latent_dim)] = reference[:, : int(rho_latent_dim)]
    delta = output[:, int(rho_latent_dim) :] - reference[:, int(rho_latent_dim) :]
    delta = torch.clamp(
        delta,
        min=-float(maximum_coordinate_delta),
        max=float(maximum_coordinate_delta),
    )
    rms = torch.sqrt(torch.mean(delta.square(), dim=-1, keepdim=True)).clamp_min(
        1.0e-12
    )
    delta = delta * torch.clamp(
        float(maximum_local_rms_delta) / rms, max=1.0
    )
    output[:, int(rho_latent_dim) :] = (
        reference[:, int(rho_latent_dim) :] + delta
    )
    return output


def reward_regression_metrics(predicted, target) -> dict[str, float | bool]:
    first = np.asarray(predicted, dtype=np.float64).reshape(-1)
    second = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = bool(
        first.shape == second.shape
        and first.size >= 2
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
    )
    if not finite:
        return {
            "finite": False,
            "count": int(min(first.size, second.size)),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "normalized_rmse": float("nan"),
        }
    return {
        "finite": True,
        "count": int(first.size),
        "pearson": _correlation(first, second),
        "spearman": _correlation(_ranks(first), _ranks(second)),
        "normalized_rmse": float(
            np.sqrt(np.mean((first - second) ** 2))
            / max(float(np.std(second)), 1.0e-12)
        ),
    }


def save_structured_reward_checkpoint(
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
            "schema": "vpfp_structured_solver_reward_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "reward_normalization": dict(reward_normalization),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
        },
        destination,
    )


def load_structured_reward_checkpoint(path: str | Path, *, device="cpu"):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_structured_solver_reward_checkpoint_v1":
        raise ValueError("unexpected structured reward checkpoint schema")
    model = VPFPStructuredRewardModel.build(payload["model_config"]).to(
        device=device
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if float(np.std(first)) <= 1.0e-12 or float(np.std(second)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError(
            "torch is required for structured VPFP reward guidance"
        ) from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
