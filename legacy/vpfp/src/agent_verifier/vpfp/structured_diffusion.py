from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_v2 import _real_fourier_half_modes
from agent_verifier.vpfp.continuous_diffusion import (
    diffusion_model_predictions,
    diffusion_schedule,
    diffusion_training_target,
    q_sample,
    sample_continuous_latents,
    timestep_embedding,
)


_ACTIVE_LATENT_NORMALIZATION_SCHEMAS = {
    "coordinate_whitened": (
        "vpfp_translation_invariant_latent_normalization_v1"
    ),
    "block_energy": "vpfp_block_energy_structured_normalization_v1",
    "channel_energy": "vpfp_channel_energy_structured_normalization_v1",
}


def structured_active_normalization_schema(normalization_mode: str) -> str:
    """Return the registered active-latent schema for a training mode."""

    try:
        return _ACTIVE_LATENT_NORMALIZATION_SCHEMAS[str(normalization_mode)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported structured normalization mode: {normalization_mode}"
        ) from exc


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPStructuredDiffusionConfig:
    rho_time_order: int = 2
    rho_mode_count: int = 84
    kinetic_global_dim: int = 64
    local_channels: int = 120
    local_time_order: int = 2
    local_spatial_mode: int = 4
    rho_time_feature_count: int = 0
    local_time_feature_count: int = 0
    condition_dim: int = 2
    width: int = 256
    layers: int = 6
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    architecture: str = "transformer"
    physical_grid_size: int = 13
    dropout: float = 0.0
    time_embedding_dim: int = 128
    timesteps: int = 200
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    noise_schedule: str = "linear"
    prediction_type: str = "epsilon"
    dtype: str = "float32"
    representation: str = "vpfp_structured_direct_spectral_diffusion_v1"

    @property
    def rho_time_count(self) -> int:
        if int(self.rho_time_feature_count) > 0:
            return int(self.rho_time_feature_count)
        return int(self.rho_time_order) + 1

    @property
    def rho_feature_count(self) -> int:
        return 2 * int(self.rho_mode_count)

    @property
    def rho_latent_dim(self) -> int:
        return self.rho_time_count * self.rho_feature_count

    @property
    def local_time_count(self) -> int:
        if int(self.local_time_feature_count) > 0:
            return int(self.local_time_feature_count)
        return int(self.local_time_order) + 1

    @property
    def local_spatial_feature_count(self) -> int:
        return (2 * int(self.local_spatial_mode) + 1) ** 2

    @property
    def local_latent_dim(self) -> int:
        return (
            int(self.local_channels)
            * self.local_time_count
            * self.local_spatial_feature_count
        )

    @property
    def active_latent_dim(self) -> int:
        return self.rho_latent_dim + self.local_latent_dim

    @property
    def full_latent_dim(self) -> int:
        return self.active_latent_dim + int(self.kinetic_global_dim)

    @property
    def latent_dim(self) -> int:
        return self.active_latent_dim

    @property
    def token_count(self) -> int:
        return (
            self.rho_time_count
            + self.local_time_count * self.local_spatial_feature_count
        )

    def validate(self) -> None:
        if min(
            self.rho_mode_count,
            self.local_channels,
            self.condition_dim,
            self.width,
            self.layers,
            self.attention_heads,
            self.feedforward_multiplier,
            self.time_embedding_dim,
            self.timesteps,
        ) < 1:
            raise ValueError("structured diffusion dimensions must be positive")
        if min(
            self.rho_time_order,
            self.kinetic_global_dim,
            self.local_time_order,
            self.local_spatial_mode,
            self.rho_time_feature_count,
            self.local_time_feature_count,
        ) < 0:
            raise ValueError("structured diffusion orders must be nonnegative")
        if self.architecture not in {
            "transformer",
            "dit_transformer",
            "periodic_conv",
        }:
            raise ValueError("unsupported structured diffusion architecture")
        if (
            self.architecture in {"transformer", "dit_transformer"}
            and self.width % self.attention_heads
        ):
            raise ValueError("width must be divisible by attention_heads")
        if self.architecture == "periodic_conv":
            rho_max_mode = _rho_max_mode(self)
            if int(self.physical_grid_size) < 2 * rho_max_mode + 1:
                raise ValueError("physical grid aliases represented rho modes")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("structured diffusion beta schedule is invalid")
        if self.noise_schedule not in {"linear", "cosine"}:
            raise ValueError("noise_schedule must be linear or cosine")
        if self.prediction_type not in {"epsilon", "v"}:
            raise ValueError("prediction_type must be epsilon or v")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "VPFPStructuredDiffusionConfig":
        return cls(**dict(data))


def resolve_structured_diffusion_evaluation_artifacts(
    report_path: str | Path,
    *,
    latent_artifact: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_selection: str = "best",
) -> dict[str, Any]:
    """Resolve offline or online diffusion artifacts for sampling and audit."""

    path = Path(report_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    schema = str(report.get("schema") or "")
    if checkpoint_selection not in {"best", "final"}:
        raise ValueError("checkpoint_selection must be 'best' or 'final'")
    if schema == "vpfp_structured_direct_diffusion_report_v1":
        default_checkpoint = report.get("checkpoint")
        default_latent = report.get("latent_artifact")
        route = "offline_static_corpus"
    elif schema == "vpfp_structured_diffusion_online_report_v1":
        checkpoint_key = (
            "best_checkpoint"
            if checkpoint_selection == "best"
            else "final_checkpoint"
        )
        default_checkpoint = report.get(checkpoint_key)
        default_latent = report.get("evaluation_latent_artifact")
        route = "online_equation_first_stream"
    else:
        raise ValueError(
            f"unexpected structured diffusion report schema: {schema}"
        )

    checkpoint = Path(checkpoint_path or default_checkpoint or "").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"structured diffusion checkpoint is missing: {checkpoint}"
        )
    latent_value = latent_artifact or default_latent
    if not latent_value:
        raise ValueError(
            "online diffusion evaluation requires an independent "
            "--latent-artifact holdout"
        )
    latent = Path(latent_value).resolve()
    if not latent.is_file():
        raise FileNotFoundError(
            f"structured diffusion latent artifact is missing: {latent}"
        )
    return {
        "report": report,
        "report_path": path,
        "report_schema": schema,
        "route": route,
        "checkpoint": checkpoint,
        "latent_artifact": latent,
        "checkpoint_selection": str(checkpoint_selection),
        "checkpoint_overridden": checkpoint_path is not None,
        "latent_artifact_overridden": latent_artifact is not None,
    }


class VPFPStructuredDenoiser:
    @staticmethod
    def build(config: VPFPStructuredDiffusionConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPStructuredDiffusionConfig)
            else VPFPStructuredDiffusionConfig.from_dict(dict(config))
        )
        cfg.validate()
        if cfg.architecture == "periodic_conv":
            return _VPFPPeriodicGridDenoiser(_torch(), cfg)
        if cfg.architecture == "dit_transformer":
            return _VPFPStructuredDiTDenoiser(_torch(), cfg)
        return _VPFPStructuredDenoiser(_torch(), cfg)


class VPFPBlockwiseScoreEnsemble:
    """Use independently trained rho and local score blocks in one trajectory."""

    @staticmethod
    def build(*, rho_model, local_model):
        return _VPFPBlockwiseScoreEnsemble(_torch(), rho_model, local_model)


class _VPFPBlockwiseScoreEnsemble(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    _COMPATIBILITY_FIELDS = (
        "rho_time_order",
        "rho_time_feature_count",
        "rho_mode_count",
        "kinetic_global_dim",
        "local_channels",
        "local_time_order",
        "local_time_feature_count",
        "local_spatial_mode",
        "condition_dim",
        "timesteps",
        "beta_start",
        "beta_end",
        "noise_schedule",
        "prediction_type",
        "dtype",
        "representation",
    )

    def __init__(self, torch, rho_model, local_model):
        super().__init__()
        self.torch = torch
        self.rho_model = rho_model
        self.local_model = local_model
        self.config = local_model.config
        mismatches = [
            name
            for name in self._COMPATIBILITY_FIELDS
            if getattr(rho_model.config, name) != getattr(local_model.config, name)
        ]
        if mismatches:
            raise ValueError(
                "blockwise score models are incompatible: "
                + ", ".join(mismatches)
            )

    def forward(self, noisy_latent, timesteps, conditions):
        rho_prediction = self.rho_model(noisy_latent, timesteps, conditions)
        local_prediction = self.local_model(noisy_latent, timesteps, conditions)
        rho, _ = split_active_latent(rho_prediction, self.config)
        _, local = split_active_latent(local_prediction, self.config)
        return join_active_latent(rho, local, self.config)


class _VPFPStructuredDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config: VPFPStructuredDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        width = int(config.width)
        self.rho_input = torch.nn.Linear(config.rho_feature_count, width)
        self.local_input = torch.nn.Linear(config.local_channels, width)
        self.rho_time_position = torch.nn.Parameter(
            torch.randn(config.rho_time_count, width) / np.sqrt(float(width))
        )
        self.local_time_position = torch.nn.Parameter(
            torch.randn(config.local_time_count, width) / np.sqrt(float(width))
        )
        self.local_spatial_position = torch.nn.Parameter(
            torch.randn(config.local_spatial_feature_count, width)
            / np.sqrt(float(width))
        )
        self.token_type = torch.nn.Parameter(
            torch.randn(2, width) / np.sqrt(float(width))
        )
        self.diffusion_time = torch.nn.Sequential(
            torch.nn.Linear(config.time_embedding_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(config.condition_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(config.attention_heads),
            dim_feedforward=int(config.feedforward_multiplier) * width,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=int(config.layers),
            enable_nested_tensor=False,
        )
        self.output_norm = torch.nn.LayerNorm(width)
        self.rho_output = torch.nn.Linear(width, config.rho_feature_count)
        self.local_output = torch.nn.Linear(width, config.local_channels)
        torch.nn.init.zeros_(self.rho_output.weight)
        torch.nn.init.zeros_(self.rho_output.bias)
        torch.nn.init.zeros_(self.local_output.weight)
        torch.nn.init.zeros_(self.local_output.bias)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy_latent, timesteps, conditions):
        torch = self.torch
        parameter = next(self.parameters())
        latent = torch.as_tensor(
            noisy_latent,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        condition = torch.as_tensor(
            conditions,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        time = torch.as_tensor(
            timesteps,
            device=parameter.device,
        ).reshape(-1)
        if latent.ndim != 2 or int(latent.shape[1]) != self.config.latent_dim:
            raise ValueError("noisy latent shape does not match structured config")
        if (
            condition.ndim != 2
            or int(condition.shape[1]) != self.config.condition_dim
        ):
            raise ValueError("condition shape does not match structured config")
        batch = int(latent.shape[0])
        if int(time.shape[0]) != batch or int(condition.shape[0]) != batch:
            raise ValueError("structured diffusion batch dimensions do not match")

        rho, local = split_active_latent(latent, self.config)
        rho_tokens = self.rho_input(rho)
        local_tokens = self.local_input(
            local.permute(0, 2, 3, 1).reshape(
                batch,
                self.config.local_time_count
                * self.config.local_spatial_feature_count,
                self.config.local_channels,
            )
        )
        rho_tokens = rho_tokens + (
            self.rho_time_position[None]
            + self.token_type[None, :1]
        )
        local_position = (
            self.local_time_position[:, None, :]
            + self.local_spatial_position[None, :, :]
        ).reshape(1, -1, self.config.width)
        local_tokens = local_tokens + (
            local_position + self.token_type[None, 1:2]
        )
        diffusion_embedding = timestep_embedding(
            torch,
            time,
            self.config.time_embedding_dim,
            dtype=parameter.dtype,
        )
        global_context = (
            self.diffusion_time(diffusion_embedding)
            + self.condition(condition)
        )[:, None, :]
        tokens = torch.cat((rho_tokens, local_tokens), dim=1)
        tokens = self.output_norm(
            self.transformer(tokens + global_context)
        )
        rho_count = self.config.rho_time_count
        rho_prediction = self.rho_output(tokens[:, :rho_count])
        local_prediction = self.local_output(tokens[:, rho_count:])
        local_prediction = local_prediction.reshape(
            batch,
            self.config.local_time_count,
            self.config.local_spatial_feature_count,
            self.config.local_channels,
        ).permute(0, 3, 1, 2)
        return join_active_latent(
            rho_prediction,
            local_prediction,
            self.config,
        )


class _VPFPDiTBlock(
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
        self.mlp_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        hidden = int(feedforward_multiplier) * int(width)
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
            1.0 + attention_scale[:, None, :]
        ) + attention_shift[:, None, :]
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + attention_gate[:, None, :] * attention_output
        mlp_input = self.mlp_norm(tokens)
        mlp_input = mlp_input * (
            1.0 + mlp_scale[:, None, :]
        ) + mlp_shift[:, None, :]
        return tokens + mlp_gate[:, None, :] * self.mlp(mlp_input)


class _VPFPStructuredDiTDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    """Structured DiT with time and equation conditioning in every block."""

    def __init__(self, torch, config: VPFPStructuredDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        width = int(config.width)
        self.rho_input = torch.nn.Linear(config.rho_feature_count, width)
        self.local_input = torch.nn.Linear(config.local_channels, width)
        self.rho_time_position = torch.nn.Parameter(
            torch.randn(config.rho_time_count, width) / np.sqrt(float(width))
        )
        self.local_time_position = torch.nn.Parameter(
            torch.randn(config.local_time_count, width) / np.sqrt(float(width))
        )
        self.local_spatial_position = torch.nn.Parameter(
            torch.randn(config.local_spatial_feature_count, width)
            / np.sqrt(float(width))
        )
        self.token_type = torch.nn.Parameter(
            torch.randn(2, width) / np.sqrt(float(width))
        )
        self.diffusion_time = torch.nn.Sequential(
            torch.nn.Linear(config.time_embedding_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(config.condition_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.blocks = torch.nn.ModuleList(
            [
                _VPFPDiTBlock(
                    torch,
                    width=width,
                    attention_heads=int(config.attention_heads),
                    feedforward_multiplier=int(config.feedforward_multiplier),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.layers))
            ]
        )
        self.final_norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, eps=1.0e-6
        )
        self.final_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(width, 2 * width),
        )
        torch.nn.init.zeros_(self.final_modulation[-1].weight)
        torch.nn.init.zeros_(self.final_modulation[-1].bias)
        self.rho_output = torch.nn.Linear(width, config.rho_feature_count)
        self.local_output = torch.nn.Linear(width, config.local_channels)
        torch.nn.init.zeros_(self.rho_output.weight)
        torch.nn.init.zeros_(self.rho_output.bias)
        torch.nn.init.zeros_(self.local_output.weight)
        torch.nn.init.zeros_(self.local_output.bias)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy_latent, timesteps, conditions):
        torch = self.torch
        parameter = next(self.parameters())
        latent = torch.as_tensor(
            noisy_latent,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        condition = torch.as_tensor(
            conditions,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        time = torch.as_tensor(timesteps, device=parameter.device).reshape(-1)
        if latent.ndim != 2 or int(latent.shape[1]) != self.config.latent_dim:
            raise ValueError("noisy latent shape does not match structured config")
        if (
            condition.ndim != 2
            or int(condition.shape[1]) != self.config.condition_dim
        ):
            raise ValueError("condition shape does not match structured config")
        batch = int(latent.shape[0])
        if int(time.shape[0]) != batch or int(condition.shape[0]) != batch:
            raise ValueError("structured diffusion batch dimensions do not match")

        rho, local = split_active_latent(latent, self.config)
        rho_tokens = self.rho_input(rho)
        local_tokens = self.local_input(
            local.permute(0, 2, 3, 1).reshape(
                batch,
                self.config.local_time_count
                * self.config.local_spatial_feature_count,
                self.config.local_channels,
            )
        )
        rho_tokens = rho_tokens + (
            self.rho_time_position[None] + self.token_type[None, :1]
        )
        local_position = (
            self.local_time_position[:, None, :]
            + self.local_spatial_position[None, :, :]
        ).reshape(1, -1, self.config.width)
        local_tokens = local_tokens + (
            local_position + self.token_type[None, 1:2]
        )
        time_embedding = timestep_embedding(
            torch,
            time,
            self.config.time_embedding_dim,
            dtype=parameter.dtype,
        )
        context = self.diffusion_time(time_embedding) + self.condition(condition)
        tokens = torch.cat((rho_tokens, local_tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens, context)
        final_shift, final_scale = self.final_modulation(context).chunk(
            2, dim=-1
        )
        tokens = self.final_norm(tokens) * (
            1.0 + final_scale[:, None, :]
        ) + final_shift[:, None, :]
        rho_count = self.config.rho_time_count
        rho_prediction = self.rho_output(tokens[:, :rho_count])
        local_prediction = self.local_output(tokens[:, rho_count:])
        local_prediction = local_prediction.reshape(
            batch,
            self.config.local_time_count,
            self.config.local_spatial_feature_count,
            self.config.local_channels,
        ).permute(0, 3, 1, 2)
        return join_active_latent(
            rho_prediction,
            local_prediction,
            self.config,
        )


class _PeriodicResidualBlock(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, width: int, dropout: float):
        super().__init__()
        groups = _group_count(width)
        self.norm1 = torch.nn.GroupNorm(groups, width)
        self.norm2 = torch.nn.GroupNorm(groups, width)
        self.context = torch.nn.Linear(width, 2 * width)
        self.conv1 = torch.nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.conv2 = torch.nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.dropout = torch.nn.Dropout2d(float(dropout))

    def forward(self, values, context):
        scale, shift = self.context(context).chunk(2, dim=-1)
        hidden = self.norm1(values)
        hidden = hidden * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv1(self.torch_silu(hidden))
        hidden = self.conv2(self.dropout(self.torch_silu(self.norm2(hidden))))
        return (values + hidden) / np.sqrt(2.0)

    @staticmethod
    def torch_silu(values):
        return _torch().nn.functional.silu(values)


class _VPFPPeriodicGridDenoiser(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    """Score network on an internal periodic grid, projected to spectral fields."""

    def __init__(self, torch, config: VPFPStructuredDiffusionConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        grid_size = int(config.physical_grid_size)
        rho_basis = _physical_fourier_basis(
            torch,
            max_mode=_rho_max_mode(config),
            grid_size=grid_size,
            include_constant=False,
        )
        local_basis = _physical_fourier_basis(
            torch,
            max_mode=int(config.local_spatial_mode),
            grid_size=grid_size,
            include_constant=True,
        )
        if int(rho_basis.shape[1]) != int(config.rho_feature_count):
            raise ValueError("rho physical basis does not match latent layout")
        if int(local_basis.shape[1]) != int(config.local_spatial_feature_count):
            raise ValueError("local physical basis does not match latent layout")
        self.register_buffer("rho_basis", rho_basis)
        self.register_buffer("local_basis", local_basis)

        field_channels = (
            int(config.rho_time_count)
            + int(config.local_channels) * int(config.local_time_count)
        )
        width = int(config.width)
        self.input = torch.nn.Conv2d(
            field_channels,
            width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.diffusion_time = torch.nn.Sequential(
            torch.nn.Linear(config.time_embedding_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(config.condition_dim, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.blocks = torch.nn.ModuleList(
            [
                _PeriodicResidualBlock(torch, width, float(config.dropout))
                for _ in range(int(config.layers))
            ]
        )
        self.output_norm = torch.nn.GroupNorm(_group_count(width), width)
        self.output = torch.nn.Conv2d(
            width,
            field_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy_latent, timesteps, conditions):
        torch = self.torch
        parameter = next(self.parameters())
        latent = torch.as_tensor(
            noisy_latent,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        condition = torch.as_tensor(
            conditions,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        time = torch.as_tensor(timesteps, device=parameter.device).reshape(-1)
        if latent.ndim != 2 or int(latent.shape[1]) != self.config.latent_dim:
            raise ValueError("noisy latent shape does not match structured config")
        if condition.ndim != 2 or int(condition.shape[1]) != self.config.condition_dim:
            raise ValueError("condition shape does not match structured config")
        batch = int(latent.shape[0])
        if int(time.shape[0]) != batch or int(condition.shape[0]) != batch:
            raise ValueError("structured diffusion batch dimensions do not match")

        grid = self.latent_to_physical_grid(latent)
        embedding = timestep_embedding(
            torch,
            time,
            self.config.time_embedding_dim,
            dtype=parameter.dtype,
        )
        context = self.diffusion_time(embedding) + self.condition(condition)
        hidden = self.input(grid)
        for block in self.blocks:
            hidden = block(hidden, context)
        prediction_grid = self.output(
            torch.nn.functional.silu(self.output_norm(hidden))
        )
        return self.physical_grid_to_latent(prediction_grid)

    def latent_to_physical_grid(self, values):
        rho, local = split_active_latent(values, self.config)
        batch = int(values.shape[0])
        size = int(self.config.physical_grid_size)
        rho_grid = self.torch.einsum(
            "btf,pf->btp", rho, self.rho_basis
        ).reshape(batch, self.config.rho_time_count, size, size)
        local_grid = self.torch.einsum(
            "bctf,pf->bctp", local, self.local_basis
        ).reshape(
            batch,
            self.config.local_channels * self.config.local_time_count,
            size,
            size,
        )
        return self.torch.cat((rho_grid, local_grid), dim=1)

    def physical_grid_to_latent(self, values):
        batch = int(values.shape[0])
        size = int(self.config.physical_grid_size)
        expected_channels = (
            self.config.rho_time_count
            + self.config.local_channels * self.config.local_time_count
        )
        if tuple(values.shape) != (batch, expected_channels, size, size):
            raise ValueError("physical grid shape does not match structured config")
        flat = values.reshape(batch, expected_channels, size * size)
        rho_grid = flat[:, : self.config.rho_time_count]
        local_grid = flat[:, self.config.rho_time_count :].reshape(
            batch,
            self.config.local_channels,
            self.config.local_time_count,
            size * size,
        )
        rho = self.torch.einsum("btp,pf->btf", rho_grid, self.rho_basis)
        local = self.torch.einsum(
            "bctp,pf->bctf", local_grid, self.local_basis
        )
        return join_active_latent(rho, local, self.config)


def split_active_latent(values, config: VPFPStructuredDiffusionConfig):
    if values.ndim != 2 or int(values.shape[1]) != config.active_latent_dim:
        raise ValueError("active latent shape does not match structured config")
    batch = int(values.shape[0])
    rho = values[:, : config.rho_latent_dim].reshape(
        batch,
        config.rho_time_count,
        config.rho_feature_count,
    )
    local = values[:, config.rho_latent_dim :].reshape(
        batch,
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    return rho, local


def join_active_latent(rho, local, config: VPFPStructuredDiffusionConfig):
    batch = int(rho.shape[0])
    if tuple(rho.shape) != (
        batch,
        config.rho_time_count,
        config.rho_feature_count,
    ):
        raise ValueError("rho latent block shape does not match structured config")
    if tuple(local.shape) != (
        batch,
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    ):
        raise ValueError("local latent block shape does not match structured config")
    return _torch().cat(
        (rho.reshape(batch, -1), local.reshape(batch, -1)),
        dim=-1,
    )


def active_latent_from_full(values, config: VPFPStructuredDiffusionConfig):
    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or int(tensor.shape[1]) != config.full_latent_dim:
        raise ValueError("full latent shape does not match structured config")
    rho_end = config.rho_latent_dim
    local_start = rho_end + int(config.kinetic_global_dim)
    return torch.cat((tensor[:, :rho_end], tensor[:, local_start:]), dim=-1)


def full_latent_from_active(values, config: VPFPStructuredDiffusionConfig):
    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or int(tensor.shape[1]) != config.active_latent_dim:
        raise ValueError("active latent shape does not match structured config")
    batch = int(tensor.shape[0])
    rho_end = config.rho_latent_dim
    zeros = torch.zeros(
        (batch, int(config.kinetic_global_dim)),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat(
        (tensor[:, :rho_end], zeros, tensor[:, rho_end:]),
        dim=-1,
    )


def structured_diffusion_loss(
    model,
    clean_latent,
    conditions,
    *,
    timesteps=None,
    noise=None,
    rho_noise_weight: float = 1.0,
    local_noise_weight: float = 1.0,
    spectral_energy_weight: float = 0.0,
    rho_spectral_energy_weight: float | None = None,
    local_spectral_energy_weight: float | None = None,
    canonical_energy_transform=None,
    canonical_energy_config=None,
    canonical_rho_spectral_energy_weight: float = 0.0,
    canonical_local_spectral_energy_weight: float = 0.0,
    spectral_energy_max_timestep_fraction: float = 0.5,
    rho_batch_swd_weight: float = 0.0,
    local_batch_swd_weight: float = 0.0,
    batch_swd_projections: int = 32,
    batch_swd_max_timestep_fraction: float = 0.5,
    batch_swd_projection_seed: int = 1701,
):
    torch = _torch()
    parameter = next(model.parameters())
    clean = torch.as_tensor(
        clean_latent,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    condition = torch.as_tensor(
        conditions,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if timesteps is None:
        timesteps = torch.randint(
            0,
            int(model.config.timesteps),
            (int(clean.shape[0]),),
            device=parameter.device,
        )
    else:
        timesteps = torch.as_tensor(
            timesteps,
            dtype=torch.long,
            device=parameter.device,
        ).reshape(-1)
    if int(timesteps.shape[0]) != int(clean.shape[0]):
        raise ValueError("diffusion timesteps do not match the clean batch")
    if noise is None:
        noise = torch.randn_like(clean)
    else:
        noise = torch.as_tensor(
            noise,
            dtype=parameter.dtype,
            device=parameter.device,
        )
    if noise.shape != clean.shape:
        raise ValueError("diffusion noise does not match the clean batch")
    schedule = diffusion_schedule(
        model.config,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    noisy = q_sample(clean, timesteps, noise, schedule)
    prediction = model(noisy, timesteps, condition)
    target = diffusion_training_target(
        clean,
        noise,
        timesteps,
        schedule,
        str(getattr(model.config, "prediction_type", "epsilon")),
    )
    predicted_rho, predicted_local = split_active_latent(
        prediction,
        model.config,
    )
    target_rho, target_local = split_active_latent(target, model.config)
    rho_mse = torch.mean((predicted_rho - target_rho).square())
    local_mse = torch.mean((predicted_local - target_local).square())
    noise_weights = (float(rho_noise_weight), float(local_noise_weight))
    if any(not np.isfinite(value) or value < 0.0 for value in noise_weights):
        raise ValueError("rho/local noise weights must be finite and nonnegative")
    weight_sum = sum(noise_weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one rho/local noise weight must be positive")
    rho_scale = 2.0 * noise_weights[0] / weight_sum
    local_scale = 2.0 * noise_weights[1] / weight_sum
    noise_loss = 0.5 * (rho_scale * rho_mse + local_scale * local_mse)
    loss = noise_loss
    metrics = {
        "loss": loss,
        "rho_noise_mse": rho_mse,
        "local_noise_mse": local_mse,
        "timestep_mean": timesteps.to(dtype=parameter.dtype).mean(),
    }
    weights = {
        "spectral_energy_weight": spectral_energy_weight,
        "rho_spectral_energy_weight": rho_spectral_energy_weight,
        "local_spectral_energy_weight": local_spectral_energy_weight,
        "canonical_rho_spectral_energy_weight": canonical_rho_spectral_energy_weight,
        "canonical_local_spectral_energy_weight": canonical_local_spectral_energy_weight,
        "rho_batch_swd_weight": rho_batch_swd_weight,
        "local_batch_swd_weight": local_batch_swd_weight,
    }
    for name, value in weights.items():
        if value is not None and (
            not np.isfinite(float(value)) or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 < float(spectral_energy_max_timestep_fraction) <= 1.0:
        raise ValueError("spectral energy timestep fraction must lie in (0,1]")
    if int(batch_swd_projections) < 1:
        raise ValueError("batch SWD projection count must be positive")
    if not 0 <= int(batch_swd_projection_seed) < 2**63 - 1:
        raise ValueError("batch SWD projection seed is outside the valid range")
    if not 0.0 < float(batch_swd_max_timestep_fraction) <= 1.0:
        raise ValueError("batch SWD timestep fraction must lie in (0,1]")
    rho_energy_weight = (
        0.5 * float(spectral_energy_weight)
        if rho_spectral_energy_weight is None
        else float(rho_spectral_energy_weight)
    )
    local_energy_weight = (
        0.5 * float(spectral_energy_weight)
        if local_spectral_energy_weight is None
        else float(local_spectral_energy_weight)
    )
    canonical_rho_energy_weight = float(canonical_rho_spectral_energy_weight)
    canonical_local_energy_weight = float(canonical_local_spectral_energy_weight)
    rho_swd_weight = float(rho_batch_swd_weight)
    local_swd_weight = float(local_batch_swd_weight)
    has_energy_loss = (
        rho_energy_weight > 0.0
        or local_energy_weight > 0.0
        or canonical_rho_energy_weight > 0.0
        or canonical_local_energy_weight > 0.0
    )
    has_batch_swd_loss = rho_swd_weight > 0.0 or local_swd_weight > 0.0
    if has_energy_loss or has_batch_swd_loss:
        _, clean_prediction = diffusion_model_predictions(
            noisy,
            prediction,
            timesteps,
            schedule,
            str(getattr(model.config, "prediction_type", "epsilon")),
        )
        clean_rho, clean_local = split_active_latent(clean, model.config)
        predicted_clean_rho, predicted_clean_local = split_active_latent(
            clean_prediction,
            model.config,
        )

    if has_energy_loss:
        max_timestep = int(
            np.floor(
                (int(model.config.timesteps) - 1)
                * float(spectral_energy_max_timestep_fraction)
            )
        )
        active = timesteps <= max_timestep
        zero = prediction.new_zeros(())
        rho_energy_loss = zero
        local_energy_loss = zero
        if bool(torch.any(active).item()):
            rho_energy_loss = _masked_log_spectral_energy_loss(
                predicted_clean_rho,
                clean_rho,
                active,
                has_constant=False,
            )
            local_energy_loss = _masked_log_spectral_energy_loss(
                predicted_clean_local,
                clean_local,
                active,
                has_constant=True,
            )
        canonical_rho_energy_loss = zero
        canonical_local_energy_loss = zero
        if (
            canonical_rho_energy_weight > 0.0
            or canonical_local_energy_weight > 0.0
        ):
            if (
                canonical_energy_transform is None
                or canonical_energy_config is None
            ):
                raise ValueError(
                    "canonical spectral energy requires a transform and config"
                )
            canonical_clean = canonical_energy_transform(clean)
            canonical_prediction = canonical_energy_transform(clean_prediction)
            canonical_clean_rho, canonical_clean_local = split_active_latent(
                canonical_clean, canonical_energy_config
            )
            canonical_prediction_rho, canonical_prediction_local = (
                split_active_latent(
                    canonical_prediction, canonical_energy_config
                )
            )
            if bool(torch.any(active).item()):
                canonical_rho_energy_loss = _masked_log_spectral_energy_loss(
                    canonical_prediction_rho,
                    canonical_clean_rho,
                    active,
                    has_constant=False,
                )
                canonical_local_energy_loss = _masked_log_spectral_energy_loss(
                    canonical_prediction_local,
                    canonical_clean_local,
                    active,
                    has_constant=True,
                )
        loss = (
            noise_loss
            + rho_energy_weight * rho_energy_loss
            + local_energy_weight * local_energy_loss
            + canonical_rho_energy_weight * canonical_rho_energy_loss
            + canonical_local_energy_weight * canonical_local_energy_loss
        )
        metrics.update(
            {
                "loss": loss,
                "noise_loss": noise_loss,
                "rho_log_spectral_energy_loss": rho_energy_loss,
                "local_log_spectral_energy_loss": local_energy_loss,
                "rho_spectral_energy_weight": prediction.new_tensor(
                    rho_energy_weight
                ),
                "local_spectral_energy_weight": prediction.new_tensor(
                    local_energy_weight
                ),
                "canonical_rho_log_spectral_energy_loss": (
                    canonical_rho_energy_loss
                ),
                "canonical_local_log_spectral_energy_loss": (
                    canonical_local_energy_loss
                ),
                "canonical_rho_spectral_energy_weight": prediction.new_tensor(
                    canonical_rho_energy_weight
                ),
                "canonical_local_spectral_energy_weight": prediction.new_tensor(
                    canonical_local_energy_weight
                ),
                "spectral_energy_active_fraction": active.to(
                    dtype=parameter.dtype
                ).mean(),
            }
        )
    if has_batch_swd_loss:
        max_swd_timestep = int(
            np.floor(
                (int(model.config.timesteps) - 1)
                * float(batch_swd_max_timestep_fraction)
            )
        )
        swd_active = timesteps <= max_swd_timestep
        zero = prediction.new_zeros(())
        rho_swd_loss = zero
        local_swd_loss = zero
        if int(torch.count_nonzero(swd_active).item()) >= 2:
            rho_swd_loss = _masked_batch_sliced_wasserstein_loss(
                predicted_clean_rho,
                clean_rho,
                swd_active,
                projection_count=int(batch_swd_projections),
                projection_seed=int(batch_swd_projection_seed),
            )
            local_swd_loss = _masked_batch_sliced_wasserstein_loss(
                predicted_clean_local,
                clean_local,
                swd_active,
                projection_count=int(batch_swd_projections),
                projection_seed=int(batch_swd_projection_seed) + 1,
            )
        loss = (
            loss
            + rho_swd_weight * rho_swd_loss
            + local_swd_weight * local_swd_loss
        )
        metrics.update(
            {
                "loss": loss,
                "rho_batch_swd_loss": rho_swd_loss,
                "local_batch_swd_loss": local_swd_loss,
                "rho_batch_swd_weight": prediction.new_tensor(rho_swd_weight),
                "local_batch_swd_weight": prediction.new_tensor(
                    local_swd_weight
                ),
                "batch_swd_active_fraction": swd_active.to(
                    dtype=parameter.dtype
                ).mean(),
                "batch_swd_active_count": swd_active.to(
                    dtype=parameter.dtype
                ).sum(),
            }
        )
    return loss, metrics


def _masked_log_spectral_energy_loss(
    predicted,
    target,
    active,
    *,
    has_constant: bool,
):
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("spectral energy tensors must have matching batch shapes")
    if active.ndim != 1 or int(active.shape[0]) != int(predicted.shape[0]):
        raise ValueError("spectral energy mask must match the batch")
    feature_count = int(predicted.shape[-1])
    paired_count = feature_count - int(has_constant)
    if paired_count < 0 or paired_count % 2:
        raise ValueError("spectral features do not form conjugate pairs")

    def grouped_energy(values):
        parts = []
        if has_constant:
            parts.append(values[..., :1].square())
            values = values[..., 1:]
        if int(values.shape[-1]):
            parts.append(
                values.reshape(*values.shape[:-1], -1, 2)
                .square()
                .mean(dim=-1)
            )
        return parts[0] if len(parts) == 1 else _torch().cat(parts, dim=-1)

    predicted_energy = grouped_energy(predicted)
    target_energy = grouped_energy(target)
    error = (
        predicted_energy.log1p() - target_energy.log1p()
    ).square()
    return error[active].mean()


def _masked_batch_sliced_wasserstein_loss(
    predicted,
    target,
    active,
    *,
    projection_count: int,
    projection_seed: int,
):
    """Match a predicted-clean minibatch to its target marginal by seeded slices."""

    torch = _torch()
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("batch SWD tensors must have matching batch shapes")
    if active.ndim != 1 or int(active.shape[0]) != int(predicted.shape[0]):
        raise ValueError("batch SWD mask must match the batch")
    if int(projection_count) < 1:
        raise ValueError("batch SWD projection count must be positive")
    active_count = int(torch.count_nonzero(active).item())
    if active_count < 2:
        return predicted.new_zeros(())

    predicted_flat = predicted[active].reshape(active_count, -1)
    target_flat = target[active].reshape(active_count, -1)
    feature_count = int(predicted_flat.shape[1])
    generator = torch.Generator(device=predicted.device)
    generator.manual_seed(int(projection_seed))
    directions = torch.randn(
        (feature_count, int(projection_count)),
        dtype=predicted.dtype,
        device=predicted.device,
        generator=generator,
    )
    directions = directions / directions.square().sum(dim=0).sqrt().clamp_min(
        torch.finfo(predicted.dtype).eps
    )
    predicted_slices = torch.sort(predicted_flat @ directions, dim=0).values
    target_slices = torch.sort(target_flat @ directions, dim=0).values
    return torch.mean((predicted_slices - target_slices).square())


def sample_structured_latents(
    model,
    conditions,
    *,
    initial_noise=None,
    sample_steps: int | None = None,
    start_timestep: int | None = None,
    eta: float = 0.0,
    guidance=None,
):
    return sample_continuous_latents(
        model,
        conditions,
        initial_noise=initial_noise,
        sample_steps=sample_steps,
        start_timestep=start_timestep,
        eta=eta,
        guidance=guidance,
    )


def save_structured_diffusion_checkpoint(
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
            "schema": "vpfp_structured_direct_diffusion_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "active_latent_normalization": active_latent_normalization,
            "condition_normalization": condition_normalization,
            "decoder_checkpoint": str(Path(decoder_checkpoint).resolve()),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_structured_diffusion_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    torch = _torch()
    payload = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if payload.get("schema") != (
        "vpfp_structured_direct_diffusion_checkpoint_v1"
    ):
        raise ValueError("unexpected structured VPFP diffusion checkpoint schema")
    model = VPFPStructuredDenoiser.build(
        VPFPStructuredDiffusionConfig.from_dict(payload["model_config"])
    ).to(device=device)
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _rho_max_mode(config: VPFPStructuredDiffusionConfig) -> int:
    side_squared = 2 * int(config.rho_mode_count) + 1
    side = int(round(np.sqrt(float(side_squared))))
    if side * side != side_squared or side % 2 != 1:
        raise ValueError(
            "periodic-grid architecture requires a square rho Fourier layout"
        )
    return (side - 1) // 2


def _physical_fourier_basis(
    torch,
    *,
    max_mode: int,
    grid_size: int,
    include_constant: bool,
):
    if int(grid_size) < 2 * int(max_mode) + 1:
        raise ValueError("physical Fourier grid aliases represented modes")
    coordinate = torch.arange(int(grid_size), dtype=torch.float64) / float(
        grid_size
    )
    x, y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    columns = []
    if include_constant:
        columns.append(torch.ones_like(x).reshape(-1))
    for kx, ky in _real_fourier_half_modes(max_mode, max_mode):
        phase = 2.0 * torch.pi * (float(kx) * x + float(ky) * y)
        columns.extend(
            (
                np.sqrt(2.0) * torch.cos(phase).reshape(-1),
                np.sqrt(2.0) * torch.sin(phase).reshape(-1),
            )
        )
    basis = torch.stack(columns, dim=-1) / float(grid_size)
    gram = basis.T @ basis
    identity = torch.eye(int(gram.shape[0]), dtype=gram.dtype)
    if not bool(torch.allclose(gram, identity, atol=1.0e-10, rtol=1.0e-10)):
        raise ValueError("physical Fourier basis is not orthonormal")
    return basis


def _group_count(width: int) -> int:
    for groups in (16, 8, 4, 2):
        if int(width) % groups == 0:
            return groups
    return 1


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
