from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from agent_verifier.vpfp.teacher_trace import (
    COMPONENT_TOKEN_DIM,
    RHO_TOKEN_DIM,
    SCALAR_TOKEN_DIM,
    SOURCE_TOKEN_DIM,
    VPFPTeacherTraceConfig,
    validate_teacher_trace,
)


ALIGNED_TRACE_SCHEMA = "vpfp_phase_aligned_teacher_trace_v1"
CHECKPOINT_SCHEMA = "vpfp_teacher_trace_conditional_residual_flow_checkpoint_v1"


@dataclass(frozen=True)
class VPFPTeacherTraceConditionalFlowConfig:
    trace_config: dict[str, Any]
    grid_size: int = 13
    macro_latent_channels: int = 48
    micro_latent_channels: int = 144
    global_latent_dim: int = 128
    condition_width: int = 64
    hidden_width: int = 256
    condition_depth: int = 2
    mean_depth: int = 5
    flow_depth: int = 6
    time_embedding_dim: int = 128
    dropout: float = 0.0
    representation: str = "vpfp_phase_aligned_conditional_residual_flow_v1"

    @property
    def local_latent_channels(self) -> int:
        return int(self.macro_latent_channels + self.micro_latent_channels)

    def validate(self) -> None:
        VPFPTeacherTraceConfig.from_dict(self.trace_config)
        dimensions = (
            self.grid_size,
            self.macro_latent_channels,
            self.micro_latent_channels,
            self.global_latent_dim,
            self.condition_width,
            self.hidden_width,
            self.condition_depth,
            self.mean_depth,
            self.flow_depth,
            self.time_embedding_dim,
        )
        if min(int(value) for value in dimensions) < 1:
            raise ValueError("conditional flow dimensions must be positive")
        if int(self.hidden_width) < int(self.condition_width):
            raise ValueError("hidden width must cover the condition width")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("conditional flow dropout must lie in [0,1)")
        if self.representation != (
            "vpfp_phase_aligned_conditional_residual_flow_v1"
        ):
            raise ValueError("unexpected conditional flow representation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPTeacherTraceConditionalFlowConfig":
        config = cls(**dict(payload))
        config.validate()
        return config


def build_phase_aligned_teacher_trace(
    trace: Mapping[str, Any],
    config: VPFPTeacherTraceConfig,
    *,
    grid_size: int,
) -> dict[str, Any]:
    """Rasterize only the spatial phase carried by a raw teacher trace."""

    validate_teacher_trace(trace, config)
    side = int(grid_size)
    if side < 1:
        raise ValueError("phase-aligned trace grid must be positive")
    rho = torch.as_tensor(trace["rho"])
    dtype = rho.dtype
    device = rho.device
    coordinates = 2.0 * math.pi * torch.arange(
        side, dtype=dtype, device=device
    ) / float(side)
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")

    rho_modes = rho
    rho_phase = _phase_cosine(
        rho_modes,
        x=x,
        y=y,
        maximum_mode=int(config.maximum_spatial_mode),
    )
    rho_mask = torch.as_tensor(
        trace["rho_mask"], dtype=dtype, device=device
    )
    rho_field = torch.sum(
        rho_modes[..., 0, None, None]
        * rho_phase
        * rho_mask[..., None, None],
        dim=1,
        keepdim=True,
    )

    components = torch.as_tensor(
        trace["components"], dtype=dtype, device=device
    )
    component_mask = torch.as_tensor(
        trace["component_mask"], dtype=dtype, device=device
    )
    mode_layout = (
        (6, int(config.component_weight_mode_count)),
        (
            6 + 5 * int(config.component_weight_mode_count),
            int(config.component_parameter_mode_count),
        ),
        (
            6
            + 5 * int(config.component_weight_mode_count)
            + 5 * int(config.component_parameter_mode_count),
            int(config.component_parameter_mode_count),
        ),
        (
            6
            + 5 * int(config.component_weight_mode_count)
            + 10 * int(config.component_parameter_mode_count),
            int(config.component_parameter_mode_count),
        ),
        (
            6
            + 5 * int(config.component_weight_mode_count)
            + 15 * int(config.component_parameter_mode_count),
            int(config.component_parameter_mode_count),
        ),
    )
    spatial_parameters = [
        _component_mode_field(
            components,
            offset=offset,
            count=count,
            x=x,
            y=y,
            maximum_mode=int(config.maximum_spatial_mode),
        )
        for offset, count in mode_layout
    ]
    center_scale = float(config.velocity_center_range)
    base = components[..., :6]
    component_fields = torch.stack(
        (
            base[..., 0, None, None] + spatial_parameters[0],
            center_scale * base[..., 1, None, None]
            + spatial_parameters[1],
            center_scale * base[..., 2, None, None]
            + spatial_parameters[2],
            base[..., 3, None, None] + spatial_parameters[3],
            base[..., 4, None, None] + spatial_parameters[4],
            base[..., 5, None, None].expand(-1, -1, side, side),
            component_mask[..., None, None].expand(-1, -1, side, side),
        ),
        dim=2,
    )
    component_fields = component_fields * component_mask[
        ..., None, None, None
    ]

    sources = torch.as_tensor(
        trace["sources"], dtype=dtype, device=device
    )
    source_mask = torch.as_tensor(
        trace["source_mask"], dtype=dtype, device=device
    )
    maximum_mode = float(config.maximum_spatial_mode)
    kx = torch.round(sources[..., 1] * maximum_mode)
    ky = torch.round(sources[..., 2] * maximum_mode)
    angle = kx[..., None, None] * x + ky[..., None, None] * y
    sine = (
        torch.sin(angle) * sources[..., 5, None, None]
        + torch.cos(angle) * sources[..., 4, None, None]
    )
    cosine = (
        torch.cos(angle) * sources[..., 5, None, None]
        - torch.sin(angle) * sources[..., 4, None, None]
    )
    source_fields = torch.stack(
        (
            sources[..., 0, None, None] * sine,
            sources[..., 0, None, None] * cosine,
        ),
        dim=2,
    )
    source_fields = source_fields * source_mask[..., None, None, None]
    result = {
        "schema": ALIGNED_TRACE_SCHEMA,
        "config": config.to_dict(),
        "grid_size": side,
        "rho": rho_field,
        "components": component_fields,
        "sources": source_fields,
    }
    validate_phase_aligned_teacher_trace(result, config, grid_size=side)
    return result


def validate_phase_aligned_teacher_trace(
    aligned: Mapping[str, Any],
    config: VPFPTeacherTraceConfig,
    *,
    grid_size: int,
) -> None:
    if aligned.get("schema") != ALIGNED_TRACE_SCHEMA:
        raise ValueError("unexpected phase-aligned trace schema")
    if dict(aligned.get("config") or {}) != config.to_dict():
        raise ValueError("phase-aligned trace config changed")
    if int(aligned.get("grid_size", -1)) != int(grid_size):
        raise ValueError("phase-aligned trace grid changed")
    rho = torch.as_tensor(aligned["rho"])
    batch = int(rho.shape[0])
    side = int(grid_size)
    expected = {
        "rho": (batch, 1, side, side),
        "components": (
            batch,
            int(config.maximum_component_count),
            7,
            side,
            side,
        ),
        "sources": (
            batch,
            int(config.source_mode_count),
            2,
            side,
            side,
        ),
    }
    for name, shape in expected.items():
        values = torch.as_tensor(aligned[name])
        if tuple(values.shape) != shape:
            raise ValueError(f"phase-aligned {name} shape changed")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError(f"phase-aligned {name} is non-finite")


def index_phase_aligned_teacher_trace(
    aligned: Mapping[str, Any], indices
) -> dict[str, Any]:
    return {
        "schema": aligned["schema"],
        "config": dict(aligned["config"]),
        "grid_size": int(aligned["grid_size"]),
        "rho": torch.as_tensor(aligned["rho"])[indices],
        "components": torch.as_tensor(aligned["components"])[indices],
        "sources": torch.as_tensor(aligned["sources"])[indices],
    }


def move_phase_aligned_teacher_trace(
    aligned: Mapping[str, Any], *, device: str | torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in aligned.items()
    }


class VPFPTeacherTraceConditionalResidualFlow(nn.Module):
    def __init__(self, config: VPFPTeacherTraceConditionalFlowConfig):
        super().__init__()
        config.validate()
        self.config = config
        trace_config = VPFPTeacherTraceConfig.from_dict(config.trace_config)
        condition_width = int(config.condition_width)
        hidden = int(config.hidden_width)
        local_channels = int(config.local_latent_channels)

        self.rho_token = _token_projection(RHO_TOKEN_DIM, condition_width)
        self.component_token = _token_projection(
            COMPONENT_TOKEN_DIM, condition_width
        )
        self.source_token = _token_projection(
            SOURCE_TOKEN_DIM, condition_width
        )
        self.scalar_token = _token_projection(
            SCALAR_TOKEN_DIM, condition_width
        )
        self.rho_field = nn.Conv2d(1, condition_width, kernel_size=1)
        self.component_field = nn.Conv2d(
            7, condition_width, kernel_size=1
        )
        self.source_field = nn.Conv2d(2, condition_width, kernel_size=1)
        self.local_condition = nn.Conv2d(
            4 * condition_width, hidden, kernel_size=1
        )
        self.condition_blocks = nn.ModuleList(
            _PeriodicResidualBlock(hidden)
            for _ in range(int(config.condition_depth))
        )
        self.global_condition = nn.Sequential(
            nn.Linear(5 * condition_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.mean_blocks = nn.ModuleList(
            _PeriodicFiLMBlock(hidden, hidden, dropout=float(config.dropout))
            for _ in range(int(config.mean_depth))
        )
        self.mean_local = nn.Conv2d(hidden, local_channels, kernel_size=1)
        self.mean_global = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(config.global_latent_dim)),
        )

        self.flow_input = nn.Conv2d(
            local_channels + hidden, hidden, kernel_size=1
        )
        self.flow_global = nn.Linear(
            int(config.global_latent_dim), hidden
        )
        self.flow_time = nn.Sequential(
            nn.Linear(int(config.time_embedding_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow_blocks = nn.ModuleList(
            _PeriodicFiLMBlock(hidden, hidden, dropout=float(config.dropout))
            for _ in range(int(config.flow_depth))
        )
        self.flow_output = nn.Conv2d(hidden, local_channels, kernel_size=1)
        nn.init.zeros_(self.flow_output.weight)
        nn.init.zeros_(self.flow_output.bias)
        self.register_buffer(
            "residual_scale",
            torch.ones(1, local_channels, 1, 1),
            persistent=True,
        )
        self.trace_config = trace_config

    @classmethod
    def build(
        cls, config: VPFPTeacherTraceConditionalFlowConfig | Mapping[str, Any]
    ) -> "VPFPTeacherTraceConditionalResidualFlow":
        parsed = (
            config
            if isinstance(config, VPFPTeacherTraceConditionalFlowConfig)
            else VPFPTeacherTraceConditionalFlowConfig.from_dict(config)
        )
        return cls(parsed)

    def encode_condition(
        self,
        trace: Mapping[str, Any],
        aligned: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_teacher_trace(trace, self.trace_config)
        validate_phase_aligned_teacher_trace(
            aligned, self.trace_config, grid_size=int(self.config.grid_size)
        )
        parameter = next(self.parameters())

        def values(name: str) -> torch.Tensor:
            return torch.as_tensor(
                trace[name], dtype=parameter.dtype, device=parameter.device
            )

        def fields(name: str) -> torch.Tensor:
            return torch.as_tensor(
                aligned[name], dtype=parameter.dtype, device=parameter.device
            )

        rho_token = self.rho_token(values("rho"))
        component_token = self.component_token(values("components"))
        source_token = self.source_token(values("sources"))
        scalar_token = self.scalar_token(values("scalars"))
        rho_mask = torch.as_tensor(
            trace["rho_mask"], dtype=torch.bool, device=parameter.device
        )
        component_mask = torch.as_tensor(
            trace["component_mask"],
            dtype=torch.bool,
            device=parameter.device,
        )
        source_mask = torch.as_tensor(
            trace["source_mask"], dtype=torch.bool, device=parameter.device
        )
        side = int(self.config.grid_size)

        rho_local = self.rho_field(fields("rho"))
        rho_local = rho_local + _masked_mean(rho_token, rho_mask)[
            :, :, None, None
        ]
        component_local = self._set_field_embedding(
            fields("components"),
            component_token,
            component_mask,
            self.component_field,
        )
        source_local = self._set_field_embedding(
            fields("sources"),
            source_token,
            source_mask,
            self.source_field,
        )
        scalar_local = scalar_token[:, :, None, None].expand(
            -1, -1, side, side
        )
        local = self.local_condition(
            torch.cat(
                (rho_local, component_local, source_local, scalar_local),
                dim=1,
            )
        )
        for block in self.condition_blocks:
            local = block(local)
        pooled_local = local.mean(dim=(2, 3))
        global_context = self.global_condition(
            torch.cat(
                (
                    _masked_mean(rho_token, rho_mask),
                    _masked_mean(component_token, component_mask),
                    _masked_mean(source_token, source_mask),
                    scalar_token,
                    pooled_local[:, : int(self.config.condition_width)],
                ),
                dim=1,
            )
        )
        return local, global_context

    def _set_field_embedding(
        self,
        field: torch.Tensor,
        token: torch.Tensor,
        mask: torch.Tensor,
        projection: nn.Conv2d,
    ) -> torch.Tensor:
        batch, count, channels, side, _ = field.shape
        projected = projection(field.reshape(batch * count, channels, side, side))
        projected = projected.reshape(
            batch, count, int(self.config.condition_width), side, side
        )
        projected = torch.nn.functional.silu(
            projected + token[..., None, None]
        )
        weights = mask.to(projected.dtype)[..., None, None, None]
        denominator = mask.sum(dim=1).clamp_min(1).to(projected.dtype).sqrt()
        return (projected * weights).sum(dim=1) / denominator[:, None, None, None]

    def forward_mean(
        self,
        trace: Mapping[str, Any],
        aligned: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        condition, context = self.encode_condition(trace, aligned)
        hidden = condition
        for block in self.mean_blocks:
            hidden = block(hidden, context)
        local = _zero_spatial_mean(self.mean_local(hidden))
        global_code = self.mean_global(context)
        return {
            "local": local,
            "global": global_code,
            "condition": condition,
            "context": context,
        }

    def flow_velocity(
        self,
        noisy_residual: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: torch.Tensor,
        context: torch.Tensor,
        predicted_global: torch.Tensor,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        noisy = torch.as_tensor(
            noisy_residual, dtype=parameter.dtype, device=parameter.device
        )
        batch = int(noisy.shape[0])
        expected = (
            batch,
            int(self.config.local_latent_channels),
            int(self.config.grid_size),
            int(self.config.grid_size),
        )
        if tuple(noisy.shape) != expected:
            raise ValueError("conditional flow residual shape changed")
        time_values = torch.as_tensor(
            time, dtype=noisy.dtype, device=noisy.device
        ).reshape(-1)
        if tuple(time_values.shape) != (batch,):
            raise ValueError("conditional flow time shape changed")
        if bool(torch.any((time_values < 0.0) | (time_values > 1.0)).item()):
            raise ValueError("conditional flow time must lie in [0,1]")
        flow_context = (
            context
            + self.flow_global(predicted_global)
            + self.flow_time(
                _sinusoidal_embedding(
                    time_values, int(self.config.time_embedding_dim)
                )
            )
        )
        hidden = self.flow_input(torch.cat((noisy, condition), dim=1))
        for block in self.flow_blocks:
            hidden = block(hidden, flow_context)
        return _zero_spatial_mean(self.flow_output(hidden))

    def set_residual_scale(self, scale: torch.Tensor) -> None:
        value = torch.as_tensor(
            scale,
            dtype=self.residual_scale.dtype,
            device=self.residual_scale.device,
        )
        if tuple(value.shape) != tuple(self.residual_scale.shape):
            raise ValueError("conditional flow residual scale shape changed")
        if not bool(torch.all(torch.isfinite(value) & (value > 0.0)).item()):
            raise ValueError("conditional flow residual scale must be positive")
        self.residual_scale.copy_(value)

    def mean_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("flow_"):
                yield parameter

    def flow_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("flow_"):
                yield parameter


def teacher_trace_deterministic_loss(
    model: VPFPTeacherTraceConditionalResidualFlow,
    trace: Mapping[str, Any],
    aligned: Mapping[str, Any],
    target_local,
    target_global,
    *,
    macro_fraction: float = 0.35,
    global_fraction: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model.forward_mean(trace, aligned)
    local = torch.as_tensor(
        target_local,
        dtype=prediction["local"].dtype,
        device=prediction["local"].device,
    )
    global_code = torch.as_tensor(
        target_global,
        dtype=prediction["global"].dtype,
        device=prediction["global"].device,
    )
    macro, micro = _grouped_mse(
        prediction["local"],
        local,
        boundary=int(model.config.macro_latent_channels),
    )
    global_loss = torch.mean((prediction["global"] - global_code).square())
    local_loss = float(macro_fraction) * macro + (
        1.0 - float(macro_fraction)
    ) * micro
    loss = (1.0 - float(global_fraction)) * local_loss + float(
        global_fraction
    ) * global_loss
    return loss, {
        "loss": loss.detach(),
        "macro_loss": macro.detach(),
        "micro_loss": micro.detach(),
        "global_loss": global_loss.detach(),
        "predicted_local_rms": prediction["local"].square().mean().sqrt().detach(),
        "target_local_rms": local.square().mean().sqrt().detach(),
    }


def fit_teacher_trace_residual_scale(
    target_local, predicted_local, *, floor_fraction: float = 0.02
) -> torch.Tensor:
    target = torch.as_tensor(target_local)
    prediction = torch.as_tensor(
        predicted_local, dtype=target.dtype, device=target.device
    )
    if target.shape != prediction.shape or target.ndim != 4:
        raise ValueError("residual-scale inputs must be matching local fields")
    scale = (target - prediction).square().mean(dim=(0, 2, 3)).sqrt()
    positive = scale[scale > 0.0]
    reference = (
        positive.median()
        if int(positive.numel())
        else torch.ones((), dtype=scale.dtype, device=scale.device)
    )
    floor = max(float(reference) * float(floor_fraction), 1.0e-4)
    return scale.clamp_min(floor)[None, :, None, None]


def teacher_trace_residual_flow_matching_loss(
    model: VPFPTeacherTraceConditionalResidualFlow,
    trace: Mapping[str, Any],
    aligned: Mapping[str, Any],
    target_local,
    *,
    path_times: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    macro_fraction: float = 0.35,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = next(model.parameters())
    target = torch.as_tensor(
        target_local, dtype=parameter.dtype, device=parameter.device
    )
    with torch.no_grad():
        prediction = model.forward_mean(trace, aligned)
        condition = prediction["condition"]
        context = prediction["context"]
        predicted_global = prediction["global"]
        clean = _zero_spatial_mean(
            (target - prediction["local"]) / model.residual_scale
        )
    source = (
        _zero_spatial_mean(torch.randn_like(clean))
        if noise is None
        else _zero_spatial_mean(
            torch.as_tensor(noise, dtype=clean.dtype, device=clean.device)
        )
    )
    if source.shape != clean.shape:
        raise ValueError("conditional flow noise shape changed")
    times = (
        torch.rand(clean.shape[0], dtype=clean.dtype, device=clean.device)
        if path_times is None
        else torch.as_tensor(
            path_times, dtype=clean.dtype, device=clean.device
        ).reshape(-1)
    )
    if tuple(times.shape) != (int(clean.shape[0]),):
        raise ValueError("conditional flow path-time shape changed")
    interpolation = (
        (1.0 - times[:, None, None, None]) * source
        + times[:, None, None, None] * clean
    )
    target_velocity = clean - source
    predicted_velocity = model.flow_velocity(
        interpolation,
        times,
        condition=condition,
        context=context,
        predicted_global=predicted_global,
    )
    macro, micro = _grouped_mse(
        predicted_velocity,
        target_velocity,
        boundary=int(model.config.macro_latent_channels),
    )
    loss = float(macro_fraction) * macro + (
        1.0 - float(macro_fraction)
    ) * micro
    return loss, {
        "loss": loss.detach(),
        "macro_velocity_mse": macro.detach(),
        "micro_velocity_mse": micro.detach(),
        "path_time_mean": times.mean().detach(),
        "normalized_residual_rms": clean.square().mean().sqrt().detach(),
    }


def sample_teacher_trace_conditional_flow(
    model: VPFPTeacherTraceConditionalResidualFlow,
    trace: Mapping[str, Any],
    aligned: Mapping[str, Any],
    *,
    sample_steps: int = 16,
    solver: str = "heun",
    initial_noise: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    if int(sample_steps) < 1:
        raise ValueError("conditional flow sample steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("conditional flow solver must be euler or heun")
    if float(temperature) < 0.0:
        raise ValueError("conditional flow temperature must be nonnegative")
    model.eval()
    with torch.no_grad():
        prediction = model.forward_mean(trace, aligned)
        condition = prediction["condition"]
        context = prediction["context"]
        predicted_global = prediction["global"]
        shape = prediction["local"].shape
        current = (
            torch.randn(shape, dtype=prediction["local"].dtype, device=prediction["local"].device)
            if initial_noise is None
            else torch.as_tensor(
                initial_noise,
                dtype=prediction["local"].dtype,
                device=prediction["local"].device,
            ).clone()
        )
        if current.shape != shape:
            raise ValueError("conditional flow initial-noise shape changed")
        current = float(temperature) * _zero_spatial_mean(current)
        step_size = 1.0 / float(sample_steps)
        batch = int(current.shape[0])
        for index in range(int(sample_steps)):
            time = torch.full(
                (batch,),
                float(index) * step_size,
                dtype=current.dtype,
                device=current.device,
            )
            velocity = model.flow_velocity(
                current,
                time,
                condition=condition,
                context=context,
                predicted_global=predicted_global,
            )
            if solver == "euler":
                current = current + step_size * velocity
            else:
                proposal = current + step_size * velocity
                next_time = torch.full_like(
                    time, float(index + 1) * step_size
                )
                next_velocity = model.flow_velocity(
                    proposal,
                    next_time,
                    condition=condition,
                    context=context,
                    predicted_global=predicted_global,
                )
                current = current + 0.5 * step_size * (
                    velocity + next_velocity
                )
            current = _zero_spatial_mean(current)
        local = _zero_spatial_mean(
            prediction["local"] + current * model.residual_scale
        )
    return {
        "local": local,
        "global": predicted_global,
        "mean_local": prediction["local"],
        "normalized_residual": current,
    }


def latent_prediction_metrics(
    predicted_local,
    predicted_global,
    target_local,
    target_global,
    *,
    macro_channels: int,
) -> dict[str, float]:
    predicted = torch.as_tensor(predicted_local).double()
    target = torch.as_tensor(target_local).double()
    predicted_global = torch.as_tensor(predicted_global).double()
    target_global = torch.as_tensor(target_global).double()

    def relative(first, second) -> float:
        numerator = torch.sum((first - second).square())
        denominator = torch.sum(second.square()).clamp_min(1.0e-12)
        return float(torch.sqrt(numerator / denominator))

    boundary = int(macro_channels)
    return {
        "local_relative_l2": relative(predicted, target),
        "macro_local_relative_l2": relative(
            predicted[:, :boundary], target[:, :boundary]
        ),
        "micro_local_relative_l2": relative(
            predicted[:, boundary:], target[:, boundary:]
        ),
        "global_relative_l2": relative(predicted_global, target_global),
        "local_rms_ratio": float(
            predicted.square().mean().sqrt()
            / target.square().mean().sqrt().clamp_min(1.0e-12)
        ),
    }


def save_teacher_trace_conditional_flow_checkpoint(
    path: str | Path,
    *,
    model: VPFPTeacherTraceConditionalResidualFlow,
    autoencoder_checkpoint: str | Path,
    autoencoder_checkpoint_sha256: str,
    trace_normalization: Mapping[str, Any],
    provenance: Mapping[str, Any],
    train_config: Mapping[str, Any],
    step: Mapping[str, int],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "autoencoder_checkpoint": str(
                Path(autoencoder_checkpoint).resolve()
            ),
            "autoencoder_checkpoint_sha256": str(
                autoencoder_checkpoint_sha256
            ),
            "trace_normalization": dict(trace_normalization),
            "provenance": dict(provenance),
            "answer_first_contract": {
                "primary_object": "continuous_hidden_solution_function",
                "training_target": "paired_frozen_autoencoder_answer_codes",
                "integer_seed_used_as_model_input": False,
                "teacher_trace_used_as_model_input": True,
                "equation_first_source_reused_as_public_source": False,
                "autoencoder_frozen": True,
                "public_source_derived_after_answer_decoding": True,
            },
            "train_config": dict(train_config),
            "step": {key: int(value) for key, value in step.items()},
        },
        destination,
    )


def load_teacher_trace_conditional_flow_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[VPFPTeacherTraceConditionalResidualFlow, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected conditional flow checkpoint schema")
    contract = dict(payload.get("answer_first_contract") or {})
    if not bool(
        contract.get("primary_object")
        == "continuous_hidden_solution_function"
        and contract.get("training_target")
        == "paired_frozen_autoencoder_answer_codes"
        and not contract.get("integer_seed_used_as_model_input")
        and contract.get("teacher_trace_used_as_model_input")
        and not contract.get("equation_first_source_reused_as_public_source")
        and contract.get("autoencoder_frozen")
        and contract.get("public_source_derived_after_answer_decoding")
    ):
        raise ValueError("conditional flow violates answer-first contract")
    model = VPFPTeacherTraceConditionalResidualFlow.build(
        payload["model_config"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _phase_cosine(features, *, x, y, maximum_mode: int) -> torch.Tensor:
    kx = torch.round(features[..., 1] * float(maximum_mode))
    ky = torch.round(features[..., 2] * float(maximum_mode))
    angle = kx[..., None, None] * x + ky[..., None, None] * y
    return (
        torch.cos(angle) * features[..., 4, None, None]
        - torch.sin(angle) * features[..., 3, None, None]
    )


def _component_mode_field(
    components,
    *,
    offset: int,
    count: int,
    x,
    y,
    maximum_mode: int,
) -> torch.Tensor:
    modes = components[..., int(offset) : int(offset) + 5 * int(count)]
    modes = modes.reshape(*components.shape[:2], int(count), 5)
    cosine = _phase_cosine(
        modes, x=x, y=y, maximum_mode=int(maximum_mode)
    )
    return torch.sum(modes[..., 0, None, None] * cosine, dim=2)


def _token_projection(input_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), int(width)),
        nn.SiLU(),
        nn.Linear(int(width), int(width)),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)[..., None]
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denominator


class _PeriodicResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = _group_count(width)
        self.norm1 = nn.GroupNorm(groups, int(width))
        self.norm2 = nn.GroupNorm(groups, int(width))
        self.conv1 = nn.Conv2d(
            int(width), int(width), 3, padding=1, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            int(width), int(width), 3, padding=1, padding_mode="circular"
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(values)))
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return values + hidden


class _PeriodicFiLMBlock(nn.Module):
    def __init__(self, width: int, context_dim: int, *, dropout: float):
        super().__init__()
        groups = _group_count(width)
        self.norm1 = nn.GroupNorm(groups, int(width), affine=False)
        self.norm2 = nn.GroupNorm(groups, int(width))
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(int(context_dim), 2 * int(width))
        )
        self.conv1 = nn.Conv2d(
            int(width), int(width), 3, padding=1, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            int(width), int(width), 3, padding=1, padding_mode="circular"
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self, values: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.modulation(context).chunk(2, dim=1)
        hidden = self.norm1(values)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[
            :, :, None, None
        ]
        hidden = self.conv1(torch.nn.functional.silu(hidden))
        hidden = self.conv2(
            self.dropout(torch.nn.functional.silu(self.norm2(hidden)))
        )
        return values + hidden


def _sinusoidal_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    half = max(1, int(width) // 2)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=time.dtype, device=time.device)
        / float(max(1, half - 1))
    )
    angle = time[:, None] * frequencies[None] * 1000.0
    result = torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
    if int(result.shape[1]) < int(width):
        result = torch.nn.functional.pad(
            result, (0, int(width) - int(result.shape[1]))
        )
    return result[:, : int(width)]


def _grouped_mse(
    prediction: torch.Tensor, target: torch.Tensor, *, boundary: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("grouped MSE inputs have different shapes")
    if not 0 < int(boundary) < int(prediction.shape[1]):
        raise ValueError("grouped MSE boundary is invalid")
    return (
        torch.mean((prediction[:, :boundary] - target[:, :boundary]).square()),
        torch.mean((prediction[:, boundary:] - target[:, boundary:]).square()),
    )


def _zero_spatial_mean(values: torch.Tensor) -> torch.Tensor:
    return values - values.mean(dim=(2, 3), keepdim=True)


def _group_count(width: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if int(width) % groups == 0:
            return groups
    return 1

def teacher_trace_residual_edm_loss(
    model: VPFPTeacherTraceConditionalResidualFlow,
    trace: Mapping[str, Any],
    aligned: Mapping[str, Any],
    target_local,
    *,
    sigmas: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    sigma_min: float = 0.002,
    sigma_max: float = 10.0,
    sigma_data: float = 1.0,
    macro_fraction: float = 0.35,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """EDM denoising loss for the normalized conditional answer residual."""

    _validate_edm_hyperparameters(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
    )
    parameter = next(model.parameters())
    target = torch.as_tensor(
        target_local, dtype=parameter.dtype, device=parameter.device
    )
    with torch.no_grad():
        prediction = model.forward_mean(trace, aligned)
        condition = prediction["condition"]
        context = prediction["context"]
        predicted_global = prediction["global"]
        clean = _zero_spatial_mean(
            (target - prediction["local"]) / model.residual_scale
        )
    batch = int(clean.shape[0])
    if sigmas is None:
        log_sigma = torch.empty(
            batch, dtype=clean.dtype, device=clean.device
        ).uniform_(math.log(float(sigma_min)), math.log(float(sigma_max)))
        sigma = torch.exp(log_sigma)
    else:
        sigma = torch.as_tensor(
            sigmas, dtype=clean.dtype, device=clean.device
        ).reshape(-1)
    if tuple(sigma.shape) != (batch,) or not bool(
        torch.all(
            torch.isfinite(sigma)
            & (sigma >= float(sigma_min))
            & (sigma <= float(sigma_max))
        ).item()
    ):
        raise ValueError("conditional EDM sigma batch is invalid")
    random_noise = (
        _zero_spatial_mean(torch.randn_like(clean))
        if noise is None
        else _zero_spatial_mean(
            torch.as_tensor(noise, dtype=clean.dtype, device=clean.device)
        )
    )
    if random_noise.shape != clean.shape:
        raise ValueError("conditional EDM noise shape changed")
    sigma_field = sigma[:, None, None, None]
    noisy = _zero_spatial_mean(clean + sigma_field * random_noise)
    denoised = _conditional_residual_edm_denoise(
        model,
        noisy,
        sigma,
        condition=condition,
        context=context,
        predicted_global=predicted_global,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        sigma_data=float(sigma_data),
    )
    weight = (
        (sigma.square() + float(sigma_data) ** 2)
        / (sigma * float(sigma_data)).square()
    )[:, None, None, None]
    squared_error = weight * (denoised - clean).square()
    boundary = int(model.config.macro_latent_channels)
    macro = squared_error[:, :boundary].mean()
    micro = squared_error[:, boundary:].mean()
    loss = float(macro_fraction) * macro + (
        1.0 - float(macro_fraction)
    ) * micro
    return loss, {
        "loss": loss.detach(),
        "macro_denoising_mse": macro.detach(),
        "micro_denoising_mse": micro.detach(),
        "sigma_mean": sigma.mean().detach(),
        "sigma_minimum": sigma.min().detach(),
        "sigma_maximum": sigma.max().detach(),
        "normalized_residual_rms": clean.square().mean().sqrt().detach(),
    }


def sample_teacher_trace_conditional_edm(
    model: VPFPTeacherTraceConditionalResidualFlow,
    trace: Mapping[str, Any],
    aligned: Mapping[str, Any],
    *,
    sample_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 10.0,
    sigma_data: float = 1.0,
    rho: float = 7.0,
    initial_noise: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Sample a conditional residual with the deterministic EDM ODE."""

    _validate_edm_hyperparameters(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
    )
    if int(sample_steps) < 2:
        raise ValueError("conditional EDM sampler needs at least two steps")
    if float(rho) <= 0.0:
        raise ValueError("conditional EDM rho must be positive")
    if float(temperature) < 0.0:
        raise ValueError("conditional EDM temperature must be nonnegative")
    model.eval()
    with torch.no_grad():
        prediction = model.forward_mean(trace, aligned)
        condition = prediction["condition"]
        context = prediction["context"]
        predicted_global = prediction["global"]
        shape = prediction["local"].shape
        random_noise = (
            torch.randn(
                shape,
                dtype=prediction["local"].dtype,
                device=prediction["local"].device,
            )
            if initial_noise is None
            else torch.as_tensor(
                initial_noise,
                dtype=prediction["local"].dtype,
                device=prediction["local"].device,
            ).clone()
        )
        if random_noise.shape != shape:
            raise ValueError("conditional EDM initial-noise shape changed")
        current = (
            float(temperature)
            * float(sigma_max)
            * _zero_spatial_mean(random_noise)
        )
        ramp = torch.linspace(
            0.0,
            1.0,
            int(sample_steps),
            dtype=current.dtype,
            device=current.device,
        )
        inverse_rho = 1.0 / float(rho)
        sigmas = (
            float(sigma_max) ** inverse_rho
            + ramp
            * (
                float(sigma_min) ** inverse_rho
                - float(sigma_max) ** inverse_rho
            )
        ).pow(float(rho))
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        batch = int(current.shape[0])
        for index in range(int(sample_steps)):
            sigma = sigmas[index]
            next_sigma = sigmas[index + 1]
            sigma_batch = sigma.expand(batch)
            denoised = _conditional_residual_edm_denoise(
                model,
                current,
                sigma_batch,
                condition=condition,
                context=context,
                predicted_global=predicted_global,
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
                sigma_data=float(sigma_data),
            )
            derivative = (current - denoised) / sigma
            proposal = current + (next_sigma - sigma) * derivative
            if index < int(sample_steps) - 1:
                next_batch = next_sigma.expand(batch)
                next_denoised = _conditional_residual_edm_denoise(
                    model,
                    proposal,
                    next_batch,
                    condition=condition,
                    context=context,
                    predicted_global=predicted_global,
                    sigma_min=float(sigma_min),
                    sigma_max=float(sigma_max),
                    sigma_data=float(sigma_data),
                )
                next_derivative = (proposal - next_denoised) / next_sigma
                current = current + 0.5 * (next_sigma - sigma) * (
                    derivative + next_derivative
                )
            else:
                current = proposal
            current = _zero_spatial_mean(current)
        local = _zero_spatial_mean(
            prediction["local"] + current * model.residual_scale
        )
    return {
        "local": local,
        "global": predicted_global,
        "mean_local": prediction["local"],
        "normalized_residual": current,
    }


def _conditional_residual_edm_denoise(
    model: VPFPTeacherTraceConditionalResidualFlow,
    noisy_residual: torch.Tensor,
    sigma: torch.Tensor,
    *,
    condition: torch.Tensor,
    context: torch.Tensor,
    predicted_global: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    sigma_data: float,
) -> torch.Tensor:
    noisy = torch.as_tensor(noisy_residual)
    sigma_values = torch.as_tensor(
        sigma, dtype=noisy.dtype, device=noisy.device
    ).reshape(-1)
    if tuple(sigma_values.shape) != (int(noisy.shape[0]),):
        raise ValueError("conditional EDM denoiser sigma shape changed")
    sigma_field = sigma_values[:, None, None, None]
    data_square = float(sigma_data) ** 2
    denominator = sigma_field.square() + data_square
    c_skip = data_square / denominator
    c_out = sigma_field * float(sigma_data) / torch.sqrt(denominator)
    c_in = torch.rsqrt(denominator)
    log_range = math.log(float(sigma_max)) - math.log(float(sigma_min))
    noise_level = (
        torch.log(sigma_values) - math.log(float(sigma_min))
    ) / log_range
    network = model.flow_velocity(
        c_in * noisy,
        noise_level.clamp(0.0, 1.0),
        condition=condition,
        context=context,
        predicted_global=predicted_global,
    )
    return _zero_spatial_mean(c_skip * noisy + c_out * network)


def _validate_edm_hyperparameters(
    *, sigma_min: float, sigma_max: float, sigma_data: float
) -> None:
    if not (
        0.0 < float(sigma_min) < float(sigma_max)
        and float(sigma_data) > 0.0
    ):
        raise ValueError("conditional EDM sigma parameters are invalid")
