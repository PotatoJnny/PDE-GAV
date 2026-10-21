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


CHECKPOINT_SCHEMA = "vpfp_teacher_trace_answer_student_checkpoint_v1"


@dataclass(frozen=True)
class VPFPTeacherTraceStudentConfig:
    trace_config: dict[str, Any]
    grid_size: int = 13
    macro_latent_channels: int = 48
    micro_latent_channels: int = 144
    global_latent_dim: int = 128
    model_width: int = 256
    encoder_depth: int = 4
    decoder_depth: int = 2
    attention_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    representation: str = "vpfp_teacher_trace_answer_student_v1"

    @property
    def local_latent_channels(self) -> int:
        return int(self.macro_latent_channels + self.micro_latent_channels)

    def validate(self) -> None:
        VPFPTeacherTraceConfig.from_dict(self.trace_config)
        if min(
            int(self.grid_size),
            int(self.macro_latent_channels),
            int(self.micro_latent_channels),
            int(self.global_latent_dim),
            int(self.model_width),
            int(self.encoder_depth),
            int(self.decoder_depth),
            int(self.attention_heads),
        ) < 1:
            raise ValueError("teacher trace student dimensions must be positive")
        if int(self.model_width) % int(self.attention_heads):
            raise ValueError("teacher trace width must divide attention heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("teacher trace MLP ratio must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("teacher trace dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPTeacherTraceStudentConfig":
        result = cls(**dict(payload))
        result.validate()
        return result


class VPFPTeacherTraceAnswerStudent(nn.Module):
    def __init__(self, config: VPFPTeacherTraceStudentConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = int(config.model_width)
        self.rho_projection = _token_projection(RHO_TOKEN_DIM, width)
        self.component_projection = _token_projection(
            COMPONENT_TOKEN_DIM, width
        )
        self.source_projection = _token_projection(SOURCE_TOKEN_DIM, width)
        self.scalar_projection = _token_projection(SCALAR_TOKEN_DIM, width)
        self.type_embedding = nn.Parameter(torch.zeros(4, width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(config.attention_heads),
            dim_feedforward=int(round(width * float(config.mlp_ratio))),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config.encoder_depth)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=int(config.attention_heads),
            dim_feedforward=int(round(width * float(config.mlp_ratio))),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(config.decoder_depth)
        )
        query = _periodic_query_features(
            int(config.grid_size), width, dtype=torch.float32
        )
        self.register_buffer("periodic_query", query, persistent=True)
        self.query_bias = nn.Parameter(torch.zeros(1, query.shape[0], width))
        self.local_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, int(config.local_latent_channels)),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, int(config.global_latent_dim)),
        )
        self.reset_parameters()

    @classmethod
    def build(
        cls, config: VPFPTeacherTraceStudentConfig | Mapping[str, Any]
    ) -> "VPFPTeacherTraceAnswerStudent":
        resolved = (
            config
            if isinstance(config, VPFPTeacherTraceStudentConfig)
            else VPFPTeacherTraceStudentConfig.from_dict(config)
        )
        return cls(resolved)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.query_bias, std=0.02)

    def forward(self, trace: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        trace_config = VPFPTeacherTraceConfig.from_dict(
            self.config.trace_config
        )
        validate_teacher_trace(trace, trace_config)
        parameter = next(self.parameters())

        def values(name: str) -> torch.Tensor:
            return torch.as_tensor(
                trace[name], dtype=parameter.dtype, device=parameter.device
            )

        rho = self.rho_projection(values("rho")) + self.type_embedding[0]
        components = (
            self.component_projection(values("components"))
            + self.type_embedding[1]
        )
        sources = (
            self.source_projection(values("sources"))
            + self.type_embedding[2]
        )
        scalar = (
            self.scalar_projection(values("scalars"))[:, None]
            + self.type_embedding[3]
        )
        batch = int(rho.shape[0])
        cls = self.cls_token.expand(batch, -1, -1)
        memory = torch.cat((cls, scalar, rho, components, sources), dim=1)
        false = torch.zeros((batch, 2), dtype=torch.bool, device=memory.device)
        padding = torch.cat(
            (
                false,
                ~torch.as_tensor(
                    trace["rho_mask"], dtype=torch.bool, device=memory.device
                ),
                ~torch.as_tensor(
                    trace["component_mask"],
                    dtype=torch.bool,
                    device=memory.device,
                ),
                ~torch.as_tensor(
                    trace["source_mask"],
                    dtype=torch.bool,
                    device=memory.device,
                ),
            ),
            dim=1,
        )
        memory = self.encoder(memory, src_key_padding_mask=padding)
        query = (
            self.periodic_query.to(dtype=parameter.dtype)[None]
            + self.query_bias
        ).expand(batch, -1, -1)
        decoded = self.decoder(
            query,
            memory,
            memory_key_padding_mask=padding,
        )
        side = int(self.config.grid_size)
        local = self.local_head(decoded).transpose(1, 2).reshape(
            batch, int(self.config.local_latent_channels), side, side
        )
        local = local - local.mean(dim=(2, 3), keepdim=True)
        global_code = self.global_head(memory[:, 0])
        if not bool(torch.all(torch.isfinite(local)).item()):
            raise FloatingPointError("teacher trace student local code is non-finite")
        if not bool(torch.all(torch.isfinite(global_code)).item()):
            raise FloatingPointError("teacher trace student global code is non-finite")
        return {"local": local, "global": global_code}


def teacher_trace_student_loss(
    prediction: Mapping[str, torch.Tensor],
    target_local,
    target_global,
    *,
    macro_channels: int,
    macro_fraction: float = 0.35,
    global_fraction: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_local = torch.as_tensor(prediction["local"])
    predicted_global = torch.as_tensor(prediction["global"])
    target_local = torch.as_tensor(
        target_local,
        dtype=predicted_local.dtype,
        device=predicted_local.device,
    )
    target_global = torch.as_tensor(
        target_global,
        dtype=predicted_global.dtype,
        device=predicted_global.device,
    )
    if tuple(predicted_local.shape) != tuple(target_local.shape):
        raise ValueError("teacher trace local target shape changed")
    if tuple(predicted_global.shape) != tuple(target_global.shape):
        raise ValueError("teacher trace global target shape changed")
    boundary = int(macro_channels)
    if not 0 < boundary < int(predicted_local.shape[1]):
        raise ValueError("teacher trace macro boundary is invalid")
    if not 0.0 < float(macro_fraction) < 1.0:
        raise ValueError("teacher trace macro fraction must lie in (0,1)")
    if not 0.0 < float(global_fraction) < 1.0:
        raise ValueError("teacher trace global fraction must lie in (0,1)")
    macro = torch.mean(
        (predicted_local[:, :boundary] - target_local[:, :boundary]).square()
    )
    micro = torch.mean(
        (predicted_local[:, boundary:] - target_local[:, boundary:]).square()
    )
    global_loss = torch.mean((predicted_global - target_global).square())
    local = float(macro_fraction) * macro + (
        1.0 - float(macro_fraction)
    ) * micro
    total = (1.0 - float(global_fraction)) * local + float(
        global_fraction
    ) * global_loss
    return total, {
        "macro_loss": macro.detach(),
        "micro_loss": micro.detach(),
        "global_loss": global_loss.detach(),
    }


def save_teacher_trace_student_checkpoint(
    path: str | Path,
    *,
    model: VPFPTeacherTraceAnswerStudent,
    autoencoder_checkpoint: str | Path,
    autoencoder_checkpoint_sha256: str,
    trace_normalization: Mapping[str, Any],
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
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "ema_state_dict": copy.deepcopy(ema_state_dict),
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
            },
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_teacher_trace_student_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> tuple[VPFPTeacherTraceAnswerStudent, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected teacher trace student checkpoint schema")
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
    ):
        raise ValueError("teacher trace student violates answer-first contract")
    model = VPFPTeacherTraceAnswerStudent.build(payload["model_config"]).to(
        device
    )
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _token_projection(input_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), int(width)),
        nn.SiLU(),
        nn.Linear(int(width), int(width)),
    )


def _periodic_query_features(
    side: int, width: int, *, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = 2.0 * math.pi * torch.arange(side, dtype=dtype) / side
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    rows = []
    frequency = 1
    while len(rows) < int(width):
        rows.extend(
            (
                torch.sin(frequency * x),
                torch.cos(frequency * x),
                torch.sin(frequency * y),
                torch.cos(frequency * y),
            )
        )
        frequency += 1
    return torch.stack(rows[: int(width)], dim=-1).reshape(side * side, width)
