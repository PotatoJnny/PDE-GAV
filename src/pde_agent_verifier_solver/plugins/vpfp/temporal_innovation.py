"""Full-dimensional, unfitted PDE-time coordinates for raw VPFP VALUE controls.

This transform does not consume full latents with slopes, whitened diffusion
states or physical f coefficients. Every original time node, density Fourier
mode and Hermite/spatial coefficient remains present. Differencing and prefix
summation are inverse in real arithmetic, not universally bitwise in FP32/64.
No scaling, timestep division, truncation, learned predictor or physical
projection is performed. Normalization and diffusion-time noise are separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import torch
from torch import nn

from .value_control_diffusion import VPFPValueEDMConfig

CONFIG_SCHEMA = "vpfp_raw_value_temporal_innovation_config_v1"
SCHEMA = "vpfp_full_value_temporal_innovation_transform_v1"
RAW_VALUE_SCHEMA = "vpfp_raw_independent_value_controls_v1"
COORDINATE_SCHEMA = "vpfp_raw_block_temporal_increment_coordinates_v1"


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class VPFPTemporalInnovationConfig:
    value_config: VPFPValueEDMConfig
    kind: str = "identity"
    blocks: str = "both"
    schema: str = CONFIG_SCHEMA

    def __post_init__(self):
        if not isinstance(self.value_config, VPFPValueEDMConfig):
            raise TypeError("explicit raw VALUE layout required, not full-state config")
        if type(self.kind) is not str or self.kind not in ("identity", "time_increments"):
            raise ValueError("kind must be identity or time_increments")
        if type(self.blocks) is not str or self.blocks not in ("rho", "kinetic", "both"):
            raise ValueError("blocks must be rho, kinetic or both")
        if self.schema != CONFIG_SCHEMA:
            raise ValueError("unknown temporal innovation config schema")
        # In particular reject non-JSON/nonfinite nested layout metadata.
        _json(self.value_config.to_dict())

    @property
    def selected_blocks(self):
        if self.kind == "identity":
            return ()
        return ("rho", "kinetic") if self.blocks == "both" else (self.blocks,)

    def to_dict(self):
        return _json({"schema": self.schema, "value_config": self.value_config.to_dict(),
            "kind": self.kind, "blocks": self.blocks})

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema", "value_config", "kind", "blocks"}:
            raise ValueError("unexpected temporal innovation config fields")
        result = cls(VPFPValueEDMConfig.from_dict(payload["value_config"]),
            kind=payload["kind"], blocks=payload["blocks"], schema=payload["schema"])
        if (json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
                != json.dumps(result.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))):
            raise ValueError("complete canonical raw VALUE configuration required")
        return result


class VPFPTemporalInnovationTransform(nn.Module):
    """Stateless encode/decode; output preserves the input dtype and device.

    rho layout [B,T,R], kinetic [B,H,T,K], flat rho then flat kinetic.
    Unselected blocks are copied exactly. encode/decode never change inputs or
    silently cast, move, smooth or clip them. .to() has no tensors to migrate:
    the actual input, not ambient default dtype or module placement, governs
    arithmetic. CPU and CUDA are supported; unmaterialized/meta tensors are not.
    """

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, VPFPTemporalInnovationConfig):
            raise TypeError("explicit VPFPTemporalInnovationConfig required")
        self.config = config

    def _validate(self, values, role, *, output=False):
        cfg = self.config.value_config
        if (not isinstance(values, torch.Tensor) or values.layout != torch.strided
                or values.dtype not in (torch.float32, torch.float64)
                or values.device.type not in ("cpu", "cuda")
                or values.ndim != 2 or values.shape[0] < 1
                or values.shape[1] != cfg.latent_dimension):
            raise ValueError(f"{role} must be nonempty dense FP32/FP64 [B,{cfg.latent_dimension}] raw VALUE coordinates on CPU/CUDA")
        if not bool(torch.isfinite(values).all()):
            if output:
                raise FloatingPointError(f"{role} overflow/nonfinite result; no clipping or precision promotion")
            raise ValueError(f"{role} must be finite")
        return values

    @staticmethod
    def _difference(values):
        return torch.cat((values[..., :1, :], values[..., 1:, :]-values[..., :-1, :]), dim=-2)

    @staticmethod
    def _prefix_sum(values):
        # Deliberately fixed left-to-right native-dtype additions. Unlike a
        # parallel cumsum, this operator's finite arithmetic order is explicit.
        current = values[..., 0, :]
        recovered = [current]
        for index in range(1, values.shape[-2]):
            current = current+values[..., index, :]
            recovered.append(current)
        return torch.stack(recovered, dim=-2)

    def _transform(self, values, inverse):
        values = self._validate(values, "innovation input" if inverse else "raw value input")
        cfg = self.config.value_config
        if self.config.kind == "identity":
            return values.clone()
        batch = values.shape[0]
        rho = values[:, :cfg.rho_dimension].reshape(batch, cfg.control_count, cfg.rho_modes)
        kinetic = values[:, cfg.rho_dimension:].reshape(batch, cfg.hermite_channels, cfg.control_count, cfg.kinetic_modes)
        operation = self._prefix_sum if inverse else self._difference
        if "rho" in self.config.selected_blocks: rho = operation(rho)
        if "kinetic" in self.config.selected_blocks: kinetic = operation(kinetic)
        result = torch.cat((rho.flatten(1), kinetic.flatten(1)), dim=1)
        return self._validate(result, "decoded raw values" if inverse else "encoded innovations", output=True)

    def encode(self, raw_values):
        return self._transform(raw_values, inverse=False)

    def decode(self, coordinates):
        return self._transform(coordinates, inverse=True)

    def forward(self, raw_values):
        return self.encode(raw_values)

    def to_payload(self):
        cfg = self.config.value_config
        selected = self.config.selected_blocks
        return {"schema": SCHEMA, "config": self.config.to_dict(),
            "input_schema": RAW_VALUE_SCHEMA,
            "output_schema": COORDINATE_SCHEMA if selected else RAW_VALUE_SCHEMA,
            "layout": {"flat_order": ["rho", "kinetic"], "rho": [cfg.control_count, cfg.rho_modes],
                "kinetic": [cfg.hermite_channels, cfg.control_count, cfg.kinetic_modes],
                "dimension": cfg.latent_dimension, "full_with_slopes_accepted": False,
                "time_axis_within_each_block": -2},
            "operators": {name: "e0=a0;ei=ai-a_previous" if name in selected else "identity"
                for name in ("rho", "kinetic")},
            "arithmetic": {"dtype": "preserve_input_float32_or_float64",
                "device": "preserve_input_cpu_or_cuda",
                "inverse": "fixed_left_to_right_native_dtype_sum" if selected else "identity",
                "real_arithmetic_invertible": True, "real_arithmetic_jacobian_determinant": 1,
                "floating_point_bitwise_roundtrip_guaranteed": False,
                "floating_roundtrip_error": "subtraction/cumulative_addition_roundoff; caller must check actual data"},
            "normalization": "none", "time_interval_division": False,
            "full_state_support": "all original value nodes/channels/Fourier modes",
            "fit_required": False, "physical_projection": False,
            "physics_or_distribution_admission": False}

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError("invalid temporal innovation transform payload")
        transform = cls(VPFPTemporalInnovationConfig.from_dict(payload["config"]))
        try:
            same = (json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
                == json.dumps(transform.to_payload(), sort_keys=True, allow_nan=False, separators=(",", ":")))
        except (TypeError, ValueError):
            same = False
        if not same:
            raise ValueError("temporal innovation payload conventions/schema/layout differ")
        return transform
