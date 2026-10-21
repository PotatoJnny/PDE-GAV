from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any


STRUCTURED_NS_REWARD_GUIDANCE_SCHEMA = (
    "ns2d_structured_coefficient_reward_guidance_v1"
)


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for structured NS guidance") from exc
    return torch


@dataclass(frozen=True)
class StructuredRewardModelConfig:
    feature_dim: int = 256
    width: int = 512
    depth: int = 6
    time_width: int = 128

    def validate(self) -> None:
        if min(
            int(self.feature_dim),
            int(self.width),
            int(self.depth),
            int(self.time_width),
        ) < 1:
            raise ValueError("structured reward-model dimensions must be positive")
        if int(self.time_width) % 2:
            raise ValueError("structured reward-model time width must be even")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        return cls(**dict(payload))


class StructuredRewardModel:
    @staticmethod
    def build(config: StructuredRewardModelConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, StructuredRewardModelConfig)
            else StructuredRewardModelConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _StructuredRewardModel(_require_torch(), cfg)


class _RewardBlock(_require_torch().nn.Module):
    def __init__(self, torch, width: int, time_width: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.condition = torch.nn.Linear(time_width, 2 * width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, 2 * width),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * width, width),
        )

    def forward(self, values, time_embedding):
        scale, shift = self.condition(time_embedding).chunk(2, dim=-1)
        conditioned = self.norm(values) * (1.0 + scale) + shift
        return values + self.mlp(conditioned) / math.sqrt(2.0)


class _StructuredRewardModel(_require_torch().nn.Module):
    """Predict the current solver's scalar error from a noisy PCA state."""

    def __init__(self, torch, config: StructuredRewardModelConfig) -> None:
        super().__init__()
        config.validate()
        self.torch = torch
        self.config = config
        self.input = torch.nn.Linear(int(config.feature_dim), int(config.width))
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(int(config.time_width), int(config.time_width)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(config.time_width), int(config.time_width)),
        )
        self.blocks = torch.nn.ModuleList(
            _RewardBlock(torch, int(config.width), int(config.time_width))
            for _ in range(int(config.depth))
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(int(config.width)),
            torch.nn.Linear(int(config.width), 1),
        )

    def forward(self, values, time):
        if values.ndim != 2 or int(values.shape[1]) != int(
            self.config.feature_dim
        ):
            raise ValueError("structured reward-model input has the wrong shape")
        if time.shape != (values.shape[0],):
            raise ValueError("structured reward-model time has the wrong shape")
        embedding = self.time_mlp(
            _continuous_time_embedding(
                self.torch,
                time,
                int(self.config.time_width),
            )
        )
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.output(hidden)[:, 0]


class StructuredRewardGradient:
    """Turn a learned reward into a bounded rectified-flow velocity update.

    The reward gradient supplies only a direction. Per sample, its RMS is first
    matched to the frozen base-flow velocity RMS. ``strength`` then controls the
    requested relative update, while ``max_flow_ratio`` is a numerical guardrail.
    """

    def __init__(
        self,
        model,
        *,
        strength: float,
        max_flow_ratio: float,
        minimum_rms: float = 1.0e-8,
    ) -> None:
        if not math.isfinite(float(strength)) or float(strength) < 0.0:
            raise ValueError("structured reward guidance strength must be finite")
        if (
            not math.isfinite(float(max_flow_ratio))
            or float(max_flow_ratio) <= 0.0
        ):
            raise ValueError("structured reward guidance cap must be positive")
        if float(minimum_rms) <= 0.0:
            raise ValueError("structured reward guidance minimum RMS must be positive")
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.strength = float(strength)
        self.max_flow_ratio = float(max_flow_ratio)
        self.minimum_rms = float(minimum_rms)
        self._calls = 0
        self._samples = 0
        self._degenerate = None
        self._ratio_sum = None
        self._ratio_max = None
        self._gradient_rms_sum = None
        self._prediction_sum = None

    def __call__(self, values, time, base_velocity):
        tensors = self._tensor_update(values, time, base_velocity)
        self._commit_diagnostics(int(values.shape[0]), tensors)
        return tensors[0].to(device=values.device, dtype=values.dtype)

    def _tensor_update(self, values, time, base_velocity):
        """Pure tensor work shared by eager execution and the NS CUDA graph.

        This does not commit diagnostics or perform the CPU finite check.
        Callers must invoke ``_commit_diagnostics`` once per actual stage.
        """
        torch = _require_torch()
        if values.shape != base_velocity.shape:
            raise ValueError("guidance state and base velocity shapes differ")
        with torch.enable_grad():
            differentiated = values.detach().float().requires_grad_(True)
            prediction = self.model(differentiated, time.detach().float())
            gradient = torch.autograd.grad(prediction.sum(), differentiated)[0]

        base = base_velocity.detach().float()
        base_rms = base.square().mean(dim=1).sqrt().clamp_min(self.minimum_rms)
        gradient_rms = gradient.square().mean(dim=1).sqrt()
        valid = torch.isfinite(gradient_rms) & (
            gradient_rms >= self.minimum_rms
        )
        denominator = gradient_rms.clamp_min(self.minimum_rms)
        direction = gradient * (base_rms / denominator)[:, None]
        direction = torch.where(valid[:, None], direction, torch.zeros_like(direction))
        proposed = self.strength * direction
        proposed_rms = proposed.square().mean(dim=1).sqrt()
        ratio = proposed_rms / base_rms
        cap = torch.clamp(
            self.max_flow_ratio / ratio.clamp_min(self.minimum_rms),
            max=1.0,
        )
        update = proposed * cap[:, None]
        applied_ratio = update.square().mean(dim=1).sqrt() / base_rms
        return update, valid, applied_ratio, gradient_rms, prediction

    def _commit_diagnostics(self, sample_count: int, tensors) -> None:
        """Validate this stage before publishing its original diagnostics."""
        torch = _require_torch()
        update, valid, applied_ratio, gradient_rms, prediction = tensors
        if not torch.isfinite(update).all():
            raise FloatingPointError("structured reward guidance became non-finite")

        self._calls += 1
        self._samples += sample_count
        # Keep diagnostic reductions on the sampling device. Reading this
        # scalar here would synchronize the GPU at every integration stage.
        self._degenerate = _add_scalar(self._degenerate, (~valid).sum().detach())
        self._ratio_sum = _add_scalar(self._ratio_sum, applied_ratio.sum().detach())
        current_max = applied_ratio.max().detach()
        self._ratio_max = (
            current_max
            if self._ratio_max is None
            else torch.maximum(self._ratio_max, current_max)
        )
        self._gradient_rms_sum = _add_scalar(
            self._gradient_rms_sum,
            gradient_rms.sum().detach(),
        )
        self._prediction_sum = _add_scalar(
            self._prediction_sum,
            prediction.sum().detach(),
        )

    def diagnostics(self) -> dict[str, Any]:
        denominator = max(self._samples, 1)
        return {
            "calls": int(self._calls),
            "sample_evaluations": int(self._samples),
            "degenerate_gradient_evaluations": (
                0 if self._degenerate is None else int(self._degenerate.detach().cpu())
            ),
            "strength": float(self.strength),
            "max_flow_ratio": float(self.max_flow_ratio),
            "mean_applied_flow_ratio": _scalar(self._ratio_sum) / denominator,
            "max_applied_flow_ratio": _scalar(self._ratio_max),
            "mean_reward_gradient_rms": (
                _scalar(self._gradient_rms_sum) / denominator
            ),
            "mean_predicted_normalized_reward": (
                _scalar(self._prediction_sum) / denominator
            ),
        }


def load_structured_reward_guidance_checkpoint(
    path: str | Path,
    *,
    map_location: str | Any = "cpu",
):
    torch = _require_torch()
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("schema") != STRUCTURED_NS_REWARD_GUIDANCE_SCHEMA:
        raise ValueError("unexpected structured NS reward-guidance checkpoint schema")
    model = StructuredRewardModel.build(
        StructuredRewardModelConfig.from_dict(dict(payload["model_config"]))
    ).to(map_location)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _continuous_time_embedding(torch, time, width: int):
    if time.ndim != 1:
        raise ValueError("structured reward-model time must have shape [batch]")
    half = int(width) // 2
    frequencies = torch.exp(
        math.log(1_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = 2.0 * math.pi * time[:, None] * frequencies[None, :]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def _add_scalar(current, value):
    return value if current is None else current + value


def _scalar(value) -> float:
    return 0.0 if value is None else float(value.detach().cpu())
