from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_verifier.vpfp.structured_diffusion import split_active_latent


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPHybridRewardModelConfig:
    macro_feature_dim: int
    micro_feature_dim: int = 512
    hidden_width: int = 160
    hidden_layers: int = 3
    dropout: float = 0.05
    dtype: str = "float32"
    representation: str = "vpfp_hybrid_dual_continuous_reward_guidance_v1"

    def validate(self) -> None:
        if min(
            int(self.macro_feature_dim),
            int(self.micro_feature_dim),
            int(self.hidden_width),
            int(self.hidden_layers),
        ) < 1:
            raise ValueError("hybrid reward model dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("hybrid reward dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPHybridRewardModelConfig":
        return cls(**dict(data))


class VPFPHybridRewardModel:
    @staticmethod
    def build(config: VPFPHybridRewardModelConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPHybridRewardModelConfig)
            else VPFPHybridRewardModelConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPHybridRewardModel(_torch(), cfg)


class _VPFPHybridRewardModel(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config: VPFPHybridRewardModelConfig):
        super().__init__()
        self.torch = torch
        self.config = config
        width = int(config.hidden_width)
        self.macro_encoder = torch.nn.Sequential(
            torch.nn.Linear(int(config.macro_feature_dim), width),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(width),
        )
        self.micro_encoder = torch.nn.Sequential(
            torch.nn.Linear(int(config.micro_feature_dim), width),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(width),
        )
        self.input = torch.nn.Linear(3 * width, width)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(width),
                    torch.nn.Linear(width, 2 * width),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(float(config.dropout)),
                    torch.nn.Linear(2 * width, width),
                )
                for _ in range(int(config.hidden_layers))
            ]
        )
        self.output_norm = torch.nn.LayerNorm(width)
        self.absolute_macro_head = torch.nn.Linear(width, 1)
        self.absolute_interaction_head = torch.nn.Linear(width, 1)
        self.centered_head = torch.nn.Linear(width, 1)
        self.register_buffer(
            "macro_mean", torch.zeros(int(config.macro_feature_dim))
        )
        self.register_buffer(
            "macro_std", torch.ones(int(config.macro_feature_dim))
        )
        self.register_buffer(
            "micro_mean", torch.zeros(int(config.micro_feature_dim))
        )
        self.register_buffer(
            "micro_std", torch.ones(int(config.micro_feature_dim))
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def set_feature_normalization(
        self, *, macro_mean, macro_std, micro_mean, micro_std
    ) -> None:
        torch = self.torch
        values = {
            "macro_mean": macro_mean,
            "macro_std": macro_std,
            "micro_mean": micro_mean,
            "micro_std": micro_std,
        }
        for name, value in values.items():
            destination = getattr(self, name)
            source = torch.as_tensor(
                value, dtype=destination.dtype, device=destination.device
            )
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(f"{name} has the wrong shape")
            if name.endswith("_std") and bool(torch.any(source <= 0.0).item()):
                raise ValueError(f"{name} must be positive")
            destination.copy_(source)

    def forward(self, macro_features, micro_features):
        torch = self.torch
        parameter = next(self.parameters())
        macro = torch.as_tensor(
            macro_features, dtype=parameter.dtype, device=parameter.device
        )
        micro = torch.as_tensor(
            micro_features, dtype=parameter.dtype, device=parameter.device
        )
        if macro.ndim != 2 or int(macro.shape[1]) != int(
            self.config.macro_feature_dim
        ):
            raise ValueError("macro feature shape does not match reward model")
        if micro.ndim != 2 or int(micro.shape[1]) != int(
            self.config.micro_feature_dim
        ):
            raise ValueError("micro feature shape does not match reward model")
        if int(macro.shape[0]) != int(micro.shape[0]):
            raise ValueError("macro and micro feature batches differ")
        macro_hidden = self.macro_encoder(
            (macro - self.macro_mean) / self.macro_std
        )
        micro_hidden = self.micro_encoder(
            (micro - self.micro_mean) / self.micro_std
        )
        hidden = torch.nn.functional.silu(
            self.input(
                torch.cat(
                    (macro_hidden, micro_hidden, macro_hidden * micro_hidden),
                    dim=-1,
                )
            )
        )
        block_scale = float(max(1, len(self.blocks))) ** -0.5
        for block in self.blocks:
            hidden = hidden + block_scale * block(hidden)
        hidden = torch.nn.functional.silu(self.output_norm(hidden))
        absolute = (
            self.absolute_macro_head(macro_hidden)
            + self.absolute_interaction_head(hidden)
        )[:, 0]
        return {
            "absolute": absolute,
            "centered": self.centered_head(hidden)[:, 0],
        }


def hybrid_macro_features(base_compressed, equation_conditions, transform):
    """Build torus-invariant macro statistics plus local attack context."""

    torch = _torch()
    base = torch.as_tensor(base_compressed)
    conditions = torch.as_tensor(
        equation_conditions, dtype=base.dtype, device=base.device
    )
    config = transform.compressed_config
    if base.ndim != 2 or int(base.shape[1]) != int(config.active_latent_dim):
        raise ValueError("base compressed latent has the wrong shape")
    if conditions.ndim != 2 or tuple(conditions.shape) != (
        int(base.shape[0]),
        int(config.condition_dim),
    ):
        raise ValueError("equation conditions have the wrong shape")
    rho, local = split_active_latent(base, config)
    if int(rho.shape[-1]) % 2:
        raise ValueError("rho Fourier features must form sine/cosine pairs")
    epsilon = torch.finfo(base.dtype).eps
    rho_pairs = rho.reshape(
        int(base.shape[0]), int(config.rho_time_count), -1, 2
    )
    rho_amplitude = torch.sqrt(
        torch.sum(rho_pairs.square(), dim=-1) + epsilon
    ).reshape(int(base.shape[0]), -1)

    scales = transform.component_scales.to(
        dtype=base.dtype, device=base.device
    )[None, :, None, None]
    whitened_local = local / scales
    component_rms = torch.sqrt(
        torch.mean(whitened_local.square(), dim=(2, 3)) + epsilon
    )
    spatial_dc = torch.sqrt(
        torch.mean(whitened_local[..., :1].square(), dim=(1, 2)) + epsilon
    )
    spatial_pairs = whitened_local[..., 1:].reshape(
        int(base.shape[0]), int(config.local_channels),
        int(config.local_time_count), -1, 2
    )
    spatial_pair_rms = torch.sqrt(
        torch.mean(torch.sum(spatial_pairs.square(), dim=-1), dim=(1, 2))
        + epsilon
    )
    base_micro = transform.base_micro_coordinates(base)
    features = torch.cat(
        (
            rho_amplitude,
            component_rms,
            spatial_dc,
            spatial_pair_rms,
            base_micro,
            conditions,
        ),
        dim=-1,
    )
    if not bool(torch.all(torch.isfinite(features)).item()):
        raise FloatingPointError("hybrid macro features are non-finite")
    return features


def hybrid_macro_feature_schema(transform) -> dict[str, int]:
    config = transform.compressed_config
    return {
        "rho_pair_amplitudes": int(
            config.rho_time_count * config.rho_feature_count // 2
        ),
        "component_rms": int(config.local_channels),
        "spatial_pair_rms_with_dc": int(
            1 + (config.local_spatial_feature_count - 1) // 2
        ),
        "base_micro_coordinates": int(transform.micro_code_dim),
        "equation_conditions": int(config.condition_dim),
    }


def hybrid_reward_regression_loss(
    model,
    macro_features,
    micro_features,
    *,
    absolute_target,
    centered_target,
    group_ids,
    absolute_weight: float = 1.0,
    centered_weight: float = 1.0,
    ranking_weight: float = 0.25,
):
    torch = _torch()
    prediction = model(macro_features, micro_features)
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
    absolute_loss = torch.nn.functional.smooth_l1_loss(
        prediction["absolute"], absolute
    )
    centered_loss = torch.nn.functional.smooth_l1_loss(
        prediction["centered"], centered
    )
    ranking_loss = grouped_pairwise_ranking_loss(
        prediction["centered"], centered, group_ids
    )
    total = (
        float(absolute_weight) * absolute_loss
        + float(centered_weight) * centered_loss
        + float(ranking_weight) * ranking_loss
    )
    return total, {
        "loss": total,
        "absolute_huber": absolute_loss,
        "centered_huber": centered_loss,
        "pairwise_ranking": ranking_loss,
    }


def grouped_pairwise_ranking_loss(prediction, target, group_ids):
    torch = _torch()
    predicted = torch.as_tensor(prediction)
    expected = torch.as_tensor(
        target, dtype=predicted.dtype, device=predicted.device
    )
    groups = torch.as_tensor(group_ids, device=predicted.device).reshape(-1)
    losses = []
    for group in torch.unique(groups):
        selected = torch.nonzero(groups == group, as_tuple=False)[:, 0]
        if int(selected.numel()) < 2:
            continue
        predicted_difference = (
            predicted.index_select(0, selected)[:, None]
            - predicted.index_select(0, selected)[None, :]
        )
        expected_difference = (
            expected.index_select(0, selected)[:, None]
            - expected.index_select(0, selected)[None, :]
        )
        upper = torch.triu(
            torch.ones_like(expected_difference, dtype=torch.bool), diagonal=1
        )
        informative = upper & (torch.abs(expected_difference) > 1.0e-6)
        if bool(torch.any(informative).item()):
            signed = torch.sign(expected_difference[informative])
            losses.append(
                torch.nn.functional.softplus(
                    -signed * predicted_difference[informative]
                ).mean()
            )
    if not losses:
        return predicted.sum() * 0.0
    return torch.stack(losses).mean()


def optimize_hybrid_micro_codes(
    model,
    transform,
    base_compressed,
    equation_conditions,
    *,
    starts: int = 8,
    steps: int = 80,
    lr: float = 0.08,
    absolute_scale: float = 1.0,
    centered_scale: float = 1.0,
    seed: int = 20260715,
):
    """Optimize the differentiable bottleneck against a black-box-trained surrogate."""

    torch = _torch()
    if min(int(starts), int(steps)) < 1 or float(lr) <= 0.0:
        raise ValueError("guidance optimization parameters must be positive")
    parameter = next(model.parameters())
    base = torch.as_tensor(
        base_compressed, dtype=parameter.dtype, device=parameter.device
    )
    conditions = torch.as_tensor(
        equation_conditions, dtype=parameter.dtype, device=parameter.device
    )
    macro = hybrid_macro_features(base, conditions, transform)
    batch = int(base.shape[0])
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(int(seed))
    raw = torch.randn(
        (batch, int(starts), int(transform.micro_code_dim)),
        generator=generator,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    raw[:, 0] = 0.0
    raw = torch.nn.Parameter(raw)
    optimizer = torch.optim.Adam([raw], lr=float(lr))
    repeated_macro = macro[:, None, :].expand(-1, int(starts), -1).reshape(
        batch * int(starts), -1
    )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        flat = raw.reshape(batch * int(starts), -1)
        effective = transform.effective_micro_coordinates(flat, 1.0)
        prediction = model(repeated_macro, effective)
        objective = (
            float(absolute_scale) * prediction["absolute"]
            + float(centered_scale) * prediction["centered"]
        )
        loss = -torch.mean(objective)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("hybrid guidance objective became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 5.0)
        optimizer.step()
        with torch.no_grad():
            raw.clamp_(-4.0, 4.0)
    with torch.no_grad():
        flat = raw.reshape(batch * int(starts), -1)
        effective = transform.effective_micro_coordinates(flat, 1.0)
        prediction = model(repeated_macro, effective)
        objective = (
            float(absolute_scale) * prediction["absolute"]
            + float(centered_scale) * prediction["centered"]
        ).reshape(batch, int(starts))
        selected = torch.argmax(objective, dim=1)
        rows = torch.arange(batch, device=parameter.device)
        selected_raw = raw.detach()[rows, selected]
        selected_effective = effective.reshape(
            batch, int(starts), -1
        )[rows, selected]
    return {
        "micro_codes": selected_raw,
        "effective_micro_coordinates": selected_effective,
        "predicted_objective": objective[rows, selected],
        "selected_start": selected,
        "all_predicted_objectives": objective,
    }


def save_hybrid_reward_checkpoint(
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
            "schema": "vpfp_hybrid_reward_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "reward_normalization": dict(reward_normalization),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
        },
        destination,
    )


def load_hybrid_reward_checkpoint(path: str | Path, *, device="cpu"):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_hybrid_reward_checkpoint_v1":
        raise ValueError("unexpected hybrid reward checkpoint schema")
    model = VPFPHybridRewardModel.build(payload["model_config"]).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for hybrid reward guidance") from (
            _TORCH_IMPORT_ERROR
        )
    return _TORCH_MODULE
