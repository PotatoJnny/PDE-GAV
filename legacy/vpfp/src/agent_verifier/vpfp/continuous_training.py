from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_v2 import (
    VELOCITY_DENSITY_DICTIONARY,
    VPFPContinuousField,
    VPFPContinuousFieldConfig,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


CONTINUOUS_FEATURE_KEYS = (
    "log_rho",
    "rho_anomaly",
    "u_x",
    "u_y",
    "temperature_trace",
    "normal_stress",
    "shear_stress",
)


@dataclass(frozen=True)
class VPFPContinuousEncoderConfig:
    input_channels: int = len(CONTINUOUS_FEATURE_KEYS)
    latent_dim: int = 12
    width: int = 32
    depth: int = 3
    pooled_shape: tuple[int, int, int] = (2, 4, 4)
    dtype: str = "float32"
    representation: str = "vpfp_continuous_moment_encoder_v1"

    def validate(self) -> None:
        if min(self.input_channels, self.latent_dim, self.width, self.depth) < 1:
            raise ValueError("continuous encoder dimensions must be positive")
        if min(self.pooled_shape) < 1:
            raise ValueError("continuous encoder pooled_shape must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("continuous encoder dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pooled_shape"] = [int(value) for value in self.pooled_shape]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPContinuousEncoderConfig":
        payload = dict(data)
        if "pooled_shape" in payload:
            payload["pooled_shape"] = tuple(int(value) for value in payload["pooled_shape"])
        return cls(**payload)


@dataclass(frozen=True)
class VPFPContinuousLossConfig:
    rho_weight: float = 1.0
    rho_relative_weight: float = 0.5
    log_g_weight: float = 1.0
    f_relative_weight: float = 0.25
    moment_weight: float = 1.0
    latent_variance_weight: float = 0.05
    latent_covariance_weight: float = 0.02
    atom_diversity_weight: float = 0.01
    atom_range_weight: float = 0.01
    min_latent_std: float = 0.35
    min_atom_std: float = 0.12
    max_atom_std: float = 3.0
    max_atom_center: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VPFPContinuousEncoder:
    @staticmethod
    def build(config: VPFPContinuousEncoderConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPContinuousEncoderConfig)
            else VPFPContinuousEncoderConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPContinuousEncoder(_torch(), cfg)


class _VPFPContinuousEncoder((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch_module, config: VPFPContinuousEncoderConfig):
        super().__init__()
        torch = torch_module
        self.torch = torch
        self.config = config
        layers = []
        channels = int(config.input_channels)
        width = int(config.width)
        for level in range(int(config.depth)):
            out_channels = width * min(2**level, 4)
            layers.extend(
                (
                    torch.nn.Conv3d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        stride=1 if level == 0 else 2,
                        padding=1,
                    ),
                    torch.nn.GroupNorm(_group_count(out_channels), out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    torch.nn.GroupNorm(_group_count(out_channels), out_channels),
                    torch.nn.SiLU(),
                )
            )
            channels = out_channels
        self.features = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool3d(config.pooled_shape)
        flattened = channels * int(np.prod(config.pooled_shape))
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flattened, max(2 * config.latent_dim, width)),
            torch.nn.SiLU(),
            torch.nn.Linear(max(2 * config.latent_dim, width), config.latent_dim),
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, features):
        if features.ndim != 5:
            raise ValueError(
                f"continuous encoder expects [B,C,T,X,Y], got {tuple(features.shape)}"
            )
        return self.head(self.pool(self.features(features)))


def density_mixture_moments(model, coefficients):
    """Return u and centered second moments for normalized Gaussian atoms."""

    if model.config.velocity_representation != VELOCITY_DENSITY_DICTIONARY:
        raise ValueError("analytic mixture moments require the density-dictionary head")
    torch = _torch()
    means = model.density_atom_means
    variances = model.density_atom_stds().square()
    ux = torch.sum(coefficients * means[None, :, 0], dim=-1)
    uy = torch.sum(coefficients * means[None, :, 1], dim=-1)
    raw_xx = torch.sum(
        coefficients * (variances[None, :, 0] + means[None, :, 0].square()),
        dim=-1,
    )
    raw_yy = torch.sum(
        coefficients * (variances[None, :, 1] + means[None, :, 1].square()),
        dim=-1,
    )
    raw_xy = torch.sum(
        coefficients * (means[None, :, 0] * means[None, :, 1]),
        dim=-1,
    )
    centered_xx = raw_xx - ux.square()
    centered_yy = raw_yy - uy.square()
    centered_xy = raw_xy - ux * uy
    return torch.stack(
        (
            ux,
            uy,
            0.5 * (centered_xx + centered_yy),
            centered_xx - centered_yy,
            centered_xy,
        ),
        dim=-1,
    )


def continuous_feature_grid(
    model,
    latents,
    *,
    nt: int,
    nx: int,
    ny: int,
    chunk_size: int = 8192,
):
    """Sample compact encoder features from continuous hidden functions."""

    torch = _torch()
    parameter = next(model.parameters())
    latent = torch.as_tensor(latents, dtype=parameter.dtype, device=parameter.device)
    if latent.ndim == 1:
        latent = latent[None, :]
    t0, t1 = model.config.time_domain
    lx, ly = model.config.equation.x_domain
    t = torch.linspace(t0, t1, int(nt), dtype=parameter.dtype, device=parameter.device)
    x = torch.arange(int(nx), dtype=parameter.dtype, device=parameter.device) * (
        float(lx) / float(nx)
    )
    y = torch.arange(int(ny), dtype=parameter.dtype, device=parameter.device) * (
        float(ly) / float(ny)
    )
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="ij")
    base_points = torch.stack(
        (tt, xx, yy, torch.zeros_like(tt), torch.zeros_like(tt)), dim=-1
    ).reshape(-1, 5)
    point_count = int(base_points.shape[0])
    all_features = []
    for case_index in range(int(latent.shape[0])):
        rho_rows = []
        moment_rows = []
        case_latent = latent[case_index]
        for start in range(0, point_count, int(chunk_size)):
            points = base_points[start : start + int(chunk_size)]
            with torch.no_grad():
                fields = model.fields(points, case_latent)
                moments = density_mixture_moments(
                    model, fields["velocity_coefficients"]
                )
            rho_rows.append(fields["rho"].detach())
            moment_rows.append(moments.detach())
        rho = torch.cat(rho_rows, dim=0)
        moments = torch.cat(moment_rows, dim=0)
        channels = torch.stack(
            (
                torch.log(rho.clamp_min(torch.finfo(rho.dtype).tiny)),
                rho / float(model.config.equation.background_density) - 1.0,
                moments[:, 0],
                moments[:, 1],
                moments[:, 2],
                moments[:, 3],
                moments[:, 4],
            ),
            dim=0,
        ).reshape(len(CONTINUOUS_FEATURE_KEYS), int(nt), int(nx), int(ny))
        all_features.append(channels)
    return torch.stack(all_features, dim=0)


def fit_feature_normalization(features, *, min_std: float = 1.0e-5) -> dict[str, Any]:
    torch = _torch()
    values = torch.as_tensor(features)
    if values.ndim != 5:
        raise ValueError("feature normalization expects [B,C,T,X,Y]")
    mean = torch.mean(values, dim=(0, 2, 3, 4))
    std = torch.std(values, dim=(0, 2, 3, 4), unbiased=False).clamp_min(min_std)
    return {
        "keys": list(CONTINUOUS_FEATURE_KEYS),
        "mean": [float(value) for value in mean.detach().cpu()],
        "std": [float(value) for value in std.detach().cpu()],
        "min_std": float(min_std),
    }


def normalize_feature_grid(features, normalization: dict[str, Any]):
    torch = _torch()
    values = torch.as_tensor(features)
    mean = torch.as_tensor(
        normalization["mean"], dtype=values.dtype, device=values.device
    )[None, :, None, None, None]
    std = torch.as_tensor(
        normalization["std"], dtype=values.dtype, device=values.device
    )[None, :, None, None, None]
    return (values - mean) / std


def sample_phase_points(
    *,
    torch_module,
    batch_size: int,
    points_per_case: int,
    field_config: VPFPContinuousFieldConfig,
    device,
    dtype,
    initial_fraction: float = 0.25,
    broad_velocity_fraction: float = 0.25,
):
    torch = torch_module
    shape = (int(batch_size), int(points_per_case))
    t0, t1 = field_config.time_domain
    lx, ly = field_config.equation.x_domain
    t = float(t0) + (float(t1) - float(t0)) * torch.rand(
        shape, dtype=dtype, device=device
    )
    initial_count = min(
        int(points_per_case), max(1, int(round(points_per_case * initial_fraction)))
    )
    t[:, :initial_count] = float(t0)
    x = float(lx) * torch.rand(shape, dtype=dtype, device=device)
    y = float(ly) * torch.rand(shape, dtype=dtype, device=device)
    theta = float(field_config.equation.temperature)
    velocity = theta**0.5 * torch.randn(
        (*shape, 2), dtype=dtype, device=device
    )
    broad_count = min(
        int(points_per_case),
        max(0, int(round(points_per_case * broad_velocity_fraction))),
    )
    if broad_count:
        velocity[:, -broad_count:, :] = 2.5 * theta**0.5 * torch.randn(
            (int(batch_size), broad_count, 2), dtype=dtype, device=device
        )
    return torch.stack((t, x, y, velocity[:, :, 0], velocity[:, :, 1]), dim=-1)


def continuous_autoencoder_loss(
    *,
    encoder,
    decoder,
    teacher,
    normalized_features,
    teacher_latents,
    points_per_case: int,
    config: VPFPContinuousLossConfig,
):
    torch = _torch()
    encoded = encoder(normalized_features)
    batch = int(encoded.shape[0])
    points = sample_phase_points(
        torch_module=torch,
        batch_size=batch,
        points_per_case=points_per_case,
        field_config=decoder.config,
        device=encoded.device,
        dtype=encoded.dtype,
    )
    flat_points = points.reshape(-1, 5)
    student_latent = encoded.repeat_interleave(int(points_per_case), dim=0)
    target_latent = teacher_latents.repeat_interleave(int(points_per_case), dim=0)
    with torch.no_grad():
        target = teacher.fields(flat_points, target_latent)
    prediction = decoder.fields(flat_points, student_latent)
    epsilon = torch.finfo(encoded.dtype).eps
    rho_loss = torch.mean(
        (
            torch.log(prediction["rho"].clamp_min(epsilon))
            - torch.log(target["rho"].clamp_min(epsilon))
        ).square()
    )
    log_g_loss = torch.mean(
        (
            torch.log(prediction["g"].clamp_min(1.0e-12))
            - torch.log(target["g"].clamp_min(1.0e-12))
        ).square()
    )
    rho_relative_squared = torch.mean(
        (prediction["rho"] - target["rho"]).square()
    ) / (torch.mean(target["rho"].square()) + epsilon)
    g_relative_squared = torch.mean(
        (prediction["g"] - target["g"]).square()
    ) / (torch.mean(target["g"].square()) + epsilon)
    f_relative_squared = torch.mean(
        (prediction["f"] - target["f"]).square()
    ) / (
        torch.mean(target["f"].square()) + epsilon
    )
    rho_relative_l2 = torch.sqrt(rho_relative_squared)
    g_relative_l2 = torch.sqrt(g_relative_squared)
    f_relative_l2 = torch.sqrt(f_relative_squared)

    moment_points = points[:, :, :3].reshape(-1, 3)
    moment_points = torch.cat(
        (
            moment_points,
            torch.zeros(
                (int(moment_points.shape[0]), 2),
                dtype=encoded.dtype,
                device=encoded.device,
            ),
        ),
        dim=-1,
    )
    with torch.no_grad():
        target_moment_fields = teacher.fields(moment_points, target_latent)
        target_moments = density_mixture_moments(
            teacher, target_moment_fields["velocity_coefficients"]
        )
    predicted_moment_fields = decoder.fields(moment_points, student_latent)
    predicted_moments = density_mixture_moments(
        decoder, predicted_moment_fields["velocity_coefficients"]
    )
    moment_scale = 0.05 + target_moments.square()
    moment_loss = torch.mean(
        (predicted_moments - target_moments).square() / moment_scale
    )

    variance_loss, covariance_loss = latent_spread_losses(
        encoded, min_std=float(config.min_latent_std)
    )
    diversity_loss, range_loss = density_atom_regularization(decoder, config)
    total = (
        float(config.rho_weight) * rho_loss
        + float(config.rho_relative_weight) * rho_relative_squared
        + float(config.log_g_weight) * log_g_loss
        + float(config.f_relative_weight) * f_relative_squared
        + float(config.moment_weight) * moment_loss
        + float(config.latent_variance_weight) * variance_loss
        + float(config.latent_covariance_weight) * covariance_loss
        + float(config.atom_diversity_weight) * diversity_loss
        + float(config.atom_range_weight) * range_loss
    )
    return total, {
        "loss": total,
        "rho_loss": rho_loss,
        "log_g_loss": log_g_loss,
        "rho_relative_l2": rho_relative_l2,
        "g_relative_l2": g_relative_l2,
        "f_relative_l2": f_relative_l2,
        "f_relative_squared": f_relative_squared,
        "moment_loss": moment_loss,
        "latent_variance_loss": variance_loss,
        "latent_covariance_loss": covariance_loss,
        "atom_diversity_loss": diversity_loss,
        "atom_range_loss": range_loss,
        "latent_std_mean": torch.std(encoded, dim=0, unbiased=False).mean(),
    }


def latent_spread_losses(latent, *, min_std: float):
    torch = _torch()
    if int(latent.shape[0]) < 2:
        zero = latent.sum() * 0.0
        return zero, zero
    centered = latent - torch.mean(latent, dim=0, keepdim=True)
    std = torch.sqrt(torch.mean(centered.square(), dim=0) + 1.0e-6)
    variance = torch.mean(torch.relu(float(min_std) - std).square())
    covariance = centered.T @ centered / float(max(1, int(latent.shape[0]) - 1))
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = torch.mean(off_diagonal.square())
    return variance, covariance_loss


def density_atom_regularization(model, config: VPFPContinuousLossConfig):
    torch = _torch()
    means = model.density_atom_means
    stds = model.density_atom_stds()
    if int(means.shape[0]) > 1:
        distance = torch.cdist(means, means)
        mask = ~torch.eye(
            int(means.shape[0]), dtype=torch.bool, device=means.device
        )
        diversity = torch.mean(torch.exp(-distance[mask].square() / 0.5))
    else:
        diversity = means.sum() * 0.0
    std_range = torch.mean(torch.relu(float(config.min_atom_std) - stds).square())
    std_range = std_range + torch.mean(
        torch.relu(stds - float(config.max_atom_std)).square()
    )
    center_range = torch.mean(
        torch.relu(torch.abs(means) - float(config.max_atom_center)).square()
    )
    return diversity, std_range + center_range


def save_continuous_decoder_checkpoint(path: str | Path, model, metadata=None) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_continuous_decoder_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        destination,
    )


def load_continuous_decoder_checkpoint(path: str | Path, *, device="cpu"):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_continuous_decoder_checkpoint_v1":
        raise ValueError("unexpected continuous decoder checkpoint schema")
    model = VPFPContinuousField.build(payload["model_config"]).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def save_continuous_autoencoder_checkpoint(
    path: str | Path,
    *,
    encoder,
    decoder,
    feature_normalization: dict[str, Any],
    train_config: dict[str, Any],
    step: int,
    optimizer=None,
) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_continuous_autoencoder_checkpoint_v1",
            "encoder_config": encoder.config.to_dict(),
            "decoder_config": decoder.config.to_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "feature_normalization": feature_normalization,
            "train_config": train_config,
            "step": int(step),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
        },
        destination,
    )


def load_continuous_autoencoder_checkpoint(path: str | Path, *, device="cpu"):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_continuous_autoencoder_checkpoint_v1":
        raise ValueError("unexpected continuous autoencoder checkpoint schema")
    encoder = VPFPContinuousEncoder.build(payload["encoder_config"]).to(device=device)
    decoder = VPFPContinuousField.build(payload["decoder_config"]).to(device=device)
    encoder.load_state_dict(payload["encoder_state_dict"])
    decoder.load_state_dict(payload["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, payload


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups:
        groups -= 1
    return groups


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for VPFP continuous training") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
