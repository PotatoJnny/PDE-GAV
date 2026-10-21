from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


EQUIVARIANT_MICRO_CODEC_CHECKPOINT_SCHEMA = (
    "vpfp_equivariant_micro_codec_checkpoint_v1"
)

FOURIER_MICRO_NORMALIZATION_SCHEMA = (
    "vpfp_fourier_pair_micro_normalization_v1"
)

FOURIER_MICRO_ZCA_NORMALIZATION_SCHEMA = (
    "vpfp_fourier_pair_micro_zca_normalization_v1"
)


@dataclass(frozen=True)
class VPFPEquivariantMicroCodecConfig:
    input_channels: int
    latent_channels: int = 128
    hidden_channels: tuple[int, int] = (512, 256)
    spatial_feature_count: int = 81
    radial_bottleneck: int = 128
    residual_pca_skip: bool = False

    def validate(self) -> None:
        values = (
            self.input_channels,
            self.latent_channels,
            *self.hidden_channels,
            self.spatial_feature_count,
            self.radial_bottleneck,
        )
        if min(int(value) for value in values) < 1:
            raise ValueError("equivariant micro codec dimensions must be positive")
        if len(self.hidden_channels) != 2:
            raise ValueError("equivariant micro codec needs two hidden widths")
        if int(self.spatial_feature_count) % 2 != 1:
            raise ValueError("real Fourier layout must contain 1+2M features")

    @property
    def mode_count(self) -> int:
        return (int(self.spatial_feature_count) - 1) // 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_channels": int(self.input_channels),
            "latent_channels": int(self.latent_channels),
            "hidden_channels": [
                int(value) for value in self.hidden_channels
            ],
            "spatial_feature_count": int(self.spatial_feature_count),
            "radial_bottleneck": int(self.radial_bottleneck),
            "residual_pca_skip": bool(self.residual_pca_skip),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "VPFPEquivariantMicroCodecConfig":
        payload = dict(data)
        payload["hidden_channels"] = tuple(
            int(value) for value in payload["hidden_channels"]
        )
        config = cls(**payload)
        config.validate()
        return config


class VPFPEquivariantMicroCodec:
    @staticmethod
    def build(
        config: VPFPEquivariantMicroCodecConfig | dict[str, Any],
    ):
        cfg = (
            config
            if isinstance(config, VPFPEquivariantMicroCodecConfig)
            else VPFPEquivariantMicroCodecConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPEquivariantMicroCodec(_torch(), cfg)


class _EquivariantRadialBlock(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, channels: int, bottleneck: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(
            int(channels), elementwise_affine=True, eps=1.0e-6
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(int(channels), int(bottleneck)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(bottleneck), int(channels)),
        )

    def forward(self, values):
        radius = self.torch.sqrt(
            values.square().sum(dim=-1) + 1.0e-8
        )
        gate = 2.0 * self.torch.sigmoid(self.gate(self.norm(radius)))
        return values * gate[..., None]

    @property
    def torch(self):
        return _torch()


class _EquivariantChannelMap(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, input_channels: int, output_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(
            int(input_channels), int(output_channels), bias=False
        )

    def forward(self, values):
        return self.linear(values.transpose(-1, -2)).transpose(-1, -2)


class _EquivariantPairNetwork(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(
        self,
        torch,
        channels: tuple[int, ...],
        *,
        radial_bottleneck: int,
    ):
        super().__init__()
        self.maps = torch.nn.ModuleList(
            [
                _EquivariantChannelMap(torch, left, right)
                for left, right in zip(channels[:-1], channels[1:], strict=True)
            ]
        )
        self.radial = torch.nn.ModuleList(
            [
                _EquivariantRadialBlock(
                    torch,
                    channels[index + 1],
                    min(int(radial_bottleneck), int(channels[index + 1])),
                )
                for index in range(len(channels) - 2)
            ]
        )

    def forward(self, values):
        hidden = values
        for index, mapping in enumerate(self.maps):
            hidden = mapping(hidden)
            if index < len(self.radial):
                hidden = self.radial[index](hidden)
        return hidden


class _RealNetwork(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, channels: tuple[int, ...]):
        super().__init__()
        layers = []
        for index, (left, right) in enumerate(
            zip(channels[:-1], channels[1:], strict=True)
        ):
            layers.append(torch.nn.Linear(int(left), int(right)))
            if index + 1 < len(channels) - 1:
                layers.append(torch.nn.GELU(approximate="tanh"))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


class _VPFPEquivariantMicroCodec(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch, config):
        super().__init__()
        self.config = config
        first, second = (int(value) for value in config.hidden_channels)
        encoder = (
            int(config.input_channels),
            first,
            second,
            int(config.latent_channels),
        )
        decoder = tuple(reversed(encoder))
        self.constant_encoder = _RealNetwork(torch, encoder)
        self.constant_decoder = _RealNetwork(torch, decoder)
        self.pair_encoder = _EquivariantPairNetwork(
            torch,
            encoder,
            radial_bottleneck=int(config.radial_bottleneck),
        )
        self.pair_decoder = _EquivariantPairNetwork(
            torch,
            decoder,
            radial_bottleneck=int(config.radial_bottleneck),
        )
        if bool(config.residual_pca_skip):
            self._zero_residual_outputs(torch)

    def encode(self, normalized_local):
        constant, pairs = _split_real_fourier(normalized_local, self.config)
        constant_code = self.constant_encoder(constant)
        pair_code = self.pair_encoder(pairs)
        if bool(self.config.residual_pca_skip):
            constant_code = constant_code + _constant_channel_skip(
                constant, int(self.config.latent_channels)
            )
            pair_code = pair_code + _pair_channel_skip(
                pairs, int(self.config.latent_channels)
            )
        return _join_real_fourier(constant_code, pair_code, self.config)

    def decode(self, latent_local):
        constant, pairs = _split_real_fourier(
            latent_local,
            self.config,
            channels=int(self.config.latent_channels),
        )
        constant_value = self.constant_decoder(constant)
        pair_value = self.pair_decoder(pairs)
        if bool(self.config.residual_pca_skip):
            constant_value = constant_value + _constant_channel_skip(
                constant, int(self.config.input_channels)
            )
            pair_value = pair_value + _pair_channel_skip(
                pairs, int(self.config.input_channels)
            )
        return _join_real_fourier(
            constant_value,
            pair_value,
            self.config,
            channels=int(self.config.input_channels),
        )

    def forward(self, normalized_local):
        return self.decode(self.encode(normalized_local))
    def _zero_residual_outputs(self, torch) -> None:
        for network in (self.constant_encoder, self.constant_decoder):
            output = network.network[-1]
            torch.nn.init.zeros_(output.weight)
            if output.bias is not None:
                torch.nn.init.zeros_(output.bias)
        for network in (self.pair_encoder, self.pair_decoder):
            torch.nn.init.zeros_(network.maps[-1].linear.weight)


def _constant_channel_skip(values, output_channels: int):
    channels = int(values.shape[1])
    output_channels = int(output_channels)
    if output_channels <= channels:
        return values[:, :output_channels]
    padding = values.new_zeros(
        int(values.shape[0]), output_channels - channels
    )
    return _torch().cat((values, padding), dim=1)


def _pair_channel_skip(values, output_channels: int):
    channels = int(values.shape[2])
    output_channels = int(output_channels)
    if output_channels <= channels:
        return values[:, :, :output_channels]
    padding = values.new_zeros(
        int(values.shape[0]),
        int(values.shape[1]),
        output_channels - channels,
        int(values.shape[3]),
    )
    return _torch().cat((values, padding), dim=2)


def save_equivariant_micro_codec_checkpoint(
    path: str | Path,
    *,
    model,
    micro_normalization: dict[str, Any],
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
            "schema": EQUIVARIANT_MICRO_CODEC_CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "micro_normalization": dict(micro_normalization),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_equivariant_micro_codec_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != EQUIVARIANT_MICRO_CODEC_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected equivariant micro codec checkpoint schema")
    model = VPFPEquivariantMicroCodec.build(payload["model_config"]).to(
        device=device
    )
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload



def fit_fourier_micro_normalization(
    local_values,
    *,
    min_scale_fraction: float = 0.02,
) -> dict[str, Any]:
    torch = _torch()
    values = torch.as_tensor(local_values)
    _validate_local(values)
    if not 0.0 < float(min_scale_fraction) < 1.0:
        raise ValueError("micro normalization scale fraction must lie in (0,1)")
    constant, pairs = _split_real_fourier_tensor(values)
    constant_mean = constant.mean(dim=0)
    constant_std = constant.std(dim=0, unbiased=False)
    constant_floor = float(min_scale_fraction) * torch.sqrt(
        torch.mean((constant - constant_mean).square())
    )
    constant_floor = torch.clamp(
        constant_floor, min=torch.finfo(values.dtype).eps
    )
    constant_scale = torch.clamp(constant_std, min=constant_floor)

    pair_channel_scale = torch.sqrt(
        torch.mean(pairs.square(), dim=(0, 1, 3))
    )
    channel_floor = float(min_scale_fraction) * torch.sqrt(
        torch.mean(pairs.square())
    )
    channel_floor = torch.clamp(
        channel_floor, min=torch.finfo(values.dtype).eps
    )
    pair_channel_scale = torch.clamp(
        pair_channel_scale, min=channel_floor
    )
    channel_normalized = pairs / pair_channel_scale[None, None, :, None]
    pair_mode_scale = torch.sqrt(
        torch.mean(channel_normalized.square(), dim=(0, 2, 3))
    )
    mode_floor = float(min_scale_fraction) * torch.sqrt(
        torch.mean(channel_normalized.square())
    )
    mode_floor = torch.clamp(
        mode_floor, min=torch.finfo(values.dtype).eps
    )
    pair_mode_scale = torch.clamp(pair_mode_scale, min=mode_floor)
    pair_mean = pairs.mean(dim=0)
    pair_rms = torch.sqrt(torch.mean(pairs.square(), dim=0))
    mean_ratio = torch.max(
        torch.abs(pair_mean) / torch.clamp(pair_rms, min=channel_floor)
    )
    return {
        "schema": FOURIER_MICRO_NORMALIZATION_SCHEMA,
        "constant_mean": constant_mean.detach().cpu(),
        "constant_scale": constant_scale.detach().cpu(),
        "pair_channel_scale": pair_channel_scale.detach().cpu(),
        "pair_mode_scale": pair_mode_scale.detach().cpu(),
        "min_scale_fraction": float(min_scale_fraction),
        "nonzero_pair_mean_to_rms_max": float(mean_ratio.detach().cpu()),
        "input_channels": int(values.shape[1]),
        "spatial_feature_count": int(values.shape[3]),
    }


def fit_fourier_micro_zca_normalization(
    local_values,
    *,
    min_scale_fraction: float = 0.02,
    shrinkage: float = 0.05,
    eigenvalue_floor_fraction: float = 0.01,
) -> dict[str, Any]:
    torch = _torch()
    values = torch.as_tensor(local_values)
    _validate_local(values)
    if not 0.0 <= float(shrinkage) < 1.0:
        raise ValueError("ZCA shrinkage must lie in [0,1)")
    if not 0.0 < float(eigenvalue_floor_fraction) < 1.0:
        raise ValueError("ZCA eigenvalue floor fraction must lie in (0,1)")
    base = fit_fourier_micro_normalization(
        values, min_scale_fraction=float(min_scale_fraction)
    )
    scaled = normalize_fourier_micro(values, base)
    constant, pairs = _split_real_fourier_tensor(scaled)
    pair_samples = pairs.permute(0, 1, 3, 2).reshape(
        -1, int(values.shape[1])
    )
    constant_whitening, constant_inverse, constant_diagnostics = _fit_zca(
        constant,
        shrinkage=float(shrinkage),
        eigenvalue_floor_fraction=float(eigenvalue_floor_fraction),
    )
    pair_whitening, pair_inverse, pair_diagnostics = _fit_zca(
        pair_samples,
        shrinkage=float(shrinkage),
        eigenvalue_floor_fraction=float(eigenvalue_floor_fraction),
    )
    return {
        "schema": FOURIER_MICRO_ZCA_NORMALIZATION_SCHEMA,
        "base_normalization": base,
        "constant_whitening": constant_whitening.detach().cpu(),
        "constant_inverse": constant_inverse.detach().cpu(),
        "pair_whitening": pair_whitening.detach().cpu(),
        "pair_inverse": pair_inverse.detach().cpu(),
        "shrinkage": float(shrinkage),
        "eigenvalue_floor_fraction": float(eigenvalue_floor_fraction),
        "constant_diagnostics": constant_diagnostics,
        "pair_diagnostics": pair_diagnostics,
        "input_channels": int(values.shape[1]),
        "spatial_feature_count": int(values.shape[3]),
    }


def normalize_fourier_micro(local_values, normalization: dict[str, Any]):
    torch = _torch()
    values = torch.as_tensor(local_values)
    if normalization.get("schema") == FOURIER_MICRO_ZCA_NORMALIZATION_SCHEMA:
        _validate_zca_normalization(normalization, values)
        scaled = normalize_fourier_micro(
            values, normalization["base_normalization"]
        )
        constant, pairs = _split_real_fourier_tensor(scaled)
        constant_matrix = _normalization_tensor(
            normalization, "constant_whitening", values
        )
        pair_matrix = _normalization_tensor(
            normalization, "pair_whitening", values
        )
        whitened_constant = constant @ constant_matrix.T
        whitened_pairs = torch.einsum("bmck,dc->bmdk", pairs, pair_matrix)
        return _join_real_fourier_tensor(whitened_constant, whitened_pairs)
    _validate_normalization(normalization, values)
    constant, pairs = _split_real_fourier_tensor(values)
    constant_mean = _normalization_tensor(
        normalization, "constant_mean", values
    )
    constant_scale = _normalization_tensor(
        normalization, "constant_scale", values
    )
    channel_scale = _normalization_tensor(
        normalization, "pair_channel_scale", values
    )
    mode_scale = _normalization_tensor(
        normalization, "pair_mode_scale", values
    )
    normalized_constant = (constant - constant_mean) / constant_scale
    normalized_pairs = pairs / (
        mode_scale[None, :, None, None]
        * channel_scale[None, None, :, None]
    )
    return _join_real_fourier_tensor(normalized_constant, normalized_pairs)


def denormalize_fourier_micro(local_values, normalization: dict[str, Any]):
    torch = _torch()
    values = torch.as_tensor(local_values)
    if normalization.get("schema") == FOURIER_MICRO_ZCA_NORMALIZATION_SCHEMA:
        _validate_zca_normalization(normalization, values)
        constant, pairs = _split_real_fourier_tensor(values)
        constant_matrix = _normalization_tensor(
            normalization, "constant_inverse", values
        )
        pair_matrix = _normalization_tensor(
            normalization, "pair_inverse", values
        )
        scaled_constant = constant @ constant_matrix.T
        scaled_pairs = torch.einsum("bmck,dc->bmdk", pairs, pair_matrix)
        scaled = _join_real_fourier_tensor(scaled_constant, scaled_pairs)
        return denormalize_fourier_micro(
            scaled, normalization["base_normalization"]
        )
    _validate_normalization(normalization, values)
    constant, pairs = _split_real_fourier_tensor(values)
    constant_mean = _normalization_tensor(
        normalization, "constant_mean", values
    )
    constant_scale = _normalization_tensor(
        normalization, "constant_scale", values
    )
    channel_scale = _normalization_tensor(
        normalization, "pair_channel_scale", values
    )
    mode_scale = _normalization_tensor(
        normalization, "pair_mode_scale", values
    )
    restored_constant = constant * constant_scale + constant_mean
    restored_pairs = pairs * (
        mode_scale[None, :, None, None]
        * channel_scale[None, None, :, None]
    )
    return _join_real_fourier_tensor(restored_constant, restored_pairs)


def _split_real_fourier(values, config, *, channels: int | None = None):
    tensor = _torch().as_tensor(values)
    expected_channels = (
        int(config.input_channels) if channels is None else int(channels)
    )
    expected = (
        tensor.ndim == 4
        and int(tensor.shape[1]) == expected_channels
        and int(tensor.shape[2]) == 1
        and int(tensor.shape[3]) == int(config.spatial_feature_count)
    )
    if not expected:
        raise ValueError("micro codec tensor does not match configured layout")
    return _split_real_fourier_tensor(tensor)


def _split_real_fourier_tensor(values):
    tensor = _torch().as_tensor(values)
    _validate_local(tensor)
    batch, channels = int(tensor.shape[0]), int(tensor.shape[1])
    constant = tensor[:, :, 0, 0]
    pairs = tensor[:, :, 0, 1:].reshape(
        batch, channels, -1, 2
    ).permute(0, 2, 1, 3)
    return constant, pairs


def _join_real_fourier(
    constant,
    pairs,
    config,
    *,
    channels: int | None = None,
):
    expected_channels = (
        int(config.latent_channels) if channels is None else int(channels)
    )
    if (
        constant.ndim != 2
        or pairs.ndim != 4
        or int(constant.shape[0]) != int(pairs.shape[0])
        or int(constant.shape[1]) != expected_channels
        or int(pairs.shape[1]) != int(config.mode_count)
        or int(pairs.shape[2]) != expected_channels
        or int(pairs.shape[3]) != 2
    ):
        raise ValueError("micro codec outputs do not match configured layout")
    return _join_real_fourier_tensor(constant, pairs)


def _join_real_fourier_tensor(constant, pairs):
    batch, channels = int(constant.shape[0]), int(constant.shape[1])
    nonzero = pairs.permute(0, 2, 1, 3).reshape(batch, channels, -1)
    return _torch().cat((constant[:, :, None], nonzero), dim=-1)[:, :, None]


def _normalization_tensor(normalization, name: str, reference):
    return _torch().as_tensor(
        normalization[name],
        dtype=reference.dtype,
        device=reference.device,
    )


def _validate_normalization(normalization, values) -> None:
    _validate_local(values)
    if normalization.get("schema") != FOURIER_MICRO_NORMALIZATION_SCHEMA:
        raise ValueError("unexpected Fourier micro normalization schema")
    if (
        int(normalization.get("input_channels", -1)) != int(values.shape[1])
        or int(normalization.get("spatial_feature_count", -1))
        != int(values.shape[3])
    ):
        raise ValueError("Fourier micro normalization layout changed")


def _fit_zca(
    samples,
    *,
    shrinkage: float,
    eigenvalue_floor_fraction: float,
):
    torch = _torch()
    values = torch.as_tensor(samples)
    if values.ndim != 2 or int(values.shape[0]) < 2:
        raise ValueError("ZCA fitting requires a nontrivial sample matrix")
    covariance = values.T @ values / float(values.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    mean_variance = torch.mean(torch.diagonal(covariance))
    epsilon = torch.finfo(values.dtype).eps
    if not bool(torch.isfinite(mean_variance).item()) or float(mean_variance) <= 0.0:
        raise ValueError("ZCA covariance has no finite positive variance")
    identity = torch.eye(
        int(values.shape[1]), dtype=values.dtype, device=values.device
    )
    regularized = (
        (1.0 - float(shrinkage)) * covariance
        + float(shrinkage) * mean_variance * identity
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
    floor = torch.clamp(
        float(eigenvalue_floor_fraction) * mean_variance,
        min=epsilon,
    )
    eigenvalues = torch.clamp(eigenvalues, min=floor)
    whitening = (
        eigenvectors * torch.rsqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    inverse = (
        eigenvectors * torch.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    raw_eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probability = raw_eigenvalues / raw_eigenvalues.sum().clamp_min(epsilon)
    effective_rank = torch.exp(
        -torch.sum(probability * torch.log(probability.clamp_min(epsilon)))
    )
    if not bool(
        torch.all(torch.isfinite(whitening)).item()
        and torch.all(torch.isfinite(inverse)).item()
    ):
        raise ValueError("ZCA transform is not finite")
    diagnostics = {
        "sample_count": int(values.shape[0]),
        "channel_count": int(values.shape[1]),
        "mean_variance": float(mean_variance.detach().cpu()),
        "raw_eigenvalue_min": float(raw_eigenvalues.min().detach().cpu()),
        "raw_eigenvalue_max": float(raw_eigenvalues.max().detach().cpu()),
        "regularized_eigenvalue_min": float(eigenvalues.min().detach().cpu()),
        "regularized_eigenvalue_max": float(eigenvalues.max().detach().cpu()),
        "regularized_condition_number": float(
            (eigenvalues.max() / eigenvalues.min()).detach().cpu()
        ),
        "raw_effective_rank": float(effective_rank.detach().cpu()),
    }
    return whitening, inverse, diagnostics


def _validate_zca_normalization(normalization, values) -> None:
    _validate_local(values)
    if normalization.get("schema") != FOURIER_MICRO_ZCA_NORMALIZATION_SCHEMA:
        raise ValueError("unexpected Fourier micro ZCA normalization schema")
    channels = int(values.shape[1])
    if (
        int(normalization.get("input_channels", -1)) != channels
        or int(normalization.get("spatial_feature_count", -1))
        != int(values.shape[3])
    ):
        raise ValueError("Fourier micro ZCA normalization layout changed")
    _validate_normalization(normalization["base_normalization"], values)
    for name in (
        "constant_whitening",
        "constant_inverse",
        "pair_whitening",
        "pair_inverse",
    ):
        matrix = _torch().as_tensor(normalization[name])
        if tuple(matrix.shape) != (channels, channels):
            raise ValueError(f"invalid ZCA matrix shape for {name}")


def _validate_local(values) -> None:
    torch = _torch()
    if (
        values.ndim != 4
        or int(values.shape[2]) != 1
        or int(values.shape[3]) % 2 != 1
    ):
        raise ValueError("micro values require [B,C,1,1+2M] layout")
    if not values.is_floating_point():
        raise ValueError("micro values must use floating point")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("micro values must be finite")


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
