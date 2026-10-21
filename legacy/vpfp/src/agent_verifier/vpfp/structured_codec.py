from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
    _physical_fourier_basis,
    join_active_latent,
    split_active_latent,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPTorusCodecConfig:
    physical_grid_size: int = 15
    block_size: int = 3
    latent_channels: int = 128
    width: int = 192
    residual_blocks: int = 6
    dropout: float = 0.0
    dtype: str = "float32"
    representation: str = "vpfp_periodic_space_to_depth_codec_v1"

    @property
    def latent_grid_size(self) -> int:
        return int(self.physical_grid_size) // int(self.block_size)

    @property
    def latent_dim(self) -> int:
        return (
            int(self.latent_channels)
            * self.latent_grid_size
            * self.latent_grid_size
        )

    def validate(
        self,
        source_config: VPFPStructuredDiffusionConfig | None = None,
    ) -> None:
        if min(
            int(self.physical_grid_size),
            int(self.block_size),
            int(self.latent_channels),
            int(self.width),
            int(self.residual_blocks),
        ) < 1:
            raise ValueError("torus codec dimensions must be positive")
        if int(self.physical_grid_size) % int(self.block_size):
            raise ValueError("codec grid size must be divisible by block size")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("codec dropout must lie in [0,1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("codec dtype must be float32 or float64")
        if self.representation != "vpfp_periodic_space_to_depth_codec_v1":
            raise ValueError("unsupported torus codec representation")
        if source_config is not None:
            source_config.validate()
            rho_side = int(round(np.sqrt(2 * source_config.rho_mode_count + 1)))
            if rho_side * rho_side != 2 * source_config.rho_mode_count + 1:
                raise ValueError("codec requires a square rho Fourier layout")
            rho_max_mode = (rho_side - 1) // 2
            represented_mode = max(
                rho_max_mode,
                int(source_config.local_spatial_mode),
            )
            if int(self.physical_grid_size) < 2 * represented_mode + 1:
                raise ValueError("codec physical grid aliases represented modes")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPTorusCodecConfig":
        return cls(**dict(data))


def active_to_periodic_fields(
    values,
    source_config: VPFPStructuredDiffusionConfig,
    *,
    grid_size: int,
):
    """Synthesize normalized structured coefficients on an orthonormal grid."""

    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or int(tensor.shape[1]) != source_config.active_latent_dim:
        raise ValueError("active latent does not match codec source config")
    rho, local = split_active_latent(tensor, source_config)
    rho_basis, local_basis = _codec_bases(
        torch,
        source_config,
        grid_size=int(grid_size),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    rho_fields = torch.einsum("btf,pf->btp", rho, rho_basis)
    local_coefficients = local.reshape(
        int(tensor.shape[0]),
        source_config.local_channels * source_config.local_time_count,
        source_config.local_spatial_feature_count,
    )
    local_fields = torch.einsum(
        "bcf,pf->bcp",
        local_coefficients,
        local_basis,
    )
    fields = torch.cat((rho_fields, local_fields), dim=1)
    return fields.reshape(
        int(tensor.shape[0]),
        int(fields.shape[1]),
        int(grid_size),
        int(grid_size),
    )


def periodic_fields_to_active(
    fields,
    source_config: VPFPStructuredDiffusionConfig,
    *,
    grid_size: int,
):
    """Project periodic fields back to the represented Fourier coefficients."""

    torch = _torch()
    tensor = torch.as_tensor(fields)
    expected_channels = (
        source_config.rho_time_count
        + source_config.local_channels * source_config.local_time_count
    )
    if tensor.ndim != 4:
        raise ValueError("periodic fields do not match codec source config")
    expected = (
        int(tensor.shape[0]),
        expected_channels,
        int(grid_size),
        int(grid_size),
    )
    if tuple(tensor.shape) != expected:
        raise ValueError("periodic fields do not match codec source config")
    rho_basis, local_basis = _codec_bases(
        torch,
        source_config,
        grid_size=int(grid_size),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    flattened = tensor.reshape(int(tensor.shape[0]), expected_channels, -1)
    rho_fields = flattened[:, : source_config.rho_time_count]
    local_fields = flattened[:, source_config.rho_time_count :]
    rho = torch.einsum("btp,pf->btf", rho_fields, rho_basis)
    local_coefficients = torch.einsum(
        "bcp,pf->bcf",
        local_fields,
        local_basis,
    )
    local = local_coefficients.reshape(
        int(tensor.shape[0]),
        source_config.local_channels,
        source_config.local_time_count,
        source_config.local_spatial_feature_count,
    )
    return join_active_latent(rho, local, source_config)


def prepare_fixed_codec_corpus(
    values,
    *,
    expected_cases: int,
    expected_dim: int,
    device,
):
    """Validate a transient fixed corpus and colocate it with training."""

    torch = _torch()
    tensor = torch.as_tensor(values)
    expected = (int(expected_cases), int(expected_dim))
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"fixed codec corpus shape must be {expected}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point() or not bool(
        torch.all(torch.isfinite(tensor)).item()
    ):
        raise ValueError("fixed codec corpus must be finite floating point")
    return tensor.to(device=device)


def fit_block_energy_structured_normalization(
    values,
    source_config: VPFPStructuredDiffusionConfig,
    *,
    min_std: float = 1.0e-4,
) -> dict[str, Any]:
    """Normalize rho/local blocks without whitening their energy spectrum."""

    array = np.asarray(
        _torch().as_tensor(values).detach().cpu(),
        dtype=np.float64,
    )
    if (
        array.ndim != 2
        or int(array.shape[1]) != int(source_config.active_latent_dim)
    ):
        raise ValueError("structured normalization shape mismatch")
    if int(array.shape[0]) < 2:
        raise ValueError("structured normalization requires at least two cases")
    if not np.all(np.isfinite(array)):
        raise ValueError("structured normalization values must be finite")
    if not np.isfinite(float(min_std)) or float(min_std) <= 0.0:
        raise ValueError("minimum normalization scale must be positive")

    mean = np.mean(array, axis=0)
    mean[: source_config.rho_latent_dim] = 0.0
    local_mean = mean[source_config.rho_latent_dim :].reshape(
        source_config.local_channels,
        source_config.local_time_count,
        source_config.local_spatial_feature_count,
    )
    local_mean[..., 1:] = 0.0
    centered = array - mean[None, :]
    rho_scale = max(
        float(np.sqrt(np.mean(centered[:, : source_config.rho_latent_dim] ** 2))),
        float(min_std),
    )
    local_scale = max(
        float(np.sqrt(np.mean(centered[:, source_config.rho_latent_dim :] ** 2))),
        float(min_std),
    )
    std = np.empty(int(source_config.active_latent_dim), dtype=np.float64)
    std[: source_config.rho_latent_dim] = rho_scale
    std[source_config.rho_latent_dim :] = local_scale
    return {
        "schema": "vpfp_block_energy_structured_normalization_v1",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min_std": float(min_std),
        "rho_rms_scale": rho_scale,
        "local_rms_scale": local_scale,
        "group_action": "joint_torus_translation_phase_rotation",
        "sample_count": int(array.shape[0]),
        "energy_policy": "preserve_within_block_variance_spectrum",
    }


def fit_channel_energy_structured_normalization(
    values,
    source_config: VPFPStructuredDiffusionConfig,
    *,
    min_std: float = 1.0e-4,
) -> dict[str, Any]:
    """Balance time/channel energy while retaining the spatial spectrum."""

    array = np.asarray(
        _torch().as_tensor(values).detach().cpu(),
        dtype=np.float64,
    )
    if (
        array.ndim != 2
        or int(array.shape[1]) != int(source_config.active_latent_dim)
    ):
        raise ValueError("structured normalization shape mismatch")
    if int(array.shape[0]) < 2:
        raise ValueError("structured normalization requires at least two cases")
    if not np.all(np.isfinite(array)):
        raise ValueError("structured normalization values must be finite")
    if not np.isfinite(float(min_std)) or float(min_std) <= 0.0:
        raise ValueError("minimum normalization scale must be positive")

    mean = np.mean(array, axis=0)
    std = np.empty(int(source_config.active_latent_dim), dtype=np.float64)

    rho_values = array[:, : source_config.rho_latent_dim].reshape(
        array.shape[0],
        source_config.rho_time_count,
        source_config.rho_mode_count,
        2,
    )
    rho_mean = mean[: source_config.rho_latent_dim].reshape(
        source_config.rho_time_count,
        source_config.rho_mode_count,
        2,
    )
    rho_std = std[: source_config.rho_latent_dim].reshape(
        source_config.rho_time_count,
        source_config.rho_mode_count,
        2,
    )
    rho_mean.fill(0.0)
    rho_time_rms = np.sqrt(np.mean(rho_values**2, axis=(0, 2, 3)))
    rho_std[...] = rho_time_rms[:, None, None]

    local_values = array[:, source_config.rho_latent_dim :].reshape(
        array.shape[0],
        source_config.local_channels,
        source_config.local_time_count,
        source_config.local_spatial_feature_count,
    )
    local_mean = mean[source_config.rho_latent_dim :].reshape(
        source_config.local_channels,
        source_config.local_time_count,
        source_config.local_spatial_feature_count,
    )
    local_std = std[source_config.rho_latent_dim :].reshape(
        source_config.local_channels,
        source_config.local_time_count,
        source_config.local_spatial_feature_count,
    )
    local_std[..., 0] = np.std(local_values[..., 0], axis=0)
    if source_config.local_spatial_feature_count > 1:
        local_mean[..., 1:] = 0.0
        local_nonconstant_rms = np.sqrt(
            np.mean(local_values[..., 1:] ** 2, axis=(0, 3))
        )
        local_std[..., 1:] = local_nonconstant_rms[..., None]

    std = np.maximum(std, float(min_std))
    return {
        "schema": "vpfp_channel_energy_structured_normalization_v1",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min_std": float(min_std),
        "group_action": "joint_torus_translation_phase_rotation",
        "sample_count": int(array.shape[0]),
        "energy_policy": (
            "shared_over_spatial_modes_per_time_or_pca_channel"
        ),
    }


class VPFPTorusStructuredCodec(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    @staticmethod
    def build(
        source_config: VPFPStructuredDiffusionConfig | dict[str, Any],
        codec_config: VPFPTorusCodecConfig | dict[str, Any],
    ):
        source = (
            source_config
            if isinstance(source_config, VPFPStructuredDiffusionConfig)
            else VPFPStructuredDiffusionConfig.from_dict(dict(source_config))
        )
        codec = (
            codec_config
            if isinstance(codec_config, VPFPTorusCodecConfig)
            else VPFPTorusCodecConfig.from_dict(dict(codec_config))
        )
        codec.validate(source)
        return VPFPTorusStructuredCodec(source, codec)

    def __init__(
        self,
        source_config: VPFPStructuredDiffusionConfig,
        codec_config: VPFPTorusCodecConfig,
    ):
        super().__init__()
        torch = _torch()
        codec_config.validate(source_config)
        self.source_config = source_config
        self.codec_config = codec_config
        input_channels = (
            source_config.rho_time_count
            + source_config.local_channels * source_config.local_time_count
        )
        shuffled_channels = (
            input_channels * int(codec_config.block_size) ** 2
        )
        self.encoder_input = torch.nn.Conv2d(
            shuffled_channels,
            int(codec_config.width),
            kernel_size=1,
        )
        self.encoder_blocks = torch.nn.ModuleList(
            [
                _PeriodicCodecResidualBlock(
                    int(codec_config.width),
                    dropout=float(codec_config.dropout),
                )
                for _ in range(int(codec_config.residual_blocks))
            ]
        )
        self.encoder_output = torch.nn.Conv2d(
            int(codec_config.width),
            int(codec_config.latent_channels),
            kernel_size=1,
        )
        self.decoder_input = torch.nn.Conv2d(
            int(codec_config.latent_channels),
            int(codec_config.width),
            kernel_size=1,
        )
        self.decoder_blocks = torch.nn.ModuleList(
            [
                _PeriodicCodecResidualBlock(
                    int(codec_config.width),
                    dropout=float(codec_config.dropout),
                )
                for _ in range(int(codec_config.residual_blocks))
            ]
        )
        self.decoder_output = torch.nn.Conv2d(
            int(codec_config.width),
            shuffled_channels,
            kernel_size=1,
        )
        rho_basis, local_basis = _codec_bases(
            torch,
            source_config,
            grid_size=int(codec_config.physical_grid_size),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.register_buffer("rho_basis", rho_basis)
        self.register_buffer("local_basis", local_basis)
        dtype = (
            torch.float64
            if codec_config.dtype == "float64"
            else torch.float32
        )
        self.to(dtype=dtype)

    @property
    def latent_dim(self) -> int:
        return self.codec_config.latent_dim

    def encode(self, active_latent):
        torch = _torch()
        parameter = next(self.parameters())
        values = torch.as_tensor(
            active_latent,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        rho, local = split_active_latent(values, self.source_config)
        rho_fields = torch.einsum("btf,pf->btp", rho, self.rho_basis)
        local_coefficients = local.reshape(
            int(values.shape[0]),
            self.source_config.local_channels
            * self.source_config.local_time_count,
            self.source_config.local_spatial_feature_count,
        )
        local_fields = torch.einsum(
            "bcf,pf->bcp", local_coefficients, self.local_basis
        )
        fields = torch.cat((rho_fields, local_fields), dim=1).reshape(
            int(values.shape[0]),
            -1,
            int(self.codec_config.physical_grid_size),
            int(self.codec_config.physical_grid_size),
        )
        hidden = torch.nn.functional.pixel_unshuffle(
            fields,
            int(self.codec_config.block_size),
        )
        hidden = self.encoder_input(hidden)
        for block in self.encoder_blocks:
            hidden = block(hidden)
        return self.encoder_output(hidden)

    def decode(self, latent_code):
        torch = _torch()
        parameter = next(self.parameters())
        code = torch.as_tensor(
            latent_code,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        if code.ndim != 4:
            raise ValueError("torus codec latent code has an invalid shape")
        expected = (
            int(code.shape[0]),
            int(self.codec_config.latent_channels),
            int(self.codec_config.latent_grid_size),
            int(self.codec_config.latent_grid_size),
        )
        if tuple(code.shape) != expected:
            raise ValueError("torus codec latent code has an invalid shape")
        hidden = self.decoder_input(code)
        for block in self.decoder_blocks:
            hidden = block(hidden)
        shuffled = self.decoder_output(hidden)
        fields = torch.nn.functional.pixel_shuffle(
            shuffled,
            int(self.codec_config.block_size),
        )
        flattened = fields.reshape(int(fields.shape[0]), int(fields.shape[1]), -1)
        rho_fields = flattened[:, : self.source_config.rho_time_count]
        local_fields = flattened[:, self.source_config.rho_time_count :]
        rho = torch.einsum("btp,pf->btf", rho_fields, self.rho_basis)
        local_coefficients = torch.einsum(
            "bcp,pf->bcf", local_fields, self.local_basis
        )
        local = local_coefficients.reshape(
            int(fields.shape[0]),
            self.source_config.local_channels,
            self.source_config.local_time_count,
            self.source_config.local_spatial_feature_count,
        )
        return join_active_latent(rho, local, self.source_config)

    def forward(self, active_latent):
        return self.decode(self.encode(active_latent))


def torus_codec_reconstruction_loss(model, active_latent):
    torch = _torch()
    parameter = next(model.parameters())
    target = torch.as_tensor(
        active_latent,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    code = model.encode(target)
    reconstructed = model.decode(code)
    target_rho, target_local = split_active_latent(
        target,
        model.source_config,
    )
    reconstructed_rho, reconstructed_local = split_active_latent(
        reconstructed,
        model.source_config,
    )
    rho_mse = torch.mean((reconstructed_rho - target_rho).square())
    local_mse = torch.mean((reconstructed_local - target_local).square())
    loss = 0.5 * (rho_mse + local_mse)
    per_case_error = torch.linalg.vector_norm(
        reconstructed - target,
        dim=-1,
    ) / torch.clamp(
        torch.linalg.vector_norm(target, dim=-1),
        min=torch.finfo(target.dtype).eps,
    )
    return loss, {
        "loss": loss,
        "rho_mse": rho_mse,
        "local_mse": local_mse,
        "active_relative_l2_mean": per_case_error.mean(),
        "active_relative_l2_max": per_case_error.max(),
        "latent_rms": code.square().mean().sqrt(),
    }


def save_torus_codec_checkpoint(
    path: str | Path,
    *,
    model: VPFPTorusStructuredCodec,
    train_config: dict[str, Any],
    active_latent_normalization: dict[str, Any],
    step: int,
    optimizer_state: dict[str, Any] | None = None,
    ema_state_dict: dict[str, Any] | None = None,
) -> Path:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_torus_structured_codec_checkpoint_v1",
            "source_config": model.source_config.to_dict(),
            "codec_config": model.codec_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "optimizer_state_dict": optimizer_state,
            "active_latent_normalization": active_latent_normalization,
            "train_config": train_config,
            "step": int(step),
        },
        destination,
    )
    return destination


def load_torus_codec_checkpoint(
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
    if payload.get("schema") != "vpfp_torus_structured_codec_checkpoint_v1":
        raise ValueError("unexpected VPFP torus codec checkpoint schema")
    model = VPFPTorusStructuredCodec.build(
        payload["source_config"],
        payload["codec_config"],
    ).to(device=device)
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload


class _PeriodicCodecResidualBlock(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, width: int, *, dropout: float):
        super().__init__()
        torch = _torch()
        groups = _group_count(int(width))
        self.norm1 = torch.nn.GroupNorm(groups, int(width))
        self.norm2 = torch.nn.GroupNorm(groups, int(width))
        self.conv1 = torch.nn.Conv2d(
            int(width),
            int(width),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.conv2 = torch.nn.Conv2d(
            int(width),
            int(width),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.dropout = torch.nn.Dropout2d(float(dropout))

    def forward(self, values):
        torch = _torch()
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(values)))
        hidden = self.conv2(
            self.dropout(torch.nn.functional.silu(self.norm2(hidden)))
        )
        return (values + hidden) / np.sqrt(2.0)


def _codec_bases(
    torch,
    source_config: VPFPStructuredDiffusionConfig,
    *,
    grid_size: int,
    dtype,
    device,
):
    rho_side = int(round(np.sqrt(2 * source_config.rho_mode_count + 1)))
    rho_max_mode = (rho_side - 1) // 2
    rho_basis = _physical_fourier_basis(
        torch,
        max_mode=rho_max_mode,
        grid_size=int(grid_size),
        include_constant=False,
    ).to(dtype=dtype, device=device)
    local_basis = _physical_fourier_basis(
        torch,
        max_mode=int(source_config.local_spatial_mode),
        grid_size=int(grid_size),
        include_constant=True,
    ).to(dtype=dtype, device=device)
    return rho_basis, local_basis


def _group_count(width: int) -> int:
    for groups in (16, 8, 4, 2):
        if int(width) % groups == 0:
            return groups
    return 1


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
