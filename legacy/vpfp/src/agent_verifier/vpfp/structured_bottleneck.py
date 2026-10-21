from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


def fit_dual_pca(values, *, max_rank: int) -> dict[str, Any]:
    """Fit exact PCA through the sample Gram matrix when cases << features."""

    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2:
        raise ValueError("PCA values must be a matrix")
    case_count, feature_count = (int(value) for value in tensor.shape)
    if case_count < 2 or feature_count < 1:
        raise ValueError("PCA requires at least two cases and one feature")
    if not tensor.is_floating_point():
        raise ValueError("PCA values must use a floating dtype")
    requested_rank = int(max_rank)
    if requested_rank < 1 or requested_rank > min(case_count - 1, feature_count):
        raise ValueError("PCA max_rank is outside the centered matrix rank bound")
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        raise ValueError("PCA values must be finite")

    mean = tensor.mean(dim=0)
    centered = tensor - mean
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    maximum = torch.clamp(eigenvalues[0], min=0.0)
    singular_tolerance = (
        torch.finfo(tensor.dtype).eps
        * float(max(case_count, feature_count))
        * torch.sqrt(maximum)
    )
    eigenvalue_tolerance = torch.maximum(
        singular_tolerance.square(),
        torch.finfo(tensor.dtype).eps * float(case_count) * maximum,
    )
    available_rank = int(
        torch.count_nonzero(eigenvalues > eigenvalue_tolerance).item()
    )
    retained_rank = min(requested_rank, available_rank)
    if retained_rank < 1:
        raise ValueError("PCA fit has no numerically nonzero component")

    retained_values = eigenvalues[:retained_rank]
    retained_vectors = eigenvectors[:, :retained_rank]
    singular_values = torch.sqrt(retained_values)
    components = (centered.T @ retained_vectors) / singular_values[None, :]
    gram_error = torch.max(
        torch.abs(
            components.T @ components
            - torch.eye(
                retained_rank, dtype=tensor.dtype, device=tensor.device
            )
        )
    )
    total_variance = centered.square().sum() / float(case_count - 1)
    explained_variance = retained_values / float(case_count - 1)
    explained_ratio = explained_variance / torch.clamp(
        total_variance, min=torch.finfo(tensor.dtype).tiny
    )
    return {
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
        "available_rank": available_rank,
        "retained_rank": retained_rank,
        "orthogonality_max_abs": gram_error,
        "case_count": case_count,
        "feature_count": feature_count,
    }


def reconstruct_from_dual_pca(values, basis: dict[str, Any], *, rank: int):
    torch = _torch()
    tensor = torch.as_tensor(values)
    mean = torch.as_tensor(
        basis["mean"], dtype=tensor.dtype, device=tensor.device
    )
    components = torch.as_tensor(
        basis["components"], dtype=tensor.dtype, device=tensor.device
    )
    if tensor.ndim != 2 or mean.shape != tensor.shape[1:]:
        raise ValueError("PCA reconstruction values do not match the mean")
    if components.ndim != 2 or int(components.shape[0]) != int(tensor.shape[1]):
        raise ValueError("PCA components do not match the feature dimension")
    selected_rank = int(rank)
    if selected_rank < 1 or selected_rank > int(components.shape[1]):
        raise ValueError("PCA reconstruction rank is unavailable")
    selected = components[:, :selected_rank]
    scores = (tensor - mean) @ selected
    return mean + scores @ selected.T


def fit_shared_local_time_channel_pca(
    values,
    config,
    *,
    max_rank: int,
) -> dict[str, Any]:
    """Fit one origin-preserving PCA across local channel-time vectors.

    The same linear map is applied to every spatial Fourier coefficient.  It
    therefore commutes with each torus-translation phase rotation.
    """

    from agent_verifier.vpfp.structured_diffusion import split_active_latent

    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or int(tensor.shape[1]) != config.active_latent_dim:
        raise ValueError("shared local PCA values do not match structured config")
    if not tensor.is_floating_point():
        raise ValueError("shared local PCA values must use a floating dtype")
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        raise ValueError("shared local PCA values must be finite")
    _, local = split_active_latent(tensor, config)
    feature_count = int(config.local_channels * config.local_time_count)
    requested_rank = int(max_rank)
    if requested_rank < 1 or requested_rank > feature_count:
        raise ValueError("shared local PCA rank is outside the channel-time bound")

    observations = local.permute(0, 3, 1, 2).reshape(-1, feature_count)
    observation_count = int(observations.shape[0])
    covariance = observations.T @ observations / float(observation_count)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    maximum = torch.clamp(eigenvalues[0], min=0.0)
    tolerance = (
        torch.finfo(tensor.dtype).eps
        * float(feature_count)
        * maximum
    )
    available_rank = int(torch.count_nonzero(eigenvalues > tolerance).item())
    retained_rank = min(requested_rank, available_rank)
    if retained_rank < 1:
        raise ValueError("shared local PCA fit has no nonzero component")

    components = eigenvectors[:, :retained_rank]
    orthogonality_error = torch.max(
        torch.abs(
            components.T @ components
            - torch.eye(
                retained_rank,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        )
    )
    total_energy = torch.clamp(
        torch.trace(covariance),
        min=torch.finfo(tensor.dtype).tiny,
    )
    retained_values = eigenvalues[:retained_rank]
    return {
        "components": components,
        "explained_energy": retained_values,
        "explained_energy_ratio": retained_values / total_energy,
        "available_rank": available_rank,
        "retained_rank": retained_rank,
        "orthogonality_max_abs": orthogonality_error,
        "case_count": int(tensor.shape[0]),
        "observation_count": observation_count,
        "feature_count": feature_count,
        "centering": "origin_preserving",
        "group_action": "shared_torus_fourier_phase_rotation",
    }


def reconstruct_from_shared_local_time_channel_pca(
    values,
    basis: dict[str, Any],
    config,
    *,
    rank: int,
):
    from agent_verifier.vpfp.structured_diffusion import (
        join_active_latent,
        split_active_latent,
    )

    torch = _torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or int(tensor.shape[1]) != config.active_latent_dim:
        raise ValueError("shared local PCA values do not match structured config")
    components = torch.as_tensor(
        basis["components"],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    feature_count = int(config.local_channels * config.local_time_count)
    if components.ndim != 2 or tuple(components.shape[:1]) != (feature_count,):
        raise ValueError("shared local PCA components do not match channel-time")
    selected_rank = int(rank)
    if selected_rank < 1 or selected_rank > int(components.shape[1]):
        raise ValueError("shared local PCA reconstruction rank is unavailable")

    rho, local = split_active_latent(tensor, config)
    observations = local.permute(0, 3, 1, 2).reshape(-1, feature_count)
    selected = components[:, :selected_rank]
    reconstructed = observations @ selected @ selected.T
    reconstructed_local = reconstructed.reshape(
        int(tensor.shape[0]),
        config.local_spatial_feature_count,
        config.local_channels,
        config.local_time_count,
    ).permute(0, 2, 3, 1)
    return join_active_latent(rho, reconstructed_local, config)


def encode_shared_local_time_channel(
    values,
    basis: dict[str, Any],
    source_config,
    compressed_config,
    *,
    rank: int,
):
    from agent_verifier.vpfp.structured_diffusion import (
        join_active_latent,
        split_active_latent,
    )

    _validate_compressed_layout(source_config, compressed_config, rank=int(rank))
    torch = _torch()
    tensor = torch.as_tensor(values)
    rho, local = split_active_latent(tensor, source_config)
    components = torch.as_tensor(
        basis["components"], dtype=tensor.dtype, device=tensor.device
    )[:, : int(rank)]
    observations = local.permute(0, 3, 1, 2).reshape(
        int(tensor.shape[0]),
        source_config.local_spatial_feature_count,
        source_config.local_channels * source_config.local_time_count,
    )
    scores = observations @ components
    compressed_local = scores.permute(0, 2, 1).unsqueeze(2)
    return join_active_latent(rho, compressed_local, compressed_config)


def decode_shared_local_time_channel(
    values,
    basis: dict[str, Any],
    compressed_config,
    target_config,
    *,
    rank: int,
):
    from agent_verifier.vpfp.structured_diffusion import (
        join_active_latent,
        split_active_latent,
    )

    _validate_compressed_layout(target_config, compressed_config, rank=int(rank))
    torch = _torch()
    tensor = torch.as_tensor(values)
    rho, compressed_local = split_active_latent(tensor, compressed_config)
    components = torch.as_tensor(
        basis["components"], dtype=tensor.dtype, device=tensor.device
    )[:, : int(rank)]
    scores = compressed_local[:, :, 0, :].permute(0, 2, 1)
    observations = scores @ components.T
    local = observations.reshape(
        int(tensor.shape[0]),
        target_config.local_spatial_feature_count,
        target_config.local_channels,
        target_config.local_time_count,
    ).permute(0, 2, 3, 1)
    return join_active_latent(rho, local, target_config)


def _validate_compressed_layout(source_config, compressed_config, *, rank: int):
    if int(rank) < 1:
        raise ValueError("shared local bottleneck rank must be positive")
    if (
        source_config.rho_time_count != compressed_config.rho_time_count
        or source_config.rho_feature_count != compressed_config.rho_feature_count
        or source_config.local_spatial_feature_count
        != compressed_config.local_spatial_feature_count
    ):
        raise ValueError("shared local bottleneck layouts are incompatible")
    if (
        compressed_config.local_time_count != 1
        or compressed_config.local_channels != int(rank)
    ):
        raise ValueError("compressed local layout must have shape [rank,1,space]")


def load_shared_local_bottleneck_contract(checkpoint: dict[str, Any]):
    """Load and validate the optional shared channel-time PCA contract."""

    from agent_verifier.vpfp.structured_diffusion import (
        VPFPStructuredDiffusionConfig,
    )

    train_config = dict(checkpoint.get("train_config") or {})
    metadata = dict(train_config.get("bottleneck") or {})
    if not bool(metadata.get("enabled")):
        return None
    if metadata.get("strategy") != "shared_local_time_channel":
        raise ValueError("unsupported structured bottleneck strategy")
    rank = int(metadata.get("rank") or 0)
    basis_path = Path(str(metadata.get("basis") or "")).resolve()
    if rank < 1 or not basis_path.is_file():
        raise ValueError("structured bottleneck has missing rank or basis")
    torch = _torch()
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)
    if basis.get("schema") != "vpfp_structured_pca_basis_v2":
        raise ValueError("unexpected structured bottleneck basis schema")
    if basis.get("strategy") != "shared_local_time_channel":
        raise ValueError("structured bottleneck basis strategy mismatch")
    basis_decoder = Path(str(basis.get("decoder_checkpoint") or "")).resolve()
    checkpoint_decoder = Path(
        str(checkpoint.get("decoder_checkpoint") or "")
    ).resolve()
    if basis_decoder != checkpoint_decoder:
        raise ValueError("structured bottleneck decoder binding changed")
    if rank > int(basis.get("retained_rank", 0)):
        raise ValueError("structured bottleneck rank is unavailable")
    canonical_config = VPFPStructuredDiffusionConfig.from_dict(
        train_config.get("canonical_structured_config") or {}
    )
    model_config = VPFPStructuredDiffusionConfig.from_dict(
        checkpoint.get("model_config") or {}
    )
    canonical_config.validate()
    model_config.validate()
    _validate_compressed_layout(canonical_config, model_config, rank=rank)
    components = np.asarray(basis.get("components"), dtype=np.float64)
    expected_features = int(
        canonical_config.local_channels * canonical_config.local_time_count
    )
    if (
        components.ndim != 2
        or components.shape[0] != expected_features
        or components.shape[1] < rank
        or not np.all(np.isfinite(components))
    ):
        raise ValueError("structured bottleneck components are invalid")
    normalization = basis.get("active_latent_normalization")
    if not isinstance(normalization, dict):
        raise ValueError("structured bottleneck normalization is missing")
    mean = np.asarray(normalization.get("mean"), dtype=np.float64)
    std = np.asarray(normalization.get("std"), dtype=np.float64)
    expected = (int(canonical_config.active_latent_dim),)
    if (
        mean.shape != expected
        or std.shape != expected
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError("structured bottleneck normalization is invalid")
    return {
        "basis": basis,
        "basis_path": basis_path,
        "rank": rank,
        "model_config": model_config,
        "canonical_config": canonical_config,
        "canonical_normalization": normalization,
    }


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for structured PCA") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
