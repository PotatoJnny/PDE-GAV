from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from agent_verifier.vpfp.shared_global_latent import (
    denormalize_shared_latent,
    fit_shared_latent_normalization,
    normalize_shared_latent,
)
from agent_verifier.vpfp.shared_global_latent_diffusion import (
    VPFPSharedGlobalLatentEDM,
    VPFPSharedGlobalLatentEDMConfig,
)
from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
    _physical_fourier_basis,
    _rho_max_mode,
    join_active_latent,
    split_active_latent,
)


LOSSLESS_FIELD_CHECKPOINT_SCHEMA = (
    "vpfp_lossless_shared_solution_field_edm_checkpoint_v1"
)
LOSSLESS_FIELD_REPRESENTATION = "vpfp_lossless_shared_periodic_solution_field_v1"
NUMERICALLY_LOSSLESS_RELATIVE_L2_TOLERANCE = 5.0e-4
LOSSLESS_FIELD_ZCA_NORMALIZATION_SCHEMA = (
    "vpfp_lossless_shared_field_joint_zca_normalization_v1"
)
LOSSLESS_FIELD_BLOCK_ZCA_NORMALIZATION_SCHEMA = (
    "vpfp_lossless_shared_field_block_zca_normalization_v1"
)
LOSSLESS_GLOBAL_RESIDUAL_SCHEMA = (
    "vpfp_lossless_shared_field_global_residual_v1"
)


@dataclass(frozen=True)
class VPFPLosslessSharedFieldConfig:
    """Lossless joint periodic field for rho and nonlinear micro codes."""

    latent_config: VPFPStructuredDiffusionConfig
    grid_size: int = 13
    dtype: str = "float32"
    representation: str = LOSSLESS_FIELD_REPRESENTATION

    @property
    def field_channels(self) -> int:
        latent = self.latent_config
        return int(latent.rho_time_count) + int(
            latent.local_channels * latent.local_time_count
        )

    def validate(self) -> None:
        self.latent_config.validate()
        if int(self.grid_size) < 1:
            raise ValueError("lossless shared field grid size must be positive")
        maximum_mode = max(
            _rho_max_mode(self.latent_config),
            int(self.latent_config.local_spatial_mode),
        )
        if int(self.grid_size) < 2 * maximum_mode + 1:
            raise ValueError("lossless shared field grid aliases represented modes")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("lossless shared field dtype is unsupported")
        if str(self.representation) != LOSSLESS_FIELD_REPRESENTATION:
            raise ValueError("unexpected lossless shared field representation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_config": self.latent_config.to_dict(),
            "grid_size": int(self.grid_size),
            "dtype": str(self.dtype),
            "representation": str(self.representation),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPLosslessSharedFieldConfig":
        values = dict(payload)
        values["latent_config"] = VPFPStructuredDiffusionConfig.from_dict(
            dict(values["latent_config"])
        )
        config = cls(**values)
        config.validate()
        return config


class VPFPLosslessSharedFieldTransform(nn.Module):
    """Orthogonal Fourier synthesis/analysis with no learned bottleneck."""

    @staticmethod
    def build(
        config: VPFPLosslessSharedFieldConfig | Mapping[str, Any],
    ) -> "VPFPLosslessSharedFieldTransform":
        parsed = (
            config
            if isinstance(config, VPFPLosslessSharedFieldConfig)
            else VPFPLosslessSharedFieldConfig.from_dict(config)
        )
        parsed.validate()
        return VPFPLosslessSharedFieldTransform(parsed)

    def __init__(self, config: VPFPLosslessSharedFieldConfig):
        super().__init__()
        config.validate()
        self.config = config
        latent = config.latent_config
        rho_basis = _physical_fourier_basis(
            torch,
            max_mode=_rho_max_mode(latent),
            grid_size=int(config.grid_size),
            include_constant=False,
        )
        micro_basis = _physical_fourier_basis(
            torch,
            max_mode=int(latent.local_spatial_mode),
            grid_size=int(config.grid_size),
            include_constant=True,
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.register_buffer("rho_basis", rho_basis.to(dtype=dtype))
        self.register_buffer("micro_basis", micro_basis.to(dtype=dtype))

    def active_to_field(self, active_latent) -> torch.Tensor:
        values = torch.as_tensor(active_latent)
        latent = self.config.latent_config
        expected = (int(values.shape[0]), int(latent.active_latent_dim))
        if tuple(values.shape) != expected:
            raise ValueError("active latent shape does not match lossless field")
        rho, micro = split_active_latent(values, latent)
        rho_field = torch.einsum(
            "btf,gf->btg", rho, self.rho_basis.to(rho)
        )
        micro_field = torch.einsum(
            "bctf,gf->bctg", micro, self.micro_basis.to(micro)
        )
        batch = int(values.shape[0])
        side = int(self.config.grid_size)
        return torch.cat(
            (
                rho_field.reshape(
                    batch, int(latent.rho_time_count), side, side
                ),
                micro_field.reshape(
                    batch,
                    int(latent.local_channels * latent.local_time_count),
                    side,
                    side,
                ),
            ),
            dim=1,
        )

    def field_to_active(self, field) -> torch.Tensor:
        values = torch.as_tensor(field)
        latent = self.config.latent_config
        side = int(self.config.grid_size)
        expected = (
            int(values.shape[0]),
            int(self.config.field_channels),
            side,
            side,
        )
        if tuple(values.shape) != expected:
            raise ValueError("lossless shared field has the wrong shape")
        batch = int(values.shape[0])
        flat = values.reshape(batch, int(values.shape[1]), side * side)
        rho_field = flat[:, : int(latent.rho_time_count)]
        micro_field = flat[:, int(latent.rho_time_count) :].reshape(
            batch,
            int(latent.local_channels),
            int(latent.local_time_count),
            side * side,
        )
        rho = torch.einsum(
            "btg,gf->btf", rho_field, self.rho_basis.to(rho_field)
        )
        micro = torch.einsum(
            "bctg,gf->bctf", micro_field, self.micro_basis.to(micro_field)
        )
        return join_active_latent(rho, micro, latent)

    def roundtrip(self, active_latent) -> torch.Tensor:
        return self.field_to_active(self.active_to_field(active_latent))


def fit_lossless_shared_field_normalization(
    field,
    *,
    mode: str = "scale",
    shrinkage: float = 0.05,
    eigenvalue_floor_fraction: float = 0.01,
    macro_channels: int | None = None,
) -> dict[str, Any]:
    """Fit a translation-invariant, invertible answer-coordinate transform."""

    values = torch.as_tensor(field).detach()
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("lossless field must have shape [batch, channel, x, y]")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("lossless field normalization requires finite values")
    if mode not in {"scale", "joint_zca", "block_zca"}:
        raise ValueError("lossless field normalization mode is unsupported")
    base = fit_shared_latent_normalization(values)
    if mode == "scale":
        return base
    if not 0.0 <= float(shrinkage) < 1.0:
        raise ValueError("joint ZCA shrinkage must lie in [0,1)")
    if not 0.0 < float(eigenvalue_floor_fraction) < 1.0:
        raise ValueError("joint ZCA eigenvalue floor must lie in (0,1)")

    scaled = normalize_shared_latent(values, base)
    samples = scaled.permute(0, 2, 3, 1).reshape(
        -1, int(values.shape[1])
    )
    channel_count = int(values.shape[1])
    if mode == "block_zca":
        boundary = int(macro_channels or 0)
        if not 0 < boundary < channel_count:
            raise ValueError(
                "block ZCA needs a strict macro/micro channel boundary"
            )
        blocks = ((0, boundary), (boundary, channel_count))
    else:
        boundary = None
        blocks = ((0, channel_count),)
    whitening = torch.zeros(
        channel_count,
        channel_count,
        dtype=values.dtype,
        device=values.device,
    )
    inverse = torch.zeros_like(whitening)
    block_diagnostics = []
    for start, stop in blocks:
        block_whitening, block_inverse, diagnostics = _fit_zca_block(
            samples[:, start:stop],
            shrinkage=float(shrinkage),
            eigenvalue_floor_fraction=float(eigenvalue_floor_fraction),
        )
        whitening[start:stop, start:stop] = block_whitening
        inverse[start:stop, start:stop] = block_inverse
        block_diagnostics.append(
            {"channel_start": start, "channel_stop": stop, **diagnostics}
        )
    identity = torch.eye(
        channel_count, dtype=values.dtype, device=values.device
    )
    inverse_error = torch.max(torch.abs(inverse @ whitening - identity))
    if not bool(
        torch.all(torch.isfinite(whitening)).item()
        and torch.all(torch.isfinite(inverse)).item()
    ):
        raise ValueError("ZCA transform is not finite")
    return {
        "schema": (
            LOSSLESS_FIELD_BLOCK_ZCA_NORMALIZATION_SCHEMA
            if mode == "block_zca"
            else LOSSLESS_FIELD_ZCA_NORMALIZATION_SCHEMA
        ),
        "base_normalization": base,
        "whitening": whitening.detach().cpu(),
        "inverse": inverse.detach().cpu(),
        "translation_invariant_fit": True,
        "mixes_macro_and_micro_channels": mode == "joint_zca",
        "macro_micro_block_diagonal": mode == "block_zca",
        "macro_channel_count": boundary,
        "learned": False,
        "invertible": True,
        "channel_count": channel_count,
        "sample_count": int(samples.shape[0]),
        "shrinkage": float(shrinkage),
        "eigenvalue_floor_fraction": float(eigenvalue_floor_fraction),
        "inverse_construction": "numerical_inverse_of_stored_whitening",
        "diagnostics": {
            "inverse_operator_max_abs_error": float(
                inverse_error.detach().cpu()
            ),
            "blocks": block_diagnostics,
        },
    }


def _fit_zca_block(
    samples: torch.Tensor,
    *,
    shrinkage: float,
    eigenvalue_floor_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    covariance = samples.T @ samples / float(samples.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    mean_variance = torch.mean(torch.diagonal(covariance))
    epsilon = torch.finfo(samples.dtype).eps
    if not bool(torch.isfinite(mean_variance).item()) or float(
        mean_variance
    ) <= 0.0:
        raise ValueError("ZCA covariance has no positive finite variance")
    identity = torch.eye(
        int(samples.shape[1]), dtype=samples.dtype, device=samples.device
    )
    regularized = (
        (1.0 - shrinkage) * covariance
        + shrinkage * mean_variance * identity
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
    floor = torch.clamp(
        eigenvalue_floor_fraction * mean_variance,
        min=epsilon,
    )
    eigenvalues = torch.clamp(eigenvalues, min=floor)
    whitening = (
        eigenvectors * torch.rsqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    inverse = torch.linalg.inv(
        whitening.to(dtype=torch.float64)
    ).to(dtype=samples.dtype)
    raw_eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probability = raw_eigenvalues / raw_eigenvalues.sum().clamp_min(epsilon)
    effective_rank = torch.exp(
        -torch.sum(probability * torch.log(probability.clamp_min(epsilon)))
    )
    return whitening, inverse, {
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


def normalize_lossless_shared_field(
    field, normalization: Mapping[str, Any]
) -> torch.Tensor:
    values = torch.as_tensor(field)
    if normalization.get("schema") not in {
        LOSSLESS_FIELD_ZCA_NORMALIZATION_SCHEMA,
        LOSSLESS_FIELD_BLOCK_ZCA_NORMALIZATION_SCHEMA,
    }:
        return normalize_shared_latent(values, normalization)
    _validate_joint_zca_normalization(values, normalization)
    scaled = normalize_shared_latent(
        values, normalization["base_normalization"]
    )
    whitening = _normalization_matrix(normalization, "whitening", values)
    return torch.einsum("bchw,dc->bdhw", scaled, whitening)


def denormalize_lossless_shared_field(
    field, normalization: Mapping[str, Any]
) -> torch.Tensor:
    values = torch.as_tensor(field)
    if normalization.get("schema") not in {
        LOSSLESS_FIELD_ZCA_NORMALIZATION_SCHEMA,
        LOSSLESS_FIELD_BLOCK_ZCA_NORMALIZATION_SCHEMA,
    }:
        return denormalize_shared_latent(values, normalization)
    _validate_joint_zca_normalization(values, normalization)
    inverse = _normalization_matrix(normalization, "inverse", values)
    scaled = torch.einsum("bchw,dc->bdhw", values, inverse)
    return denormalize_shared_latent(
        scaled, normalization["base_normalization"]
    )


def project_normalized_lossless_shared_field(
    field,
    *,
    transform: VPFPLosslessSharedFieldTransform,
    normalization: Mapping[str, Any],
) -> torch.Tensor:
    """Project a normalized field onto the exact continuous-answer image."""

    values = torch.as_tensor(field)
    _validate_answer_subspace_normalization(
        values, transform=transform, normalization=normalization
    )
    physical = denormalize_lossless_shared_field(values, normalization)
    projected = transform.active_to_field(
        transform.field_to_active(physical)
    )
    return normalize_lossless_shared_field(projected, normalization)


def sample_normalized_lossless_shared_field_subspace_noise(
    template,
    *,
    transform: VPFPLosslessSharedFieldTransform,
    normalization: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw unit point-variance Gaussian noise only in answer coordinates."""

    values = torch.as_tensor(template)
    _validate_answer_subspace_normalization(
        values, transform=transform, normalization=normalization
    )
    latent = transform.config.latent_config
    batch = int(values.shape[0])
    macro_channels = int(latent.rho_time_count)
    micro_channels = int(transform.config.field_channels) - macro_channels
    point_count = int(transform.config.grid_size) ** 2

    def synthesize(channels: int, basis: torch.Tensor) -> torch.Tensor:
        feature_count = int(basis.shape[1])
        coefficients = torch.randn(
            (batch, channels, feature_count),
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )
        scale = (float(point_count) / float(feature_count)) ** 0.5
        flat = torch.einsum(
            "bcf,gf->bcg", coefficients, basis.to(values)
        )
        return scale * flat.reshape(
            batch,
            channels,
            int(transform.config.grid_size),
            int(transform.config.grid_size),
        )

    return torch.cat(
        (
            synthesize(macro_channels, transform.rho_basis),
            synthesize(micro_channels, transform.micro_basis),
        ),
        dim=1,
    )


def _validate_answer_subspace_normalization(
    values: torch.Tensor,
    *,
    transform: VPFPLosslessSharedFieldTransform,
    normalization: Mapping[str, Any],
) -> None:
    expected = (
        int(values.shape[0]),
        int(transform.config.field_channels),
        int(transform.config.grid_size),
        int(transform.config.grid_size),
    )
    if tuple(values.shape) != expected or not values.is_floating_point():
        raise ValueError(
            "normalized lossless field shape does not match transform"
        )
    schema = normalization.get("schema")
    if schema == LOSSLESS_FIELD_ZCA_NORMALIZATION_SCHEMA:
        raise ValueError(
            "joint ZCA mixes answer groups with different Fourier supports"
        )
    if schema == LOSSLESS_FIELD_BLOCK_ZCA_NORMALIZATION_SCHEMA:
        expected_macro = int(
            transform.config.latent_config.rho_time_count
        )
        if int(normalization.get("macro_channel_count", -1)) != expected_macro:
            raise ValueError("block ZCA macro answer boundary changed")


def _validate_joint_zca_normalization(
    values: torch.Tensor, normalization: Mapping[str, Any]
) -> None:
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("lossless field must have shape [batch, channel, x, y]")
    channels = int(values.shape[1])
    if int(normalization.get("channel_count", -1)) != channels:
        raise ValueError("joint ZCA channel layout changed")
    for name in ("whitening", "inverse"):
        matrix = torch.as_tensor(normalization[name])
        if tuple(matrix.shape) != (channels, channels):
            raise ValueError(f"joint ZCA {name} matrix has the wrong shape")


def _normalization_matrix(
    normalization: Mapping[str, Any], name: str, values: torch.Tensor
) -> torch.Tensor:
    return torch.as_tensor(
        normalization[name], dtype=values.dtype, device=values.device
    )


def fit_lossless_global_residual_decomposition(
    normalized_fields,
    *,
    rank: int,
) -> dict[str, Any]:
    """Fit shared case-level coordinates while retaining an exact residual."""

    values = torch.as_tensor(normalized_fields).detach()
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("global-residual fit expects [case, channel, x, y]")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("global-residual fit requires finite answer fields")
    case_count = int(values.shape[0])
    selected_rank = int(rank)
    if not 0 < selected_rank < case_count:
        raise ValueError("global-residual rank must be below the fit case count")

    mean = values.mean(dim=0)
    centered = (values - mean).reshape(case_count, -1)
    gram = centered @ centered.T / float(case_count - 1)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    retained = eigenvalues[:selected_rank]
    floor = torch.finfo(values.dtype).eps * eigenvalues[0].clamp_min(1.0)
    if not bool(torch.all(retained > floor).item()):
        raise ValueError("global-residual fit requested a numerically null mode")
    components = eigenvectors[:, :selected_rank].T @ centered
    components = components / torch.sqrt(
        float(case_count - 1) * retained
    )[:, None]
    identity = torch.eye(
        selected_rank, dtype=values.dtype, device=values.device
    )
    orthogonality_error = torch.max(
        torch.abs(components @ components.T - identity)
    )
    total_variance = eigenvalues.clamp_min(0.0).sum().clamp_min(floor)
    return {
        "schema": LOSSLESS_GLOBAL_RESIDUAL_SCHEMA,
        "field_shape": [int(value) for value in values.shape[1:]],
        "rank": selected_rank,
        "fit_case_count": case_count,
        "mean": mean.cpu(),
        "components": components.reshape(
            selected_rank, *values.shape[1:]
        ).cpu(),
        "code_scale": torch.sqrt(retained).cpu(),
        "learned": False,
        "invertible_with_residual": True,
        "truncates_hidden_answer": False,
        "answer_first_semantics": (
            "joint_solution_coordinates_plus_exact_orthogonal_residual"
        ),
        "diagnostics": {
            "captured_variance_fraction": float(
                retained.sum() / total_variance
            ),
            "component_orthogonality_max_abs_error": float(
                orthogonality_error
            ),
            "leading_explained_variance": float(retained[0]),
            "trailing_explained_variance": float(retained[-1]),
        },
    }


def fit_lossless_block_global_residual_decomposition(
    normalized_fields,
    *,
    macro_channels: int,
    macro_rank: int,
    micro_rank: int,
) -> dict[str, Any]:
    """Fit balanced macro/micro answer coordinates with an exact residual."""

    values = torch.as_tensor(normalized_fields).detach()
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("block global fit expects [case, channel, x, y]")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("block global fit requires finite answer fields")
    case_count, channel_count = int(values.shape[0]), int(values.shape[1])
    boundary = int(macro_channels)
    selected_macro_rank = int(macro_rank)
    selected_micro_rank = int(micro_rank)
    if not 0 < boundary < channel_count:
        raise ValueError("block global fit needs a strict macro boundary")
    if min(selected_macro_rank, selected_micro_rank) < 1:
        raise ValueError("block global ranks must be positive")
    if max(selected_macro_rank, selected_micro_rank) >= case_count:
        raise ValueError("block global ranks must be below the fit case count")

    mean = values.mean(dim=0)
    centered_field = values - mean
    macro = centered_field[:, :boundary].reshape(case_count, -1)
    micro = centered_field[:, boundary:].reshape(case_count, -1)
    macro_fit = _fit_lossless_pca_block(macro, selected_macro_rank)
    micro_fit = _fit_lossless_pca_block(micro, selected_micro_rank)

    rank = selected_macro_rank + selected_micro_rank
    components = torch.zeros(
        rank,
        *values.shape[1:],
        dtype=values.dtype,
        device=values.device,
    )
    components[:selected_macro_rank, :boundary] = macro_fit[
        "components"
    ].reshape(selected_macro_rank, boundary, *values.shape[2:])
    components[selected_macro_rank:, boundary:] = micro_fit[
        "components"
    ].reshape(
        selected_micro_rank,
        channel_count - boundary,
        *values.shape[2:],
    )
    flat_components = components.reshape(rank, -1)
    centered = centered_field.reshape(case_count, -1)
    scores = centered @ flat_components.T
    robust = _fit_robust_asinh_code(scores)
    code = (
        torch.asinh(
            (scores - robust["score_center"]) / robust["score_scale"]
        )
        - robust["code_center"]
    ) / robust["code_scale"]
    identity = torch.eye(rank, dtype=values.dtype, device=values.device)
    orthogonality_error = torch.max(
        torch.abs(flat_components @ flat_components.T - identity)
    )
    total_variance = (
        macro_fit["total_variance"] + micro_fit["total_variance"]
    ).clamp_min(torch.finfo(values.dtype).eps)
    retained_variance = (
        macro_fit["retained"].sum() + micro_fit["retained"].sum()
    )
    return {
        "schema": LOSSLESS_GLOBAL_RESIDUAL_SCHEMA,
        "field_shape": [int(value) for value in values.shape[1:]],
        "rank": rank,
        "fit_case_count": case_count,
        "mean": mean.cpu(),
        "components": components.cpu(),
        "code_transform": "robust_asinh",
        "score_center": robust["score_center"].cpu(),
        "score_scale": robust["score_scale"].cpu(),
        "code_center": robust["code_center"].cpu(),
        "code_scale": robust["code_scale"].cpu(),
        "coordinate_layout": "block_balanced_macro_micro_pca",
        "macro_channels": boundary,
        "macro_rank": selected_macro_rank,
        "micro_rank": selected_micro_rank,
        "component_groups": (
            ["macro"] * selected_macro_rank
            + ["micro"] * selected_micro_rank
        ),
        "learned": False,
        "invertible_with_residual": True,
        "truncates_hidden_answer": False,
        "answer_first_semantics": (
            "balanced_solution_coordinates_plus_exact_orthogonal_residual"
        ),
        "diagnostics": {
            "captured_variance_fraction": float(
                retained_variance / total_variance
            ),
            "macro_captured_variance_fraction": float(
                macro_fit["retained"].sum()
                / macro_fit["total_variance"]
            ),
            "micro_captured_variance_fraction": float(
                micro_fit["retained"].sum()
                / micro_fit["total_variance"]
            ),
            "component_orthogonality_max_abs_error": float(
                orthogonality_error
            ),
            "fit_code_rms": float(torch.sqrt(torch.mean(code.square()))),
            "fit_code_abs_q99": float(torch.quantile(torch.abs(code), 0.99)),
            "score_scale_minimum": float(robust["score_scale"].min()),
            "score_scale_maximum": float(robust["score_scale"].max()),
        },
    }


def _fit_lossless_pca_block(centered: torch.Tensor, rank: int):
    case_count = int(centered.shape[0])
    gram = centered @ centered.T / float(case_count - 1)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    retained = eigenvalues[: int(rank)]
    floor = torch.finfo(centered.dtype).eps * eigenvalues[0].clamp_min(1.0)
    if not bool(torch.all(retained > floor).item()):
        raise ValueError("block global fit requested a numerically null mode")
    components = eigenvectors[:, : int(rank)].T @ centered
    components = components / torch.sqrt(
        float(case_count - 1) * retained
    )[:, None]
    return {
        "components": components,
        "retained": retained,
        "total_variance": eigenvalues.clamp_min(0.0).sum().clamp_min(floor),
    }


def _fit_robust_asinh_code(scores: torch.Tensor):
    score_center = torch.median(scores, dim=0).values
    centered = scores - score_center
    rms = torch.sqrt(torch.mean(centered.square(), dim=0))
    mad = 1.4826 * torch.median(torch.abs(centered), dim=0).values
    floor = torch.finfo(scores.dtype).eps * rms.max().clamp_min(1.0)
    score_scale = torch.maximum(mad, 0.05 * rms).clamp_min(floor)
    transformed = torch.asinh(centered / score_scale)
    code_center = transformed.mean(dim=0)
    code_scale = torch.sqrt(
        torch.mean((transformed - code_center).square(), dim=0)
    ).clamp_min(torch.finfo(scores.dtype).eps)
    return {
        "score_center": score_center,
        "score_scale": score_scale,
        "code_center": code_center,
        "code_scale": code_scale,
    }


def lossless_global_code_tail_guard_config(
    decomposition: Mapping[str, Any],
    *,
    threshold_multiplier: float = 1.25,
    transition_multiplier: float = 0.50,
) -> dict[str, float]:
    """Register a smooth tail guard from answer-coordinate fit statistics."""

    if str(decomposition.get("code_transform")) != "robust_asinh":
        raise ValueError("global answer-code tail guard requires robust_asinh")
    if float(threshold_multiplier) <= 1.0 or float(transition_multiplier) <= 0.0:
        raise ValueError("global answer-code tail guard multipliers are invalid")
    diagnostics = dict(decomposition.get("diagnostics") or {})
    fit_q99 = float(diagnostics.get("fit_code_abs_q99", 0.0))
    if not math.isfinite(fit_q99) or fit_q99 <= 0.0:
        raise ValueError("global answer-code fit q99 is unavailable")
    threshold = float(threshold_multiplier) * fit_q99
    transition = float(transition_multiplier) * fit_q99
    return {
        "fit_absolute_q99": fit_q99,
        "identity_threshold": threshold,
        "transition_width": transition,
        "absolute_bound": threshold + transition,
        "equation_dynamics_used": False,
    }


def soft_bound_lossless_global_codes(
    global_codes,
    decomposition: Mapping[str, Any],
    *,
    threshold_multiplier: float = 1.25,
    transition_multiplier: float = 0.50,
) -> torch.Tensor:
    """Bound generated answer-coordinate tails while remaining differentiable."""

    values = torch.as_tensor(global_codes)
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("global answer codes must be a floating matrix")
    config = lossless_global_code_tail_guard_config(
        decomposition,
        threshold_multiplier=float(threshold_multiplier),
        transition_multiplier=float(transition_multiplier),
    )
    threshold = values.new_tensor(config["identity_threshold"])
    width = values.new_tensor(config["transition_width"])
    magnitude = torch.abs(values)
    bounded_magnitude = torch.where(
        magnitude <= threshold,
        magnitude,
        threshold + width * torch.tanh((magnitude - threshold) / width),
    )
    bounded = torch.sign(values) * bounded_magnitude
    if not bool(torch.all(torch.isfinite(bounded)).item()):
        raise FloatingPointError("bounded global answer codes are non-finite")
    return bounded

def encode_lossless_global_residual(
    normalized_fields,
    decomposition: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an answer field into whitened joint coordinates and residual."""

    values = torch.as_tensor(normalized_fields)
    mean, components, scale = _global_residual_tensors(values, decomposition)
    centered = (values - mean).reshape(int(values.shape[0]), -1)
    flat_components = components.reshape(int(components.shape[0]), -1)
    score = centered @ flat_components.T
    code = _global_score_to_code(score, decomposition, scale)
    residual = centered - score @ flat_components
    return residual.reshape_as(values), code


def decode_lossless_global_residual(
    residual_fields,
    global_codes,
    decomposition: Mapping[str, Any],
) -> torch.Tensor:
    """Reconstruct the complete answer field without a PCA truncation."""

    residual = torch.as_tensor(residual_fields)
    mean, components, scale = _global_residual_tensors(
        residual, decomposition
    )
    code = torch.as_tensor(
        global_codes, dtype=residual.dtype, device=residual.device
    )
    expected = (int(residual.shape[0]), int(components.shape[0]))
    if tuple(code.shape) != expected:
        raise ValueError("global-residual code shape changed")
    score = _global_code_to_score(code, decomposition, scale)
    flat_components = components.reshape(int(components.shape[0]), -1)
    reconstructed = (
        residual.reshape(int(residual.shape[0]), -1)
        + score @ flat_components
        + mean.reshape(1, -1)
    )
    return reconstructed.reshape_as(residual)


def _global_score_to_code(score, decomposition, linear_scale):
    transform = str(decomposition.get("code_transform", "linear"))
    if transform == "linear":
        return score / linear_scale
    if transform != "robust_asinh":
        raise ValueError("unsupported global-residual code transform")
    score_center, score_scale, code_center = _global_code_transform_tensors(
        score, decomposition
    )
    return (
        torch.asinh((score - score_center) / score_scale) - code_center
    ) / linear_scale


def _global_code_to_score(code, decomposition, linear_scale):
    transform = str(decomposition.get("code_transform", "linear"))
    if transform == "linear":
        return code * linear_scale
    if transform != "robust_asinh":
        raise ValueError("unsupported global-residual code transform")
    score_center, score_scale, code_center = _global_code_transform_tensors(
        code, decomposition
    )
    return score_center + score_scale * torch.sinh(
        code * linear_scale + code_center
    )


def _global_code_transform_tensors(values, decomposition):
    rank = int(decomposition["rank"])
    rows = []
    for name in ("score_center", "score_scale", "code_center"):
        tensor = torch.as_tensor(
            decomposition[name], dtype=values.dtype, device=values.device
        )[None]
        if tuple(tensor.shape) != (1, rank):
            raise ValueError(f"global-residual {name} shape changed")
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise ValueError(f"global-residual {name} is non-finite")
        rows.append(tensor)
    if bool(torch.any(rows[1] <= 0.0).item()):
        raise ValueError("global-residual score scale is invalid")
    return tuple(rows)

def project_lossless_global_residual_complement(
    residual_fields,
    decomposition: Mapping[str, Any],
) -> torch.Tensor:
    """Remove only the fitted global coordinates from a tangent residual."""

    residual = torch.as_tensor(residual_fields)
    _mean, components, _scale = _global_residual_tensors(
        residual, decomposition
    )
    flat = residual.reshape(int(residual.shape[0]), -1)
    flat_components = components.reshape(int(components.shape[0]), -1)
    return (flat - (flat @ flat_components.T) @ flat_components).reshape_as(
        residual
    )


def _global_residual_tensors(values, decomposition):
    if decomposition.get("schema") != LOSSLESS_GLOBAL_RESIDUAL_SCHEMA:
        raise ValueError("unexpected lossless global-residual schema")
    expected_shape = tuple(int(value) for value in decomposition["field_shape"])
    if values.ndim != 4 or tuple(values.shape[1:]) != expected_shape:
        raise ValueError("global-residual field shape changed")
    rank = int(decomposition["rank"])
    mean = torch.as_tensor(
        decomposition["mean"], dtype=values.dtype, device=values.device
    )[None]
    components = torch.as_tensor(
        decomposition["components"],
        dtype=values.dtype,
        device=values.device,
    )
    scale = torch.as_tensor(
        decomposition["code_scale"],
        dtype=values.dtype,
        device=values.device,
    )[None]
    if tuple(mean.shape[1:]) != expected_shape:
        raise ValueError("global-residual mean shape changed")
    if tuple(components.shape) != (rank, *expected_shape):
        raise ValueError("global-residual component shape changed")
    if tuple(scale.shape) != (1, rank) or bool(torch.any(scale <= 0.0)):
        raise ValueError("global-residual code scale is invalid")
    return mean, components, scale


def save_lossless_shared_field_edm_checkpoint(
    path: str | Path,
    *,
    model: VPFPSharedGlobalLatentEDM,
    field_config: VPFPLosslessSharedFieldConfig,
    field_normalization: Mapping[str, Any],
    condition_normalization: Mapping[str, Any],
    solution_reliability_report: str | Path,
    solution_reliability_report_sha256: str,
    upstream_report: str | Path,
    upstream_report_sha256: str,
    train_config: Mapping[str, Any],
    step: int,
    global_residual_decomposition: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    ema_state_dict: Mapping[str, Any] | None = None,
) -> None:
    field_config.validate()
    if int(model.config.grid_size) != int(field_config.grid_size):
        raise ValueError("EDM and lossless field grid sizes disagree")
    if int(model.config.latent_channels) != int(field_config.field_channels):
        raise ValueError("EDM channels do not match the lossless field")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": LOSSLESS_FIELD_CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "field_config": field_config.to_dict(),
            "field_normalization": dict(field_normalization),
            "condition_normalization": dict(condition_normalization),
            "solution_reliability_report": str(
                Path(solution_reliability_report).resolve()
            ),
            "solution_reliability_report_sha256": str(
                solution_reliability_report_sha256
            ),
            "upstream_report": str(Path(upstream_report).resolve()),
            "upstream_report_sha256": str(upstream_report_sha256),
            "answer_first_contract": {
                "hidden_continuous_solution_is_primary": True,
                "public_source_is_operator_image_of_hidden_answer": True,
                "equation_first_source_is_training_target": False,
                "equation_first_dynamics_are_generation_constraints": False,
                "released_answer_coordinate_policy": "trusted_and_transition",
            },
            "global_residual_decomposition": (
                None
                if global_residual_decomposition is None
                else dict(global_residual_decomposition)
            ),
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_lossless_shared_field_edm_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[
    VPFPSharedGlobalLatentEDM,
    VPFPLosslessSharedFieldTransform,
    dict[str, Any],
]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != LOSSLESS_FIELD_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected lossless shared field checkpoint schema")
    contract = dict(payload.get("answer_first_contract") or {})
    if not bool(
        contract.get("hidden_continuous_solution_is_primary")
        and contract.get("public_source_is_operator_image_of_hidden_answer")
        and not contract.get("equation_first_source_is_training_target")
        and not contract.get("equation_first_dynamics_are_generation_constraints")
        and contract.get("released_answer_coordinate_policy")
        == "trusted_and_transition"
    ):
        raise ValueError("lossless shared field checkpoint violates answer-first contract")
    field_config = VPFPLosslessSharedFieldConfig.from_dict(
        payload["field_config"]
    )
    model_config = VPFPSharedGlobalLatentEDMConfig.from_dict(
        payload["model_config"]
    )
    decomposition = payload.get("global_residual_decomposition")
    if int(model_config.sampled_global_dim) > 0:
        if not isinstance(decomposition, Mapping):
            raise ValueError(
                "joint-global checkpoint is missing its exact residual map"
            )
        if decomposition.get("schema") != LOSSLESS_GLOBAL_RESIDUAL_SCHEMA:
            raise ValueError("joint-global checkpoint residual map changed")
        if int(decomposition.get("rank", -1)) != int(
            model_config.sampled_global_dim
        ):
            raise ValueError("joint-global checkpoint rank changed")
    elif decomposition is not None:
        raise ValueError(
            "field-only checkpoint cannot contain global coordinates"
        )
    if int(model_config.grid_size) != int(field_config.grid_size):
        raise ValueError("lossless field checkpoint grid binding changed")
    if int(model_config.latent_channels) != int(field_config.field_channels):
        raise ValueError("lossless field checkpoint channel binding changed")
    model = VPFPSharedGlobalLatentEDM.build(model_config).to(device=device)
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    transform = VPFPLosslessSharedFieldTransform.build(field_config).to(
        device=device
    )
    transform.eval()
    return model, transform, payload
