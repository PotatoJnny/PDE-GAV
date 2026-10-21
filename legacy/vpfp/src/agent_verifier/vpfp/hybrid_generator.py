from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_diffusion import (
    denormalize_vectors,
    normalize_vectors,
)
from agent_verifier.vpfp.hermite_density import total_degree_mode_pairs
from agent_verifier.vpfp.continuous_v2 import _real_fourier_half_modes
from agent_verifier.vpfp.structured_bottleneck import (
    decode_shared_local_time_channel,
    encode_shared_local_time_channel,
)
from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
    active_latent_from_full,
    full_latent_from_active,
    split_active_latent,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPHybridGeneratorConfig:
    """Macro projection and differentiable micro-residual contract.

    The rank is local channel-time rank, not the total latent dimension.  The
    residual lives in whitened PCA coordinates, keeps density fixed, and uses
    complete Fourier sine/cosine pairs so the admissible set commutes with
    translations on the spatial torus.
    """

    bottleneck_rank: int = 256
    micro_component_count: int = 96
    micro_code_pair_count: int = 256
    micro_code_seed: int = 20260715
    micro_min_hermite_degree: int = 4
    micro_min_spatial_frequency: float = 2.0
    micro_sobolev_order: float = 1.0
    micro_weighted_rms_radius: float = 0.08
    micro_pair_amplitude_radius: float = 0.35
    component_scale_floor: float = 1.0e-4
    representation: str = "vpfp_online_rank256_hybrid_continuous_v1"

    def validate(self) -> None:
        if self.bottleneck_rank < 1:
            raise ValueError("bottleneck_rank must be positive")
        if not 1 <= self.micro_component_count <= self.bottleneck_rank:
            raise ValueError("micro_component_count must lie in [1, rank]")
        if self.micro_code_pair_count < 1:
            raise ValueError("micro_code_pair_count must be positive")
        if self.micro_min_hermite_degree < 0:
            raise ValueError("micro_min_hermite_degree must be nonnegative")
        if self.micro_min_spatial_frequency < 0.0:
            raise ValueError("micro_min_spatial_frequency must be nonnegative")
        if self.micro_sobolev_order < 0.0:
            raise ValueError("micro_sobolev_order must be nonnegative")
        if self.micro_weighted_rms_radius <= 0.0:
            raise ValueError("micro_weighted_rms_radius must be positive")
        if self.micro_pair_amplitude_radius <= 0.0:
            raise ValueError("micro_pair_amplitude_radius must be positive")
        if self.component_scale_floor <= 0.0:
            raise ValueError("component_scale_floor must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPHybridGeneratorConfig":
        return cls(**dict(data))


class VPFPHybridLatentTransform:
    """Project online equation-first latents and parameterize micro attacks."""

    def __init__(
        self,
        basis: dict[str, Any],
        *,
        config: VPFPHybridGeneratorConfig | dict[str, Any] | None = None,
        expected_decoder_checkpoint: str | Path | None = None,
    ):
        self.config = (
            config
            if isinstance(config, VPFPHybridGeneratorConfig)
            else VPFPHybridGeneratorConfig.from_dict(dict(config or {}))
        )
        self.config.validate()
        self.basis = dict(basis)
        self.canonical_config = VPFPStructuredDiffusionConfig.from_dict(
            dict(self.basis.get("structured_config") or {})
        )
        self.canonical_config.validate()
        self.compressed_config = VPFPStructuredDiffusionConfig.from_dict(
            {
                **self.canonical_config.to_dict(),
                "local_channels": int(self.config.bottleneck_rank),
                "local_time_order": 0,
                "local_time_feature_count": 1,
                "representation": self.config.representation,
            }
        )
        self.compressed_config.validate()
        self._validate_basis(expected_decoder_checkpoint)
        self.component_mask, self.component_scores = (
            self._build_micro_component_mask()
        )
        self.spatial_mask, self.spatial_frequencies = (
            self._build_spatial_mask()
        )
        torch = _torch()
        explained = torch.as_tensor(
            self.basis["explained_energy"], dtype=torch.float64
        )[: int(self.config.bottleneck_rank)]
        self.component_scales = torch.sqrt(torch.clamp(
            explained,
            min=float(self.config.component_scale_floor) ** 2,
        ))
        self.sobolev_weights = torch.pow(
            1.0 + self.spatial_frequencies.square(),
            0.5 * float(self.config.micro_sobolev_order),
        )
        self.micro_code_pairs, self.micro_code_flat_indices = (
            self._build_micro_code_pairs()
        )

    @property
    def micro_code_dim(self) -> int:
        return 2 * int(self.config.micro_code_pair_count)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        config: VPFPHybridGeneratorConfig | dict[str, Any] | None = None,
        expected_decoder_checkpoint: str | Path | None = None,
        map_location: str = "cpu",
    ) -> "VPFPHybridLatentTransform":
        torch = _torch()
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=False
        )
        return cls(
            payload,
            config=config,
            expected_decoder_checkpoint=expected_decoder_checkpoint,
        )

    def encode_full(self, full_latents):
        """Return normalized rank-reduced active coordinates."""

        canonical_active = active_latent_from_full(
            full_latents, self.canonical_config
        )
        canonical_normalized = normalize_vectors(
            canonical_active,
            self.basis["active_latent_normalization"],
        )
        return encode_shared_local_time_channel(
            canonical_normalized,
            self.basis,
            self.canonical_config,
            self.compressed_config,
            rank=int(self.config.bottleneck_rank),
        )

    def decode_active(self, compressed_active):
        """Decode compressed coordinates to the canonical active latent."""

        canonical_normalized = decode_shared_local_time_channel(
            compressed_active,
            self.basis,
            self.compressed_config,
            self.canonical_config,
            rank=int(self.config.bottleneck_rank),
        )
        return denormalize_vectors(
            canonical_normalized,
            self.basis["active_latent_normalization"],
        )

    def decode_full(self, compressed_active):
        return full_latent_from_active(
            self.decode_active(compressed_active), self.canonical_config
        )

    def project_full(self, full_latents):
        compressed = self.encode_full(full_latents)
        return self.decode_full(compressed)

    def zero_micro_residual(self, batch_size: int, *, device=None, dtype=None):
        torch = _torch()
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            (int(batch_size), self.compressed_config.active_latent_dim),
            device=device,
            dtype=dtype or torch.float32,
        )

    def constrain_micro_residual(self, raw_residual):
        """Project raw whitened coordinates onto the micro trust region."""

        torch = _torch()
        raw = torch.as_tensor(raw_residual)
        expected = int(self.compressed_config.active_latent_dim)
        if raw.ndim != 2 or int(raw.shape[1]) != expected:
            raise ValueError("micro residual shape does not match compressed latent")
        _, local = split_active_latent(raw, self.compressed_config)
        component_mask = self.component_mask.to(
            dtype=raw.dtype, device=raw.device
        )[None, :, None, None]
        spatial_mask = self.spatial_mask.to(
            dtype=raw.dtype, device=raw.device
        )[None, None, None, :]
        constrained = local * component_mask * spatial_mask
        constrained = self._clip_fourier_pair_amplitudes(constrained)

        weights = self.sobolev_weights.to(
            dtype=raw.dtype, device=raw.device
        )[None, None, None, :]
        active_count = torch.clamp(
            torch.sum(component_mask * spatial_mask), min=1.0
        )
        weighted_mean_square = (
            torch.sum((constrained * weights).square(), dim=(1, 2, 3))
            / active_count
        )
        weighted_rms = torch.sqrt(torch.clamp(
            weighted_mean_square, min=torch.finfo(raw.dtype).tiny
        ))
        radius = float(self.config.micro_weighted_rms_radius)
        scale = torch.clamp(radius / torch.clamp(
            weighted_rms, min=torch.finfo(raw.dtype).tiny
        ), max=1.0)
        constrained = constrained * scale[:, None, None, None]

        output = torch.zeros_like(raw)
        output[:, self.compressed_config.rho_latent_dim :] = (
            constrained.reshape(int(raw.shape[0]), -1)
        )
        return output

    def residual_from_micro_code(self, micro_code):
        """Lift a 512-D torus-pair code into the constrained residual space."""

        torch = _torch()
        code = torch.as_tensor(micro_code)
        if code.ndim != 2 or int(code.shape[1]) != int(self.micro_code_dim):
            raise ValueError("micro code shape does not match hybrid bottleneck")
        indices = self.micro_code_flat_indices.to(device=code.device)[None].expand(
            int(code.shape[0]), -1
        )
        local = torch.zeros(
            (int(code.shape[0]), self.compressed_config.local_latent_dim),
            dtype=code.dtype,
            device=code.device,
        ).scatter(1, indices, code)
        raw = torch.cat(
            (
                torch.zeros(
                    (int(code.shape[0]), self.compressed_config.rho_latent_dim),
                    dtype=code.dtype,
                    device=code.device,
                ),
                local,
            ),
            dim=-1,
        )
        return self.constrain_micro_residual(raw)

    def micro_coordinates_from_residual(self, raw_residual):
        """Extract the selected whitened Fourier-pair coordinates."""

        torch = _torch()
        residual = self.constrain_micro_residual(raw_residual)
        local = residual[:, self.compressed_config.rho_latent_dim :]
        indices = self.micro_code_flat_indices.to(device=local.device)
        return local.index_select(1, indices)

    def effective_micro_coordinates(self, micro_code, radius_scales=1.0):
        """Map raw codes to the actual bounded coordinates applied to a case."""

        torch = _torch()
        residual = self.residual_from_micro_code(micro_code)
        scales = torch.as_tensor(
            radius_scales, dtype=residual.dtype, device=residual.device
        )
        if scales.ndim == 0:
            scales = scales.expand(int(residual.shape[0]))
        scales = scales.reshape(-1)
        if int(scales.shape[0]) != int(residual.shape[0]):
            raise ValueError("radius scale batch does not match micro code")
        if bool(torch.any((scales < 0.0) | (scales > 1.0)).item()):
            raise ValueError("radius scales must lie in [0, 1]")
        return self.micro_coordinates_from_residual(
            residual * scales[:, None]
        )

    def base_micro_coordinates(self, base_compressed):
        """Extract base PCA scores in the micro residual's whitened units."""

        torch = _torch()
        base = torch.as_tensor(base_compressed)
        if base.ndim != 2 or int(base.shape[1]) != int(
            self.compressed_config.active_latent_dim
        ):
            raise ValueError("base compressed latent has the wrong shape")
        _, local = split_active_latent(base, self.compressed_config)
        scales = self.component_scales.to(
            dtype=base.dtype, device=base.device
        )[None, :, None, None]
        whitened = (local / scales).reshape(int(base.shape[0]), -1)
        indices = self.micro_code_flat_indices.to(device=base.device)
        return whitened.index_select(1, indices)

    def apply_micro_code(self, base_compressed, micro_code):
        return self.apply_micro_residual(
            base_compressed, self.residual_from_micro_code(micro_code)
        )

    def apply_micro_residual(self, base_compressed, raw_residual):
        """Apply a constrained whitened residual to PCA score coordinates."""

        torch = _torch()
        base = torch.as_tensor(base_compressed)
        if base.ndim != 2 or int(base.shape[1]) != int(
            self.compressed_config.active_latent_dim
        ):
            raise ValueError("base compressed latent has the wrong shape")
        residual = self.constrain_micro_residual(raw_residual).to(
            dtype=base.dtype, device=base.device
        )
        if int(residual.shape[0]) != int(base.shape[0]):
            raise ValueError("base and residual batch dimensions differ")
        rho_delta, local_delta = split_active_latent(
            residual, self.compressed_config
        )
        scales = self.component_scales.to(
            dtype=base.dtype, device=base.device
        )[None, :, None, None]
        score_delta = torch.cat(
            (
                rho_delta.reshape(int(base.shape[0]), -1),
                (local_delta * scales).reshape(int(base.shape[0]), -1),
            ),
            dim=-1,
        )
        return base + score_delta

    def residual_diagnostics(self, constrained_residual) -> dict[str, Any]:
        torch = _torch()
        residual = torch.as_tensor(constrained_residual)
        if residual.ndim != 2:
            raise ValueError("residual diagnostics expect a matrix")
        rho, local = split_active_latent(residual, self.compressed_config)
        weights = self.sobolev_weights.to(
            dtype=residual.dtype, device=residual.device
        )[None, None, None, :]
        active_count = max(
            1,
            int(self.component_mask.sum().item())
            * int(self.spatial_mask.sum().item()),
        )
        weighted_mean_square = (
            torch.sum((local * weights).square(), dim=(1, 2, 3))
            / float(active_count)
        )
        weighted_rms = torch.sqrt(torch.clamp(
            weighted_mean_square, min=torch.finfo(residual.dtype).tiny
        ))
        return {
            "rho_max_abs": float(torch.max(torch.abs(rho)).item()),
            "local_rms_mean": float(
                torch.sqrt(torch.mean(local.square(), dim=(1, 2, 3)))
                .mean()
                .item()
            ),
            "weighted_rms_max": float(torch.max(weighted_rms).item()),
            "selected_component_count": int(self.component_mask.sum().item()),
            "selected_spatial_feature_count": int(self.spatial_mask.sum().item()),
            "effective_micro_dimension": int(
                self.component_mask.sum().item()
                * self.spatial_mask.sum().item()
            ),
            "micro_code_pair_count": int(self.config.micro_code_pair_count),
            "micro_code_dimension": int(self.micro_code_dim),
        }

    def public_contract(self) -> dict[str, Any]:
        """Describe Agent-visible behavior without private parameterization."""

        return {
            "schema": "vpfp_hybrid_public_generator_contract_v1",
            "generator_route": "online_equation_first_continuous_hidden_truth",
            "hidden_function_is_primary": True,
            "public_access": {
                "fresh_problem_sampling": True,
                "completed_hidden_solution_oracle": True,
                "pointwise_or_grid_query": True,
            },
            "private_fields_not_exposed": [
                "decoder_parameters",
                "latent_coordinates",
                "pca_basis",
                "reward_model_parameters",
            ],
            "equation_condition_keys": ["collision_nu", "temperature"],
            "source_derivation": "continuous_hidden_answer_autodiff_residual",
        }

    def private_provenance(self) -> dict[str, Any]:
        return {
            "schema": "vpfp_hybrid_latent_transform_provenance_v1",
            "config": self.config.to_dict(),
            "canonical_config": self.canonical_config.to_dict(),
            "compressed_config": self.compressed_config.to_dict(),
            "decoder_checkpoint": str(self.basis["decoder_checkpoint"]),
            "basis_strategy": str(self.basis["strategy"]),
            "basis_retained_rank": int(self.basis["retained_rank"]),
            "selected_component_indices": np.flatnonzero(
                self.component_mask.detach().cpu().numpy()
            ).astype(int).tolist(),
            "selected_component_scores": self.component_scores[
                self.component_mask
            ].detach().cpu().tolist(),
            "effective_micro_dimension": int(
                self.component_mask.sum().item()
                * self.spatial_mask.sum().item()
            ),
            "micro_code_pair_count": int(self.config.micro_code_pair_count),
            "micro_code_dimension": int(self.micro_code_dim),
            "micro_code_pair_index_sha256": hashlib.sha256(
                self.micro_code_pairs.detach().cpu().numpy().astype(
                    np.int64
                ).tobytes()
            ).hexdigest(),
        }

    def _validate_basis(self, expected_decoder_checkpoint) -> None:
        if self.basis.get("schema") != "vpfp_structured_pca_basis_v2":
            raise ValueError("unexpected structured PCA basis schema")
        if self.basis.get("strategy") != "shared_local_time_channel":
            raise ValueError("hybrid generator requires shared local PCA")
        rank = int(self.config.bottleneck_rank)
        if rank > int(self.basis.get("retained_rank", 0)):
            raise ValueError("requested hybrid bottleneck rank is unavailable")
        components = np.asarray(self.basis.get("components"))
        expected_features = int(
            self.canonical_config.local_channels
            * self.canonical_config.local_time_count
        )
        if components.ndim != 2 or components.shape[0] != expected_features:
            raise ValueError("PCA basis components do not match canonical layout")
        if components.shape[1] < rank or not np.all(np.isfinite(components)):
            raise ValueError("PCA basis is finite but lacks requested rank")
        explained = np.asarray(self.basis.get("explained_energy"))
        if explained.ndim != 1 or explained.size < rank:
            raise ValueError("PCA basis lacks explained-energy scales")
        normalization = self.basis.get("active_latent_normalization")
        if not isinstance(normalization, dict) or normalization.get("schema") != (
            "vpfp_translation_invariant_latent_normalization_v1"
        ):
            raise ValueError("PCA basis lacks canonical normalization")
        if self.basis.get("centering") != "origin_preserving":
            raise ValueError("hybrid PCA basis must preserve the origin")
        if self.basis.get("group_action") != (
            "shared_torus_fourier_phase_rotation"
        ):
            raise ValueError("hybrid PCA basis has incompatible group action")
        if expected_decoder_checkpoint is not None:
            expected = Path(expected_decoder_checkpoint).resolve()
            actual = Path(str(self.basis["decoder_checkpoint"])).resolve()
            if actual != expected:
                raise ValueError("PCA basis and Hermite decoder disagree")

    def _build_micro_component_mask(self):
        torch = _torch()
        components = torch.as_tensor(
            self.basis["components"], dtype=torch.float64
        )[:, : int(self.config.bottleneck_rank)]
        modes = total_degree_mode_pairs(
            _infer_hermite_total_degree(self.canonical_config.local_channels)
        )
        if len(modes) != int(self.canonical_config.local_channels):
            raise ValueError("canonical local channels are not triangular Hermite modes")
        degrees = torch.as_tensor(
            [first + second for first, second in modes], dtype=torch.long
        )
        matrix = components.reshape(
            self.canonical_config.local_channels,
            self.canonical_config.local_time_count,
            int(self.config.bottleneck_rank),
        )
        degree_weight = torch.clamp(
            degrees - int(self.config.micro_min_hermite_degree) + 1,
            min=0,
        ).to(dtype=matrix.dtype).square()
        component_energy = matrix.square()
        scores = torch.sum(
            component_energy * degree_weight[:, None, None], dim=(0, 1)
        ) / torch.clamp(
            torch.sum(component_energy, dim=(0, 1)),
            min=torch.finfo(matrix.dtype).tiny,
        )
        count = min(int(self.config.micro_component_count), int(scores.numel()))
        selected = torch.topk(scores, k=count, largest=True).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[selected] = True
        return mask, scores

    def _build_spatial_mask(self):
        torch = _torch()
        mode_limit = int(self.compressed_config.local_spatial_mode)
        modes = _real_fourier_half_modes(mode_limit, mode_limit)
        frequencies = [0.0]
        for kx, ky in modes:
            frequency = float(np.hypot(kx, ky))
            frequencies.extend((frequency, frequency))
        if len(frequencies) != int(
            self.compressed_config.local_spatial_feature_count
        ):
            raise RuntimeError("local Fourier feature ordering is inconsistent")
        values = torch.as_tensor(frequencies, dtype=torch.float64)
        mask = values >= float(self.config.micro_min_spatial_frequency)
        if not bool(torch.any(mask).item()):
            raise ValueError("micro spatial-frequency threshold selects no modes")
        return mask, values

    def _build_micro_code_pairs(self):
        torch = _torch()
        components = torch.nonzero(self.component_mask, as_tuple=False)[:, 0]
        mode_indices = []
        mode_count = (int(self.spatial_mask.numel()) - 1) // 2
        for mode_index in range(mode_count):
            first = 1 + 2 * mode_index
            if bool(self.spatial_mask[first]) and bool(self.spatial_mask[first + 1]):
                mode_indices.append(mode_index)
        pairs = torch.as_tensor(
            [
                (int(component), int(mode_index))
                for component in components.tolist()
                for mode_index in mode_indices
            ],
            dtype=torch.long,
        )
        count = int(self.config.micro_code_pair_count)
        if pairs.ndim != 2 or int(pairs.shape[0]) < count:
            raise ValueError("micro code requests more Fourier pairs than available")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.micro_code_seed))
        selected = pairs[torch.randperm(int(pairs.shape[0]), generator=generator)[:count]]
        spatial_width = int(self.compressed_config.local_spatial_feature_count)
        flat = []
        for component, mode_index in selected.tolist():
            start = int(component) * spatial_width + 1 + 2 * int(mode_index)
            flat.extend((start, start + 1))
        return selected, torch.as_tensor(flat, dtype=torch.long)

    def _clip_fourier_pair_amplitudes(self, local):
        torch = _torch()
        if int(local.shape[-1]) <= 1:
            return local
        pairs = local[..., 1:].reshape(*local.shape[:-1], -1, 2)
        amplitude = torch.linalg.vector_norm(pairs, dim=-1, keepdim=True)
        radius = float(self.config.micro_pair_amplitude_radius)
        clipped = pairs * torch.clamp(
            radius / torch.clamp(
                amplitude, min=torch.finfo(local.dtype).tiny
            ),
            max=1.0,
        )
        return torch.cat(
            (local[..., :1], clipped.reshape(*local.shape[:-1], -1)),
            dim=-1,
        )


def _infer_hermite_total_degree(channel_count: int) -> int:
    count = int(channel_count)
    degree = 0
    while (degree + 1) * (degree + 2) // 2 < count:
        degree += 1
    if (degree + 1) * (degree + 2) // 2 != count:
        raise ValueError("local channel count is not a complete Hermite triangle")
    return degree


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for the hybrid generator") from (
            _TORCH_IMPORT_ERROR
        )
    return _TORCH_MODULE
