from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


STRUCTURED_NS_PCA_SCHEMA = "ns2d_structured_coefficient_pca_v1"
STRUCTURED_NS_FLOW_SCHEMA = "ns2d_structured_coefficient_rectified_flow_v1"
STRUCTURED_NS_DIRECT_STANDARDIZER_SCHEMA = (
    "ns2d_structured_coefficient_direct_standardizer_v1"
)
STRUCTURED_NS_DIRECT_FLOW_SCHEMA = (
    "ns2d_structured_coefficient_direct_rectified_flow_v1"
)


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is required for the structured NS distribution model"
        ) from exc
    return torch


@dataclass(frozen=True)
class StructuredCoefficientCorpus:
    coefficients: Any
    case_ids: tuple[str, ...]
    families: tuple[str, ...]
    manifest: dict[str, Any]

    @property
    def case_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.coefficients.shape[1:])


@dataclass(frozen=True)
class StructuredCoefficientPCA:
    """Train-fitted, reversible PCA for continuous NS coefficients.

    Coordinate means are removed, while one scalar scale per physical field
    balances streamfunction and pressure. No casewise normalization is used, so
    case-to-case amplitudes remain part of the learned distribution.
    """

    coordinate_mean: Any
    field_scale: Any
    components: Any
    score_scale: Any
    singular_values: Any
    total_centered_energy: Any
    coefficient_shape: tuple[int, ...]
    fit_case_count: int
    source_signature: dict[str, Any]

    @classmethod
    def fit(
        cls,
        coefficients,
        *,
        rank: int,
        source_signature: dict[str, Any],
        oversample: int = 16,
        power_iterations: int = 4,
        seed: int = 0,
        minimum_scale: float = 1.0e-8,
    ) -> "StructuredCoefficientPCA":
        torch = _require_torch()
        values = torch.as_tensor(coefficients, dtype=torch.float32)
        if values.ndim != 5 or int(values.shape[1]) != 2:
            raise ValueError("structured coefficients must have shape [N,2,R,Kx,Ky]")
        if int(values.shape[0]) < 2:
            raise ValueError("PCA requires at least two coefficient cases")
        maximum_rank = min(int(values.shape[0]) - 1, int(values[0].numel()))
        if not 1 <= int(rank) <= maximum_rank:
            raise ValueError(f"PCA rank must lie in [1, {maximum_rank}]")
        if min(int(oversample), int(power_iterations)) < 0:
            raise ValueError("PCA oversampling and power iterations cannot be negative")
        if float(minimum_scale) <= 0.0:
            raise ValueError("minimum PCA field scale must be positive")

        coordinate_mean = values.mean(dim=0, keepdim=True)
        centered = values - coordinate_mean
        field_scale = torch.sqrt(
            centered.square().mean(dim=(0, 2, 3, 4), keepdim=True)
        ).clamp_min(float(minimum_scale))
        balanced = (centered / field_scale).flatten(1)
        total_centered_energy = balanced.double().square().sum()
        requested_q = min(
            maximum_rank,
            int(rank) + max(0, int(oversample)),
        )
        with torch.random.fork_rng(devices=_rng_devices(values)):
            torch.manual_seed(int(seed))
            if values.is_cuda:
                torch.cuda.manual_seed_all(int(seed))
            _u, singular_values, right_vectors = torch.pca_lowrank(
                balanced,
                q=requested_q,
                center=False,
                niter=int(power_iterations),
            )
        singular_values = singular_values[: int(rank)].contiguous()
        components = right_vectors[:, : int(rank)].T.contiguous()
        score_scale = (
            singular_values / float(max(int(values.shape[0]) - 1, 1)) ** 0.5
        ).clamp_min(float(minimum_scale))
        return cls(
            coordinate_mean=coordinate_mean.detach(),
            field_scale=field_scale.detach(),
            components=components.detach(),
            score_scale=score_scale.detach(),
            singular_values=singular_values.detach(),
            total_centered_energy=total_centered_energy.detach(),
            coefficient_shape=tuple(int(value) for value in values.shape[1:]),
            fit_case_count=int(values.shape[0]),
            source_signature=dict(source_signature),
        )

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def encode(self, coefficients, *, whiten: bool = True):
        torch = _require_torch()
        values = torch.as_tensor(coefficients)
        if tuple(values.shape[1:]) != self.coefficient_shape:
            raise ValueError("coefficient shape does not match PCA model")
        mean = self.coordinate_mean.to(device=values.device, dtype=values.dtype)
        scale = self.field_scale.to(device=values.device, dtype=values.dtype)
        components = self.components.to(device=values.device, dtype=values.dtype)
        scores = ((values - mean) / scale).flatten(1) @ components.T
        if whiten:
            score_scale = self.score_scale.to(
                device=values.device,
                dtype=values.dtype,
            )
            scores = scores / score_scale
        return scores

    def decode(self, scores, *, whitened: bool = True):
        torch = _require_torch()
        values = torch.as_tensor(scores)
        if values.ndim != 2 or int(values.shape[1]) != self.rank:
            raise ValueError("PCA score shape does not match model rank")
        if whitened:
            values = values * self.score_scale.to(
                device=values.device,
                dtype=values.dtype,
            )
        components = self.components.to(device=values.device, dtype=values.dtype)
        mean = self.coordinate_mean.to(device=values.device, dtype=values.dtype)
        scale = self.field_scale.to(device=values.device, dtype=values.dtype)
        balanced = values @ components
        return balanced.reshape(values.shape[0], *self.coefficient_shape) * scale + mean

    def reconstruct(self, coefficients):
        return self.decode(self.encode(coefficients), whitened=True)

    def explained_variance_fraction(self, rank: int | None = None) -> float:
        selected_rank = self.rank if rank is None else int(rank)
        if not 1 <= selected_rank <= self.rank:
            raise ValueError("requested PCA rank is unavailable")
        numerator = self.singular_values[:selected_rank].double().square().sum()
        denominator = self.total_centered_energy.double().clamp_min(1.0e-12)
        return float((numerator / denominator).item())

    def with_rank(self, rank: int) -> "StructuredCoefficientPCA":
        selected_rank = int(rank)
        if not 1 <= selected_rank <= self.rank:
            raise ValueError("requested PCA rank is unavailable")
        return type(self)(
            coordinate_mean=self.coordinate_mean,
            field_scale=self.field_scale,
            components=self.components[:selected_rank],
            score_scale=self.score_scale[:selected_rank],
            singular_values=self.singular_values[:selected_rank],
            total_centered_energy=self.total_centered_energy,
            coefficient_shape=self.coefficient_shape,
            fit_case_count=self.fit_case_count,
            source_signature=self.source_signature,
        )

    def save(self, path: str | Path) -> None:
        torch = _require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": STRUCTURED_NS_PCA_SCHEMA,
                "coordinate_mean": self.coordinate_mean.detach().cpu(),
                "field_scale": self.field_scale.detach().cpu(),
                "components": self.components.detach().cpu(),
                "score_scale": self.score_scale.detach().cpu(),
                "singular_values": self.singular_values.detach().cpu(),
                "total_centered_energy": self.total_centered_energy.detach().cpu(),
                "coefficient_shape": list(self.coefficient_shape),
                "fit_case_count": int(self.fit_case_count),
                "source_signature": self.source_signature,
            },
            target,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | Any = "cpu",
    ) -> "StructuredCoefficientPCA":
        torch = _require_torch()
        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        if payload.get("schema") != STRUCTURED_NS_PCA_SCHEMA:
            raise ValueError("unexpected structured NS PCA checkpoint schema")
        return cls(
            coordinate_mean=payload["coordinate_mean"],
            field_scale=payload["field_scale"],
            components=payload["components"],
            score_scale=payload["score_scale"],
            singular_values=payload["singular_values"],
            total_centered_energy=payload["total_centered_energy"],
            coefficient_shape=tuple(int(value) for value in payload["coefficient_shape"]),
            fit_case_count=int(payload["fit_case_count"]),
            source_signature=dict(payload["source_signature"]),
        )


@dataclass(frozen=True)
class StructuredCoefficientDirectStandardizer:
    """Train-fitted diagonal transform for a literal no-PCA control.

    Every non-constant coefficient is centered and divided by its training-set
    population standard deviation. Constant coordinates remain present in the
    7,514-dimensional flow input, but decode exactly to their training mean.
    This performs no rotation, projection, rank truncation, or casewise
    normalization.
    """

    coordinate_mean: Any
    coordinate_std: Any
    coefficient_shape: tuple[int, ...]
    fit_case_count: int
    source_signature: dict[str, Any]
    statistics_source: str = ""
    statistics_sha256: str = ""

    @classmethod
    def fit(
        cls,
        coefficients,
        *,
        source_signature: dict[str, Any],
    ) -> "StructuredCoefficientDirectStandardizer":
        torch = _require_torch()
        values = torch.as_tensor(coefficients, dtype=torch.float32)
        if values.ndim != 5 or int(values.shape[1]) != 2:
            raise ValueError("structured coefficients must have shape [N,2,R,Kx,Ky]")
        if int(values.shape[0]) < 2:
            raise ValueError("direct standardization requires at least two cases")
        return cls(
            coordinate_mean=values.mean(dim=0, keepdim=True).detach(),
            coordinate_std=values.std(dim=0, correction=0, keepdim=True).detach(),
            coefficient_shape=tuple(int(value) for value in values.shape[1:]),
            fit_case_count=int(values.shape[0]),
            source_signature=dict(source_signature),
        )

    @classmethod
    def from_statistics(
        cls,
        *,
        mean,
        std,
        fit_case_count: int,
        source_signature: dict[str, Any],
        statistics_source: str = "",
        statistics_sha256: str = "",
    ) -> "StructuredCoefficientDirectStandardizer":
        torch = _require_torch()
        coordinate_mean = torch.as_tensor(mean, dtype=torch.float32)
        coordinate_std = torch.as_tensor(std, dtype=torch.float32)
        if coordinate_mean.ndim == 4:
            coordinate_mean = coordinate_mean.unsqueeze(0)
        if coordinate_std.ndim == 4:
            coordinate_std = coordinate_std.unsqueeze(0)
        if coordinate_mean.shape != coordinate_std.shape:
            raise ValueError("direct standardizer mean and std shapes differ")
        if coordinate_mean.ndim != 5 or int(coordinate_mean.shape[1]) != 2:
            raise ValueError("direct standardizer statistics have the wrong shape")
        if bool((coordinate_std < 0).any()):
            raise ValueError("direct standardizer standard deviations cannot be negative")
        if int(fit_case_count) < 2:
            raise ValueError("direct standardizer requires at least two fit cases")
        return cls(
            coordinate_mean=coordinate_mean.contiguous(),
            coordinate_std=coordinate_std.contiguous(),
            coefficient_shape=tuple(int(value) for value in coordinate_mean.shape[1:]),
            fit_case_count=int(fit_case_count),
            source_signature=dict(source_signature),
            statistics_source=str(statistics_source),
            statistics_sha256=str(statistics_sha256),
        )

    @property
    def feature_dim(self) -> int:
        return int(math.prod(self.coefficient_shape))

    @property
    def constant_coordinate_count(self) -> int:
        return int((self.coordinate_std <= 0).sum().item())

    def encode(self, coefficients):
        torch = _require_torch()
        values = torch.as_tensor(coefficients)
        if tuple(values.shape[1:]) != self.coefficient_shape:
            raise ValueError("coefficient shape does not match direct standardizer")
        mean = self.coordinate_mean.to(device=values.device, dtype=values.dtype)
        std = self.coordinate_std.to(device=values.device, dtype=values.dtype)
        safe_std = torch.where(std > 0, std, torch.ones_like(std))
        normalized = (values - mean) / safe_std
        normalized = torch.where(std > 0, normalized, torch.zeros_like(normalized))
        return normalized.flatten(1)

    def decode(self, features):
        torch = _require_torch()
        values = torch.as_tensor(features)
        if values.ndim != 2 or int(values.shape[1]) != self.feature_dim:
            raise ValueError("direct standardized feature shape does not match model")
        mean = self.coordinate_mean.to(device=values.device, dtype=values.dtype)
        std = self.coordinate_std.to(device=values.device, dtype=values.dtype)
        return values.reshape(values.shape[0], *self.coefficient_shape) * std + mean

    def reconstruct(self, coefficients):
        return self.decode(self.encode(coefficients))

    def save(self, path: str | Path) -> None:
        torch = _require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": STRUCTURED_NS_DIRECT_STANDARDIZER_SCHEMA,
                "coordinate_mean": self.coordinate_mean.detach().cpu(),
                "coordinate_std": self.coordinate_std.detach().cpu(),
                "coefficient_shape": list(self.coefficient_shape),
                "feature_dim": int(self.feature_dim),
                "fit_case_count": int(self.fit_case_count),
                "constant_coordinate_count": int(self.constant_coordinate_count),
                "source_signature": self.source_signature,
                "statistics_source": str(self.statistics_source),
                "statistics_sha256": str(self.statistics_sha256),
            },
            target,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | Any = "cpu",
    ) -> "StructuredCoefficientDirectStandardizer":
        torch = _require_torch()
        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        if payload.get("schema") != STRUCTURED_NS_DIRECT_STANDARDIZER_SCHEMA:
            raise ValueError("unexpected structured NS direct standardizer schema")
        model = cls(
            coordinate_mean=payload["coordinate_mean"],
            coordinate_std=payload["coordinate_std"],
            coefficient_shape=tuple(int(value) for value in payload["coefficient_shape"]),
            fit_case_count=int(payload["fit_case_count"]),
            source_signature=dict(payload["source_signature"]),
            statistics_source=str(payload.get("statistics_source", "")),
            statistics_sha256=str(payload.get("statistics_sha256", "")),
        )
        if int(payload.get("feature_dim", model.feature_dim)) != model.feature_dim:
            raise ValueError("direct standardizer feature dimension is inconsistent")
        expected_constant = int(
            payload.get("constant_coordinate_count", model.constant_coordinate_count)
        )
        if expected_constant != model.constant_coordinate_count:
            raise ValueError("direct standardizer constant-coordinate count is inconsistent")
        return model


def load_structured_coefficient_corpus(
    manifest_path: str | Path,
    *,
    max_cases: int = 0,
    device: str | Any = "cpu",
) -> StructuredCoefficientCorpus:
    torch = _require_torch()
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "ns2d_structured_coefficient_dataset_v2":
        raise ValueError("unexpected structured coefficient dataset schema")
    expected_shape = tuple(int(value) for value in manifest["coefficient_shape"])
    limit = int(max_cases)
    if limit < 0:
        raise ValueError("max_cases cannot be negative")
    coefficient_parts = []
    case_ids: list[str] = []
    families: list[str] = []
    for shard in manifest["shards"]:
        shard_path = path.parent / str(shard["path"])
        with np.load(shard_path) as arrays:
            coefficients = np.asarray(arrays["coefficients"], dtype=np.float32)
            if tuple(coefficients.shape[1:]) != expected_shape:
                raise ValueError(f"coefficient shape mismatch in {shard_path}")
            remaining = coefficients.shape[0]
            if limit > 0:
                remaining = min(remaining, limit - len(case_ids))
            if remaining <= 0:
                break
            coefficient_parts.append(coefficients[:remaining].copy())
            case_ids.extend(str(value) for value in arrays["case_ids"][:remaining])
            families.extend(str(value) for value in arrays["families"][:remaining])
        if limit > 0 and len(case_ids) >= limit:
            break
    if not coefficient_parts:
        raise ValueError("structured coefficient corpus is empty")
    coefficients = torch.from_numpy(np.concatenate(coefficient_parts, axis=0)).to(
        device=device
    )
    if len(case_ids) != int(coefficients.shape[0]):
        raise RuntimeError("coefficient metadata is misaligned")
    return StructuredCoefficientCorpus(
        coefficients=coefficients,
        case_ids=tuple(case_ids),
        families=tuple(families),
        manifest=manifest,
    )


def compatible_projection_signatures(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    keys = ("split_manifest_sha256", "projection_config", "coefficient_shape")
    # Missing tags are archived v5, never permission to reuse a v5 corpus/PCA
    # for the different v6 streamfunction representation.
    legacy_abi = "continuous_structured_ns_mms_v5"
    return (first.get("continuous_mms_abi", legacy_abi)
            == second.get("continuous_mms_abi", legacy_abi)
            and all(first.get(key) == second.get(key) for key in keys))


@dataclass(frozen=True)
class StructuredCoefficientFlowConfig:
    feature_dim: int
    width: int = 768
    depth: int = 8
    time_width: int = 128
    full_rank_diagonal_velocity: bool = False

    def validate(self) -> None:
        if min(
            int(self.feature_dim),
            int(self.width),
            int(self.depth),
            int(self.time_width),
        ) < 1:
            raise ValueError("structured flow dimensions must be positive")
        if int(self.time_width) % 2:
            raise ValueError("structured flow time width must be even")

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "feature_dim": int(self.feature_dim),
            "width": int(self.width),
            "depth": int(self.depth),
            "time_width": int(self.time_width),
            "full_rank_diagonal_velocity": bool(
                self.full_rank_diagonal_velocity
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        return cls(**dict(payload))


class StructuredCoefficientRectifiedFlow:
    @staticmethod
    def build(config: StructuredCoefficientFlowConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, StructuredCoefficientFlowConfig)
            else StructuredCoefficientFlowConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _StructuredCoefficientRectifiedFlow(_require_torch(), cfg)


class _StructuredCoefficientFlowBlock(_require_torch().nn.Module):
    def __init__(self, torch, width: int, time_width: int) -> None:
        super().__init__()
        self.torch = torch
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


class _StructuredCoefficientRectifiedFlow(_require_torch().nn.Module):
    def __init__(self, torch, config: StructuredCoefficientFlowConfig) -> None:
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
            _StructuredCoefficientFlowBlock(
                torch,
                int(config.width),
                int(config.time_width),
            )
            for _ in range(int(config.depth))
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(int(config.width)),
            torch.nn.Linear(int(config.width), int(config.feature_dim)),
        )
        torch.nn.init.zeros_(self.output[-1].weight)
        torch.nn.init.zeros_(self.output[-1].bias)
        self.diagonal_velocity = (
            torch.nn.Linear(int(config.time_width), int(config.feature_dim))
            if bool(config.full_rank_diagonal_velocity)
            else None
        )
        if self.diagonal_velocity is not None:
            torch.nn.init.zeros_(self.diagonal_velocity.weight)
            torch.nn.init.zeros_(self.diagonal_velocity.bias)

    def forward(self, values, time):
        if values.ndim != 2 or int(values.shape[1]) != int(self.config.feature_dim):
            raise ValueError("structured flow input has the wrong shape")
        if time.shape != (values.shape[0],):
            raise ValueError("structured flow time has the wrong shape")
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
        velocity = self.output(hidden)
        if self.diagonal_velocity is not None:
            velocity = velocity + self.diagonal_velocity(embedding) * values
        return velocity

    def flow_matching_loss(
        self,
        clean,
        *,
        generator=None,
        noise=None,
        time=None,
    ):
        if clean.ndim != 2 or int(clean.shape[1]) != int(self.config.feature_dim):
            raise ValueError("structured flow clean batch has the wrong shape")
        noise = (
            self.torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
            if noise is None
            else noise
        )
        time = (
            self.torch.rand(
                clean.shape[0],
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
            if time is None
            else time
        )
        if noise.shape != clean.shape or time.shape != (clean.shape[0],):
            raise ValueError("structured flow noise or time shape is invalid")
        interpolated = (1.0 - time[:, None]) * noise + time[:, None] * clean
        target = clean - noise
        return self.torch.nn.functional.mse_loss(
            self(interpolated, time),
            target,
        )

    def sample(
        self,
        count: int,
        *,
        steps: int,
        device,
        generator=None,
        initial_values=None,
        guidance: Callable[[Any, Any, Any], Any] | None = None,
        guidance_scale: float = 0.0,
        method: str = "heun",
    ):
        return _sample_structured_flow(
            self,
            count,
            steps=steps,
            device=device,
            generator=generator,
            initial_values=initial_values,
            guidance=guidance,
            guidance_scale=guidance_scale,
            method=method,
        )


def _can_cache_structured_flow_conditioning(torch, model) -> bool:
    if type(model) is not _StructuredCoefficientRectifiedFlow or torch.is_grad_enabled():
        return False
    # Cache replay bypasses the top-level and block forwards, and skips some
    # conditioning forwards. Preserve customized execution through the eager
    # path instead of silently dropping hooks or replaced implementations.
    hook_names = (
        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
    )
    module_namespace = torch.nn.modules.module
    if any(getattr(module_namespace, "_global" + name, {}) for name in hook_names):
        return False
    allowed = {
        _StructuredCoefficientRectifiedFlow, _StructuredCoefficientFlowBlock,
        torch.nn.ModuleList, torch.nn.Sequential, torch.nn.Linear,
        torch.nn.LayerNorm, torch.nn.SiLU,
    }
    for layer in model.modules():
        if (
            type(layer) not in allowed or layer.training
            or any(name in vars(layer) for name in ("forward", "_call_impl", "_wrapped_call_impl"))
            or getattr(layer, "_compiled_call_impl", None) is not None
            or any(getattr(layer, name, {}) for name in hook_names)
        ):
            return False
    return True


def _sample_structured_flow(
    model,
    count: int,
    *,
    steps: int,
    device,
    generator=None,
    initial_values=None,
    guidance: Callable[[Any, Any, Any], Any] | None = None,
    guidance_scale: float = 0.0,
    method: str = "heun",
):
    torch = _require_torch()
    if min(int(count), int(steps)) < 1:
        raise ValueError("structured flow sample count and steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError("structured flow sampler must be euler or heun")
    if initial_values is None:
        values = torch.randn(
            (int(count), int(model.config.feature_dim)),
            device=device,
            generator=generator,
        )
    else:
        values = torch.as_tensor(initial_values, device=device)
        expected_shape = (int(count), int(model.config.feature_dim))
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"structured flow initial state must have shape {expected_shape}"
            )
        values = values.clone()
    # Heun evaluates the same time twice in succession: its corrector at
    # t[i + 1], then the next predictor at t[i + 1]. Only the state changes.
    # Keep one call-local conditioning entry for the frozen built-in model;
    # training, autograd, and arbitrary user models retain their forward path.
    cache_conditioning = _can_cache_structured_flow_conditioning(torch, model)
    cached_key = None
    cached_blocks = ()
    cached_diagonal = None

    def velocity_at_time(state, time, time_index):
        nonlocal cached_key, cached_blocks, cached_diagonal
        if not cache_conditioning:
            return model(state, time)
        if state.ndim != 2 or int(state.shape[1]) != int(model.config.feature_dim):
            raise ValueError("structured flow input has the wrong shape")
        if time.shape != (state.shape[0],):
            raise ValueError("structured flow time has the wrong shape")
        key = (time_index, time.dtype, time.device, tuple(time.shape))
        reuse = cached_key == key
        if not reuse:
            embedding = model.time_mlp(
                _continuous_time_embedding(torch, time, int(model.config.time_width))
            )
        hidden = model.input(state)
        conditions = []
        for block_index, block in enumerate(model.blocks):
            scale, shift = (
                cached_blocks[block_index] if reuse
                else block.condition(embedding).chunk(2, dim=-1)
            )
            conditioned = block.norm(hidden) * (1.0 + scale) + shift
            hidden = hidden + block.mlp(conditioned) / math.sqrt(2.0)
            conditions.append((scale, shift))
        velocity = model.output(hidden)
        diagonal = None
        if model.diagonal_velocity is not None:
            diagonal = (
                cached_diagonal if reuse else model.diagonal_velocity(embedding)
            )
            velocity = velocity + diagonal * state
        cached_key = key
        cached_blocks = tuple(conditions)
        cached_diagonal = diagonal
        return velocity

    step_size = 1.0 / int(steps)
    for index in range(int(steps)):
        time = torch.full(
            (int(count),),
            index / int(steps),
            device=device,
            dtype=values.dtype,
        )
        velocity = velocity_at_time(values, time, index)
        if guidance is not None and float(guidance_scale) != 0.0:
            velocity = velocity + float(guidance_scale) * guidance(
                values,
                time,
                velocity,
            )
        if method == "euler" or index + 1 == int(steps):
            values = values + step_size * velocity
            continue
        provisional = values + step_size * velocity
        next_time = torch.full_like(time, (index + 1) / int(steps))
        next_velocity = velocity_at_time(provisional, next_time, index + 1)
        if guidance is not None and float(guidance_scale) != 0.0:
            next_velocity = next_velocity + float(guidance_scale) * guidance(
                provisional,
                next_time,
                next_velocity,
            )
        values = values + 0.5 * step_size * (velocity + next_velocity)
    return values


def _continuous_time_embedding(torch, time, width: int):
    if time.ndim != 1:
        raise ValueError("structured flow time must have shape [batch]")
    half = int(width) // 2
    frequencies = torch.exp(
        math.log(1_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = 2.0 * math.pi * time[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if int(embedding.shape[-1]) < int(width):
        embedding = torch.nn.functional.pad(
            embedding,
            (0, int(width) - int(embedding.shape[-1])),
        )
    return embedding


def _rng_devices(values) -> list[int]:
    if not values.is_cuda:
        return []
    index = values.device.index
    return [0 if index is None else int(index)]
