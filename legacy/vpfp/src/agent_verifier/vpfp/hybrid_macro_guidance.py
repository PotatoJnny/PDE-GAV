from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class VPFPHybridMacroRewardConfig:
    macro_feature_dim: int
    pca_rank: int = 32
    ridge_alpha: float = 0.1
    dtype: str = "float32"
    representation: str = "vpfp_hybrid_macro_log_reward_guidance_v1"

    def validate(self) -> None:
        if int(self.macro_feature_dim) < 1 or int(self.pca_rank) < 1:
            raise ValueError("macro reward dimensions must be positive")
        if int(self.pca_rank) > int(self.macro_feature_dim):
            raise ValueError("PCA rank cannot exceed the macro feature dimension")
        if float(self.ridge_alpha) <= 0.0:
            raise ValueError("ridge alpha must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("macro reward dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPHybridMacroRewardConfig":
        return cls(**dict(data))


class VPFPHybridMacroRewardModel(torch.nn.Module):
    """Fixed PCA-ridge regressor for absolute continuous solver reward."""

    def __init__(self, config: VPFPHybridMacroRewardConfig | dict[str, Any]):
        super().__init__()
        self.config = (
            config
            if isinstance(config, VPFPHybridMacroRewardConfig)
            else VPFPHybridMacroRewardConfig.from_dict(dict(config))
        )
        self.config.validate()
        dtype = torch.float64 if self.config.dtype == "float64" else torch.float32
        feature_dim = int(self.config.macro_feature_dim)
        rank = int(self.config.pca_rank)
        self.register_buffer("feature_mean", torch.zeros(feature_dim, dtype=dtype))
        self.register_buffer("feature_scale", torch.ones(feature_dim, dtype=dtype))
        self.register_buffer("pca_components", torch.zeros(rank, feature_dim, dtype=dtype))
        self.register_buffer("score_mean", torch.zeros(rank, dtype=dtype))
        self.register_buffer("score_scale", torch.ones(rank, dtype=dtype))
        self.register_buffer("coefficient", torch.zeros(rank, dtype=dtype))
        self.register_buffer("target_mean", torch.zeros((), dtype=dtype))

    def standardized_scores(self, macro_features):
        values = torch.as_tensor(
            macro_features, dtype=self.feature_mean.dtype, device=self.feature_mean.device
        )
        if values.ndim != 2 or int(values.shape[1]) != int(self.config.macro_feature_dim):
            raise ValueError("macro features have the wrong shape")
        normalized = (values - self.feature_mean) / self.feature_scale
        scores = normalized @ self.pca_components.T
        return (scores - self.score_mean) / self.score_scale

    def forward(self, macro_features):
        return self.standardized_scores(macro_features) @ self.coefficient + self.target_mean


def fit_hybrid_macro_reward_model(
    train_features,
    train_log_rewards,
    *,
    pca_rank: int = 32,
    ridge_alpha: float = 0.1,
    dtype: str = "float32",
) -> VPFPHybridMacroRewardModel:
    features = np.asarray(train_features, dtype=np.float64)
    target = np.asarray(train_log_rewards, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != target.size:
        raise ValueError("macro training features and targets do not align")
    if features.shape[0] < 3 or not (
        np.all(np.isfinite(features)) and np.all(np.isfinite(target))
    ):
        raise ValueError("macro reward training data are insufficient or non-finite")
    rank = min(int(pca_rank), features.shape[0] - 1, features.shape[1])
    if rank < 1:
        raise ValueError("effective PCA rank must be positive")

    feature_mean = np.mean(features, axis=0)
    feature_scale = np.maximum(np.std(features, axis=0), 1.0e-6)
    normalized = (features - feature_mean) / feature_scale
    _u, _s, vh = np.linalg.svd(normalized, full_matrices=False)
    components = vh[:rank]
    scores = normalized @ components.T
    score_mean = np.mean(scores, axis=0)
    score_scale = np.maximum(np.std(scores, axis=0), 1.0e-6)
    standardized = (scores - score_mean) / score_scale

    target_mean = float(np.mean(target))
    centered = target - target_mean
    gram = standardized.T @ standardized
    coefficient = np.linalg.solve(
        gram + float(ridge_alpha) * np.eye(rank),
        standardized.T @ centered,
    )
    config = VPFPHybridMacroRewardConfig(
        macro_feature_dim=int(features.shape[1]),
        pca_rank=rank,
        ridge_alpha=float(ridge_alpha),
        dtype=str(dtype),
    )
    model = VPFPHybridMacroRewardModel(config)
    values = {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "pca_components": components,
        "score_mean": score_mean,
        "score_scale": score_scale,
        "coefficient": coefficient,
        "target_mean": np.asarray(target_mean),
    }
    with torch.no_grad():
        for name, value in values.items():
            destination = getattr(model, name)
            source = torch.as_tensor(value, dtype=destination.dtype)
            if tuple(source.shape) != tuple(destination.shape):
                raise RuntimeError(f"fitted macro reward parameter {name} has wrong shape")
            destination.copy_(source)
    model.eval()
    return model


def macro_reward_regression_metrics(predicted, expected) -> dict[str, float | bool]:
    prediction = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(expected, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("macro reward predictions and targets do not align")
    finite = bool(np.all(np.isfinite(prediction)) and np.all(np.isfinite(target)))
    if not finite:
        return {
            "finite": False,
            "correlation": float("nan"),
            "spearman_correlation": float("nan"),
            "mse": float("nan"),
        }
    return {
        "finite": True,
        "correlation": _correlation(prediction, target),
        "spearman_correlation": _correlation(_ranks(prediction), _ranks(target)),
        "mse": float(np.mean((prediction - target) ** 2)),
    }


def select_reward_tilted_candidates(
    model: VPFPHybridMacroRewardModel,
    macro_features,
    *,
    group_count: int,
    pool_size: int,
) -> dict[str, torch.Tensor]:
    if min(int(group_count), int(pool_size)) < 1:
        raise ValueError("candidate group and pool sizes must be positive")
    prediction = model(macro_features)
    if prediction.numel() != int(group_count) * int(pool_size):
        raise ValueError("candidate feature count does not match grouped pool shape")
    grouped = prediction.reshape(int(group_count), int(pool_size))
    selected_within_group = torch.argmax(grouped, dim=1)
    offsets = torch.arange(
        int(group_count), dtype=torch.long, device=grouped.device
    ) * int(pool_size)
    selected_flat = offsets + selected_within_group
    return {
        "predicted_log_reward": prediction,
        "grouped_predicted_log_reward": grouped,
        "selected_within_group": selected_within_group,
        "selected_flat_indices": selected_flat,
        "selected_predicted_log_reward": prediction.index_select(0, selected_flat),
    }


def save_hybrid_macro_reward_checkpoint(
    path: str | Path,
    *,
    model: VPFPHybridMacroRewardModel,
    train_config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_hybrid_macro_reward_checkpoint_v1",
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "train_config": dict(train_config),
        },
        destination,
    )


def load_hybrid_macro_reward_checkpoint(path: str | Path, *, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_hybrid_macro_reward_checkpoint_v1":
        raise ValueError("unexpected hybrid macro reward checkpoint schema")
    model = VPFPHybridMacroRewardModel(payload["model_config"]).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _correlation(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
