from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.triplane import TriplaneBatch, plane_shapes

Array = np.ndarray


@dataclass
class NumpyVPFPTriplaneAutoencoder:
    config: Any
    payload: dict[str, Any]

    backend: str = "numpy_pairwise_mean_triplane_autoencoder"

    def encode_features(self, features: Array) -> TriplaneBatch:
        return encode_features_pairwise(np.asarray(features, dtype=np.float32))

    def decode_planes(self, planes: TriplaneBatch | tuple[Array, Array, Array], *, nt: int | None = None, nx: int | None = None, ny: int | None = None) -> Array:
        return decode_features_pairwise(planes, nt=nt, nx=nx, ny=ny)

    def reconstruct_features(self, features: Array) -> Array:
        arr = np.asarray(features, dtype=np.float32)
        return self.decode_planes(self.encode_features(arr), nt=arr.shape[1], nx=arr.shape[2], ny=arr.shape[3])

    def eval(self):
        return self


def encode_features_pairwise(features: Array) -> TriplaneBatch:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"features must have shape [C,T,X,Y], got {arr.shape}")
    mean = np.mean(arr, axis=(1, 2, 3), keepdims=True)
    p_xy = np.mean(arr, axis=1)
    p_xt = (np.mean(arr, axis=3, keepdims=False) - mean[..., 0]).transpose(0, 2, 1)
    p_yt = (np.mean(arr, axis=2, keepdims=False) - mean[..., 0, :]).transpose(0, 2, 1)
    return TriplaneBatch(p_xy.astype(np.float32), p_xt.astype(np.float32), p_yt.astype(np.float32))


def decode_features_pairwise(planes: TriplaneBatch | tuple[Array, Array, Array], *, nt: int | None = None, nx: int | None = None, ny: int | None = None) -> Array:
    p_xy, p_xt, p_yt = planes.as_tuple() if isinstance(planes, TriplaneBatch) else planes
    xy = np.asarray(p_xy, dtype=np.float32)
    xt = np.asarray(p_xt, dtype=np.float32)
    yt = np.asarray(p_yt, dtype=np.float32)
    if xy.ndim != 3 or xt.ndim != 3 or yt.ndim != 3:
        raise ValueError("single-case triplane decode expects [C,X,Y], [C,X,T], [C,Y,T] arrays")
    c, x_count, y_count = xy.shape
    if xt.shape[0] != c or yt.shape[0] != c:
        raise ValueError("triplane channel counts do not match")
    t_count = int(nt or xt.shape[2])
    x_count = int(nx or x_count)
    y_count = int(ny or y_count)
    if xy.shape[1:] != (x_count, y_count) or xt.shape[1:] != (x_count, t_count) or yt.shape[1:] != (y_count, t_count):
        raise ValueError(
            f"NumPy triplane decode does not resample shapes; got p_xy={xy.shape}, p_xt={xt.shape}, p_yt={yt.shape}, "
            f"requested nt/nx/ny={(t_count, x_count, y_count)}"
        )
    return (
        xy[:, None, :, :]
        + xt.transpose(0, 2, 1)[:, :, :, None]
        + yt.transpose(0, 2, 1)[:, :, None, :]
    ).astype(np.float32)


def save_numpy_autoencoder_checkpoint(path: str | Path, *, config: Any, train_config: dict[str, Any], stats: dict[str, Any], step: int) -> None:
    payload = {
        "checkpoint_backend": NumpyVPFPTriplaneAutoencoder.backend,
        "model_config": config.to_dict(),
        "train_config": dict(train_config),
        "stats": dict(stats),
        "step": int(step),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_numpy_autoencoder_checkpoint(path: str | Path, *, config_loader) -> tuple[NumpyVPFPTriplaneAutoencoder, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    cfg = config_loader(payload["model_config"])
    return NumpyVPFPTriplaneAutoencoder(config=cfg, payload=payload), payload


def infer_json_checkpoint_backend(path: str | Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    backend = payload.get("checkpoint_backend")
    return str(backend) if backend is not None else None


def flatten_planes(planes: TriplaneBatch) -> Array:
    return np.concatenate(
        [
            np.asarray(planes.p_xy, dtype=np.float32).reshape(-1),
            np.asarray(planes.p_xt, dtype=np.float32).reshape(-1),
            np.asarray(planes.p_yt, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def unflatten_planes(vector: Array, shapes: dict[str, list[int]]) -> TriplaneBatch:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    pieces = {}
    offset = 0
    for name in ("p_xy", "p_xt", "p_yt"):
        shape = tuple(int(v) for v in shapes[name])
        size = int(np.prod(shape))
        pieces[name] = arr[offset : offset + size].reshape(shape).astype(np.float32)
        offset += size
    if offset != arr.size:
        raise ValueError(f"unused flattened triplane entries: used {offset}, vector has {arr.size}")
    return TriplaneBatch(pieces["p_xy"], pieces["p_xt"], pieces["p_yt"])


def fit_triplane_pca(rows: list[dict[str, Any]], *, latent_dim: int, min_std: float = 1.0e-6) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot fit VPFP triplane PCA generator with no rows")
    planes = [_load_npz_planes(row["triplane_path"]) for row in rows]
    shapes = plane_shapes(planes[0])
    matrix = np.stack([flatten_planes(item) for item in planes], axis=0).astype(np.float64)
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean[None, :]
    std = np.std(matrix, axis=0)
    std = np.maximum(std, float(min_std))
    normalized = centered / std[None, :]
    if normalized.shape[0] <= 1:
        components = np.zeros((1, normalized.shape[1]), dtype=np.float32)
        latent_scale = np.ones((1,), dtype=np.float32)
        latent_scores = np.zeros((normalized.shape[0], 1), dtype=np.float32)
        explained = np.zeros((1,), dtype=np.float32)
    else:
        _u, s, vt = np.linalg.svd(normalized, full_matrices=False)
        keep = max(1, min(int(latent_dim), int(vt.shape[0]), int(normalized.shape[0] - 1)))
        components = vt[:keep].astype(np.float32)
        latent_scale = (s[:keep] / np.sqrt(max(normalized.shape[0] - 1, 1))).astype(np.float32)
        latent_scores = (normalized @ components.T).astype(np.float32)
        total = float(np.sum(s**2))
        explained = (s[:keep] ** 2 / total).astype(np.float32) if total > 0.0 else np.zeros((keep,), dtype=np.float32)
    return {
        "checkpoint_backend": "numpy_triplane_pca_generator",
        "plane_shapes": shapes,
        "case_count": int(matrix.shape[0]),
        "vector_dim": int(matrix.shape[1]),
        "latent_dim": int(components.shape[0]),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "components": components,
        "latent_scale": latent_scale,
        "latent_scores": latent_scores,
        "explained_variance_ratio": explained,
    }


def save_triplane_pca_checkpoint(path: str | Path, *, model: dict[str, Any], train_config: dict[str, Any], encoded_manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_backend": model["checkpoint_backend"],
        "plane_shapes": model["plane_shapes"],
        "case_count": model["case_count"],
        "vector_dim": model["vector_dim"],
        "latent_dim": model["latent_dim"],
        "explained_variance_ratio": [float(v) for v in np.asarray(model["explained_variance_ratio"]).reshape(-1)],
        "train_config": dict(train_config),
        "autoencoder_checkpoint": encoded_manifest.get("autoencoder_checkpoint"),
        "encoded_schema": encoded_manifest.get("encoded_schema"),
    }
    with target.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            mean=np.asarray(model["mean"], dtype=np.float32),
            std=np.asarray(model["std"], dtype=np.float32),
            components=np.asarray(model["components"], dtype=np.float32),
            latent_scale=np.asarray(model["latent_scale"], dtype=np.float32),
            latent_scores=np.asarray(model.get("latent_scores", np.zeros((0, int(model["latent_dim"])), dtype=np.float32)), dtype=np.float32),
        )


def load_triplane_pca_checkpoint(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        return {
            "checkpoint_backend": metadata["checkpoint_backend"],
            "metadata": metadata,
            "plane_shapes": metadata["plane_shapes"],
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "std": np.asarray(data["std"], dtype=np.float32),
            "components": np.asarray(data["components"], dtype=np.float32),
            "latent_scale": np.asarray(data["latent_scale"], dtype=np.float32),
            "latent_scores": np.asarray(data["latent_scores"], dtype=np.float32) if "latent_scores" in data else np.zeros((0, int(metadata["latent_dim"])), dtype=np.float32),
        }


def infer_npz_checkpoint_backend(path: str | Path) -> str | None:
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            if "metadata" not in data:
                return None
            metadata = json.loads(str(data["metadata"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    backend = metadata.get("checkpoint_backend")
    return str(backend) if backend is not None else None


def sample_triplane_pca(model: dict[str, Any], *, samples: int, rng: np.random.Generator, jitter_scale: float = 0.05) -> list[TriplaneBatch]:
    components = np.asarray(model["components"], dtype=np.float32)
    latent_scale = np.asarray(model["latent_scale"], dtype=np.float32)
    if components.ndim != 2:
        raise ValueError("PCA components must have shape [latent_dim, vector_dim]")
    scores = np.asarray(model.get("latent_scores", np.zeros((0, components.shape[0]), dtype=np.float32)), dtype=np.float32)
    if scores.ndim == 2 and scores.shape[0] > 0 and scores.shape[1] == components.shape[0]:
        indices = rng.integers(0, scores.shape[0], size=int(samples))
        latent = scores[indices].astype(np.float32)
        scale = max(0.0, float(jitter_scale))
        if scale > 0.0:
            latent = latent + scale * rng.normal(size=latent.shape).astype(np.float32) * latent_scale.reshape(1, -1)
    else:
        latent = rng.normal(size=(int(samples), int(components.shape[0]))).astype(np.float32) * latent_scale.reshape(1, -1)
    normalized = latent @ components
    vectors = np.asarray(model["mean"], dtype=np.float32)[None, :] + normalized * np.asarray(model["std"], dtype=np.float32)[None, :]
    return [unflatten_planes(vector, model["plane_shapes"]) for vector in vectors]


def _load_npz_planes(path: str | Path) -> TriplaneBatch:
    with np.load(Path(path)) as data:
        return TriplaneBatch(
            np.asarray(data["p_xy"], dtype=np.float32),
            np.asarray(data["p_xt"], dtype=np.float32),
            np.asarray(data["p_yt"], dtype=np.float32),
        )
