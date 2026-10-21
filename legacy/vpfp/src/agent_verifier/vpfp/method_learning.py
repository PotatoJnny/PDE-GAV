from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MethodLearningConfig:
    architecture: str = "periodic_cnn"
    width: int = 64
    depth: int = 4
    modes: int = 8
    steps: int = 1600
    batch_size: int = 8
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    gradient_loss_weight: float = 0.2
    seed: int = 0

    def validate(self) -> None:
        if self.architecture not in {"periodic_cnn", "fno"}:
            raise ValueError("unsupported method-learning architecture")
        if min(self.width, self.depth, self.steps, self.batch_size) < 1:
            raise ValueError("network and training sizes must be positive")
        if self.modes < 1:
            raise ValueError("FNO mode count must be positive")
        if min(self.learning_rate, self.weight_decay) < 0.0:
            raise ValueError("optimizer values must be nonnegative")


def train_method_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    output: str | Path,
    config: MethodLearningConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train a compact direct or residual macro operator."""

    torch, nn = _torch_modules()
    config.validate()
    _validate_training_arrays(train_features, train_targets)
    _validate_training_arrays(validation_features, validation_targets)
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature channels differ")
    if train_targets.shape[1] != validation_targets.shape[1]:
        raise ValueError("training and validation target channels differ")

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    target_device = torch.device(device)
    x_mean = np.mean(train_features, axis=(0, 2, 3), keepdims=True)
    x_std = np.std(train_features, axis=(0, 2, 3), keepdims=True)
    x_std = np.maximum(x_std, 1.0e-5)
    y_scale = np.sqrt(
        np.mean(np.square(train_targets), axis=(0, 2, 3), keepdims=True)
    )
    y_scale = np.maximum(y_scale, 1.0e-7)

    train_x = torch.as_tensor(
        (train_features - x_mean) / x_std,
        dtype=torch.float32,
        device=target_device,
    )
    train_y = torch.as_tensor(
        train_targets / y_scale,
        dtype=torch.float32,
        device=target_device,
    )
    validation_x = torch.as_tensor(
        (validation_features - x_mean) / x_std,
        dtype=torch.float32,
        device=target_device,
    )
    validation_y = torch.as_tensor(
        validation_targets / y_scale,
        dtype=torch.float32,
        device=target_device,
    )
    model = build_method_model(
        config,
        input_channels=int(train_x.shape[1]),
        output_channels=int(train_y.shape[1]),
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.steps), eta_min=0.05 * config.learning_rate
    )
    generator = torch.Generator(device=target_device)
    generator.manual_seed(int(config.seed) + 7919)
    best_validation = float("inf")
    best_state = None
    history = []
    model.train()
    for step in range(int(config.steps)):
        indices = torch.randint(
            0,
            int(train_x.shape[0]),
            (min(int(config.batch_size), int(train_x.shape[0])),),
            generator=generator,
            device=target_device,
        )
        prediction = model(train_x[indices])
        loss = _normalized_loss(
            torch,
            prediction,
            train_y[indices],
            gradient_weight=float(config.gradient_loss_weight),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(config.steps):
            model.eval()
            with torch.no_grad():
                validation_prediction = model(validation_x)
                validation_loss = float(
                    _normalized_loss(
                        torch,
                        validation_prediction,
                        validation_y,
                        gradient_weight=float(config.gradient_loss_weight),
                    ).item()
                )
            history.append(
                {
                    "step": int(step + 1),
                    "train_loss": float(loss.item()),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise RuntimeError("method-learning training did not produce a checkpoint")
    checkpoint = {
        "schema": "vpfp_method_learning_checkpoint_v1",
        "config": asdict(config),
        "input_channels": int(train_x.shape[1]),
        "output_channels": int(train_y.shape[1]),
        "x_mean": np.asarray(x_mean, dtype=np.float32),
        "x_std": np.asarray(x_std, dtype=np.float32),
        "y_scale": np.asarray(y_scale, dtype=np.float32),
        "state_dict": best_state,
        "best_validation_loss": float(best_validation),
        "history": history,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return {
        "checkpoint": str(destination.resolve()),
        "best_validation_loss": float(best_validation),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "history": history,
    }


def predict_method_model(
    checkpoint_path: str | Path,
    features: np.ndarray,
    *,
    device: str = "cuda",
) -> np.ndarray:
    torch, _nn = _torch_modules()
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") != "vpfp_method_learning_checkpoint_v1":
        raise ValueError("unexpected method-learning checkpoint")
    config = MethodLearningConfig(**dict(checkpoint["config"]))
    model = build_method_model(
        config,
        input_channels=int(checkpoint["input_channels"]),
        output_channels=int(checkpoint["output_channels"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    target_device = torch.device(device)
    model.to(target_device).eval()
    value = np.asarray(features, dtype=np.float32)
    normalized = (
        value - np.asarray(checkpoint["x_mean"], dtype=np.float32)
    ) / np.asarray(checkpoint["x_std"], dtype=np.float32)
    with torch.no_grad():
        prediction = model(
            torch.as_tensor(normalized, dtype=torch.float32, device=target_device)
        ).cpu().numpy()
    return prediction * np.asarray(checkpoint["y_scale"], dtype=np.float32)


def fit_multifidelity_coefficients(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    ridge: float = 1.0e-3,
) -> np.ndarray:
    """Fit channel-wise first-minus-second corrections around second."""

    values = [np.asarray(item, dtype=np.float64) for item in (truth, first, second)]
    if any(item.shape != values[0].shape for item in values[1:]):
        raise ValueError("multifidelity arrays must have matching shapes")
    if values[0].ndim != 4:
        raise ValueError("multifidelity arrays must have shape (case,channel,x,y)")
    target = values[0] - values[2]
    difference = values[1] - values[2]
    numerator = np.sum(target * difference, axis=(0, 2, 3))
    denominator = np.sum(difference * difference, axis=(0, 2, 3))
    scale = float(ridge) * np.maximum(np.mean(denominator), 1.0e-20)
    coefficient = numerator / np.maximum(denominator + scale, 1.0e-20)
    return np.clip(coefficient, -1.0, 2.0).astype(np.float32)


def apply_multifidelity_coefficients(
    first: np.ndarray, second: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    first_value = np.asarray(first, dtype=np.float32)
    second_value = np.asarray(second, dtype=np.float32)
    coefficient = np.asarray(coefficients, dtype=np.float32)
    if first_value.shape != second_value.shape:
        raise ValueError("multifidelity predictions must have matching shapes")
    if coefficient.shape != (first_value.shape[1],):
        raise ValueError("multifidelity coefficient count does not match channels")
    return second_value + coefficient[None, :, None, None] * (
        first_value - second_value
    )


def build_method_model(
    config: MethodLearningConfig,
    *,
    input_channels: int,
    output_channels: int,
):
    _torch, nn = _torch_modules()
    if config.architecture == "periodic_cnn":
        return _PeriodicConvOperator(
            nn,
            input_channels=input_channels,
            output_channels=output_channels,
            width=int(config.width),
            depth=int(config.depth),
        )
    return _FourierOperator(
        nn,
        input_channels=input_channels,
        output_channels=output_channels,
        width=int(config.width),
        depth=int(config.depth),
        modes=int(config.modes),
    )


class _PeriodicConvOperator:
    def __new__(
        cls,
        nn,
        *,
        input_channels: int,
        output_channels: int,
        width: int,
        depth: int,
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(input_channels, width, 1)
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(
                                width,
                                width,
                                3,
                                padding=1,
                                padding_mode="circular",
                            ),
                            nn.GELU(),
                            nn.Conv2d(
                                width,
                                width,
                                3,
                                padding=1,
                                padding_mode="circular",
                            ),
                        )
                        for _ in range(depth)
                    ]
                )
                self.head = nn.Conv2d(width, output_channels, 1)

            def forward(self, value):
                import torch.nn.functional as functional

                value = functional.gelu(self.stem(value))
                for block in self.blocks:
                    value = value + 0.25 * block(value)
                return self.head(functional.gelu(value))

        return Model()


class _FourierOperator:
    def __new__(
        cls,
        nn,
        *,
        input_channels: int,
        output_channels: int,
        width: int,
        depth: int,
        modes: int,
    ):
        import torch

        class SpectralLayer(nn.Module):
            def __init__(self):
                super().__init__()
                scale = 1.0 / max(width, 1)
                self.weight = nn.Parameter(
                    scale
                    * torch.randn(
                        width,
                        width,
                        2 * modes - 1,
                        modes,
                        dtype=torch.cfloat,
                    )
                )
                self.local = nn.Conv2d(width, width, 1)

            def forward(self, value):
                transformed = torch.fft.rfft2(value, norm="ortho")
                nx = int(value.shape[-2])
                ny = int(value.shape[-1])
                kept_y = min(modes, int(transformed.shape[-1]))
                frequencies = list(range(min(modes, nx)))
                frequencies += list(range(max(nx - modes + 1, modes), nx))
                output = torch.zeros_like(transformed)
                if frequencies and kept_y:
                    index = torch.as_tensor(
                        frequencies, device=value.device, dtype=torch.long
                    )
                    selected = transformed[:, :, index, :kept_y]
                    weight = self.weight[:, :, : len(frequencies), :kept_y]
                    output[:, :, index, :kept_y] = torch.einsum(
                        "bcxy,cdxy->bdxy", selected, weight
                    )
                return torch.fft.irfft2(
                    output, s=(nx, ny), norm="ortho"
                ) + self.local(value)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(input_channels, width, 1)
                self.layers = nn.ModuleList([SpectralLayer() for _ in range(depth)])
                self.head = nn.Sequential(
                    nn.Conv2d(width, width, 1),
                    nn.GELU(),
                    nn.Conv2d(width, output_channels, 1),
                )

            def forward(self, value):
                import torch.nn.functional as functional

                value = functional.gelu(self.stem(value))
                for layer in self.layers:
                    value = functional.gelu(layer(value))
                return self.head(value)

        return Model()


def _normalized_loss(torch, prediction, target, *, gradient_weight: float):
    field = torch.mean(torch.square(prediction - target))
    if gradient_weight <= 0.0:
        return field
    prediction_dx = torch.roll(prediction, -1, -2) - prediction
    prediction_dy = torch.roll(prediction, -1, -1) - prediction
    target_dx = torch.roll(target, -1, -2) - target
    target_dy = torch.roll(target, -1, -1) - target
    gradient = 0.5 * (
        torch.mean(torch.square(prediction_dx - target_dx))
        + torch.mean(torch.square(prediction_dy - target_dy))
    )
    return field + float(gradient_weight) * gradient


def _validate_training_arrays(features: np.ndarray, targets: np.ndarray) -> None:
    feature_value = np.asarray(features)
    target_value = np.asarray(targets)
    if feature_value.ndim != 4 or target_value.ndim != 4:
        raise ValueError("method-learning arrays must be NCHW")
    if feature_value.shape[0] != target_value.shape[0]:
        raise ValueError("feature and target case counts differ")
    if feature_value.shape[-2:] != target_value.shape[-2:]:
        raise ValueError("feature and target spatial grids differ")
    if feature_value.shape[0] < 1:
        raise ValueError("method-learning arrays are empty")
    if not np.all(np.isfinite(feature_value)) or not np.all(np.isfinite(target_value)):
        raise FloatingPointError("method-learning arrays are non-finite")


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for method learning") from exc
    return torch, nn
