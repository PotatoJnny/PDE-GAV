from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


CONTINUOUS_TRIPLANE_MMS_ABI_VERSION = "continuous_triplane_mms_v1"
LEGACY_GRID_MMS_ABI_VERSION = "legacy_grid_mms_v0"
CANONICAL_GEOMETRY_ABI_VERSION = "canonical_geometry_sdf_wall_v2"
CANONICAL_GEOMETRY_CHANNELS = (
    "obstacle_signed_distance_over_domain_diagonal",
    "no_slip_wall_distance_over_domain_diagonal",
)
CANONICAL_GEOMETRY_CHANNEL_COUNT = len(CANONICAL_GEOMETRY_CHANNELS)


def canonical_geometry_abi_metadata() -> dict[str, Any]:
    return {
        "version": CANONICAL_GEOMETRY_ABI_VERSION,
        "channels": list(CANONICAL_GEOMETRY_CHANNELS),
        "channel_count": int(CANONICAL_GEOMETRY_CHANNEL_COUNT),
        "normalization": "geometry distances divided by supported-domain diagonal",
    }


@dataclass(frozen=True)
class FieldNormalizer:
    velocity_mean: tuple[float, float]
    velocity_std: tuple[float, float]
    pressure_mean: tuple[float]
    pressure_std: tuple[float]
    eps: float = 1.0e-6
    fitted_split: str = "train"
    fit_case_count: int = 0
    velocity_sample_count: int = 0
    pressure_sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi": "field_unit_gaussian_v1",
            "fitted_split": str(self.fitted_split),
            "fit_case_count": int(self.fit_case_count),
            "velocity_sample_count": int(self.velocity_sample_count),
            "pressure_sample_count": int(self.pressure_sample_count),
            "eps": float(self.eps),
            "velocity": {
                "channels": ["u", "v"],
                "mean": [float(v) for v in self.velocity_mean],
                "std": [float(v) for v in self.velocity_std],
            },
            "pressure": {
                "channels": ["pressure"],
                "mean": [float(v) for v in self.pressure_mean],
                "std": [float(v) for v in self.pressure_std],
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "FieldNormalizer | None":
        if not payload:
            return None
        velocity = payload.get("velocity", {})
        pressure = payload.get("pressure", {})
        velocity_mean = tuple(float(v) for v in velocity.get("mean", payload.get("velocity_mean", (0.0, 0.0))))
        velocity_std = tuple(float(v) for v in velocity.get("std", payload.get("velocity_std", (1.0, 1.0))))
        pressure_mean = tuple(float(v) for v in pressure.get("mean", payload.get("pressure_mean", (0.0,))))
        pressure_std = tuple(float(v) for v in pressure.get("std", payload.get("pressure_std", (1.0,))))
        return cls(
            velocity_mean=(velocity_mean + (0.0, 0.0))[:2],
            velocity_std=(velocity_std + (1.0, 1.0))[:2],
            pressure_mean=(pressure_mean + (0.0,))[:1],
            pressure_std=(pressure_std + (1.0,))[:1],
            eps=float(payload.get("eps", 1.0e-6)),
            fitted_split=str(payload.get("fitted_split", "train")),
            fit_case_count=int(payload.get("fit_case_count", 0)),
            velocity_sample_count=int(payload.get("velocity_sample_count", 0)),
            pressure_sample_count=int(payload.get("pressure_sample_count", 0)),
        )

    def normalize_velocity(self, velocity):
        return self._apply_velocity(velocity, inverse=False)

    def inverse_velocity(self, velocity):
        return self._apply_velocity(velocity, inverse=True)

    def normalize_pressure(self, pressure):
        return self._apply_pressure(pressure, inverse=False)

    def inverse_pressure(self, pressure):
        return self._apply_pressure(pressure, inverse=True)

    def _apply_velocity(self, velocity, *, inverse: bool):
        torch = _require_torch()
        tensor = velocity if torch.is_tensor(velocity) else torch.as_tensor(velocity)
        if tensor.ndim >= 3 and int(tensor.shape[-3]) == 2:
            view_shape = [1] * tensor.ndim
            view_shape[-3] = 2
        elif tensor.ndim >= 1 and int(tensor.shape[-1]) == 2:
            view_shape = [1] * tensor.ndim
            view_shape[-1] = 2
        else:
            raise ValueError(f"velocity tensor has no 2-channel velocity axis: {tuple(tensor.shape)}")
        mean = torch.as_tensor(self.velocity_mean, dtype=tensor.dtype, device=tensor.device).reshape(view_shape)
        std = torch.as_tensor(self.velocity_std, dtype=tensor.dtype, device=tensor.device).reshape(view_shape).clamp_min(float(self.eps))
        return tensor * std + mean if inverse else (tensor - mean) / std

    def _apply_pressure(self, pressure, *, inverse: bool):
        torch = _require_torch()
        tensor = pressure if torch.is_tensor(pressure) else torch.as_tensor(pressure)
        mean = torch.as_tensor(float(self.pressure_mean[0]), dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(float(self.pressure_std[0]), dtype=tensor.dtype, device=tensor.device).clamp_min(float(self.eps))
        return tensor * std + mean if inverse else (tensor - mean) / std


def fit_field_normalizer_from_case_dirs(
    case_dirs: Iterable[str | Path],
    load_case_fn: Callable[[str | Path], Any],
    *,
    max_cases: int = 0,
    eps: float = 1.0e-6,
    fitted_split: str = "train",
) -> FieldNormalizer:
    velocity_sum = np.zeros(2, dtype=np.float64)
    velocity_sumsq = np.zeros(2, dtype=np.float64)
    pressure_sum = 0.0
    pressure_sumsq = 0.0
    velocity_count = 0
    pressure_count = 0
    case_count = 0
    for raw_path in case_dirs:
        if int(max_cases) > 0 and case_count >= int(max_cases):
            break
        case = load_case_fn(raw_path)
        velocity = np.asarray(case.hidden_arrays["u_true"], dtype=np.float64)
        pressure = np.asarray(case.hidden_arrays["pressure"], dtype=np.float64)
        if velocity.ndim != 4 or int(velocity.shape[1]) != 2:
            raise ValueError(f"velocity must have shape [T,2,X,Y], got {velocity.shape}")
        if pressure.shape != (velocity.shape[0], velocity.shape[2], velocity.shape[3]):
            raise ValueError(f"pressure shape {pressure.shape} does not match velocity {velocity.shape}")
        flattened_velocity = np.moveaxis(velocity, 1, 0).reshape(2, -1)
        velocity_sum += np.sum(flattened_velocity, axis=1)
        velocity_sumsq += np.sum(np.square(flattened_velocity), axis=1)
        velocity_count += int(flattened_velocity.shape[1])
        pressure_sum += float(np.sum(pressure))
        pressure_sumsq += float(np.sum(np.square(pressure)))
        pressure_count += int(pressure.size)
        case_count += 1
    if case_count <= 0 or velocity_count <= 0 or pressure_count <= 0:
        raise ValueError("cannot fit field normalizer from an empty train split")
    velocity_mean = velocity_sum / float(velocity_count)
    velocity_var = np.maximum(velocity_sumsq / float(velocity_count) - np.square(velocity_mean), float(eps) ** 2)
    pressure_mean = pressure_sum / float(pressure_count)
    pressure_var = max(pressure_sumsq / float(pressure_count) - pressure_mean**2, float(eps) ** 2)
    return FieldNormalizer(
        velocity_mean=(float(velocity_mean[0]), float(velocity_mean[1])),
        velocity_std=(float(np.sqrt(velocity_var[0])), float(np.sqrt(velocity_var[1]))),
        pressure_mean=(float(pressure_mean),),
        pressure_std=(float(np.sqrt(pressure_var)),),
        eps=float(eps),
        fitted_split=str(fitted_split),
        fit_case_count=int(case_count),
        velocity_sample_count=int(velocity_count),
        pressure_sample_count=int(pressure_count),
    )


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for triplane autoencoder") from exc
    return torch


@dataclass(frozen=True)
class TriplaneAutoencoderConfig:
    input_channels: int = 3 + CANONICAL_GEOMETRY_CHANNEL_COUNT
    plane_channels: int = 64
    xy_resolution: tuple[int, int] = (64, 64)
    xt_resolution: tuple[int, int] = (64, 32)
    yt_resolution: tuple[int, int] = (64, 32)
    encoder_width: int = 160
    encoder_blocks: int = 4
    decoder_hidden_width: int = 384
    decoder_hidden_layers: int = 7
    coordinate_frequencies: int = 6
    encoder_summary_mode: str = "axis_unfold"
    decoder_streamfunction_output_scale: float = 0.01
    decoder_pressure_output_scale: float = 0.25
    decoder_separate_heads: bool = True
    decoder_head_hidden_width: int = 384
    decoder_head_hidden_layers: int = 2
    geometry_abi: str = CANONICAL_GEOMETRY_ABI_VERSION
    field_normalizer_abi: str = "field_unit_gaussian_v1"
    dtype: str = "float32"
    representation: str = "learned_projection"
    decoder_geometry_feature_dim: int = CANONICAL_GEOMETRY_CHANNEL_COUNT
    plane_interpolation: str = "quintic_bspline"
    continuous_mms_abi: str = CONTINUOUS_TRIPLANE_MMS_ABI_VERSION
    decoder_geometry_mask_transition: float = 0.03

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["xy_resolution"] = list(self.xy_resolution)
        data["xt_resolution"] = list(self.xt_resolution)
        data["yt_resolution"] = list(self.yt_resolution)
        return data


def normalize_triplane_autoencoder_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    # Preserve old checkpoint behavior while keeping it outside the formal
    # continuous MMS contract.
    if "continuous_mms_abi" not in out:
        out["continuous_mms_abi"] = LEGACY_GRID_MMS_ABI_VERSION
    if "plane_interpolation" not in out:
        out["plane_interpolation"] = "bilinear"
    if "decoder_geometry_feature_dim" not in out:
        out["decoder_geometry_feature_dim"] = 4
    if "decoder_geometry_mask_transition" not in out:
        out["decoder_geometry_mask_transition"] = 0.03
    for key in (
        "decoder_streamfunction_skip_scale",
        "decoder_pressure_skip_scale",
        "input_velocity_scale",
        "input_pressure_scale",
    ):
        out.pop(key, None)
    for key in ("xy_resolution", "xt_resolution", "yt_resolution"):
        if key in out:
            out[key] = tuple(out[key])
    return out


def is_formal_triplane_autoencoder_config(config: TriplaneAutoencoderConfig | dict[str, Any]) -> bool:
    cfg = config if isinstance(config, TriplaneAutoencoderConfig) else TriplaneAutoencoderConfig(**normalize_triplane_autoencoder_config_dict(dict(config)))
    return (
        str(cfg.representation) != "field_stack"
        and str(cfg.continuous_mms_abi) == CONTINUOUS_TRIPLANE_MMS_ABI_VERSION
        and str(cfg.plane_interpolation) == "quintic_bspline"
        and int(cfg.decoder_geometry_feature_dim) == CANONICAL_GEOMETRY_CHANNEL_COUNT
    )


def require_formal_triplane_autoencoder_config(
    config: TriplaneAutoencoderConfig | dict[str, Any],
    *,
    context: str,
) -> None:
    if is_formal_triplane_autoencoder_config(config):
        return
    raise ValueError(
        f"{context} requires a learned P_xy/P_xt/P_yt autoencoder with the "
        f"{CONTINUOUS_TRIPLANE_MMS_ABI_VERSION} ABI, C4 quintic B-spline interpolation, and canonical geometry; "
        "field-stack and bilinear/grid-finite-difference checkpoints are diagnostic-only legacy artifacts"
    )


@dataclass(frozen=True)
class AutoencoderLossConfig:
    lambda_vel: float = 1.0
    lambda_p: float = 0.02
    lambda_z: float = 1.0e-6


class TriplaneAutoencoder:
    @staticmethod
    def build(config: TriplaneAutoencoderConfig | dict[str, Any] | None = None):
        torch = _require_torch()
        cfg = config if isinstance(config, TriplaneAutoencoderConfig) else TriplaneAutoencoderConfig(**normalize_triplane_autoencoder_config_dict(dict(config or {})))
        return _TriplaneAutoencoder(torch, cfg)


class _TriplaneAutoencoder(_require_torch().nn.Module):
    def __init__(self, torch_module, config: TriplaneAutoencoderConfig):
        super().__init__()
        from agent_verifier.neural_generator.triplane_decoder import TriplaneDecoderConfig, _TriplaneDecoder

        self.torch = torch_module
        self.config = config
        self.encoder = FactorizedTriplaneEncoder.build(config)
        decoder_config = TriplaneDecoderConfig(
            plane_channels=config.plane_channels,
            hidden_width=config.decoder_hidden_width,
            hidden_layers=config.decoder_hidden_layers,
            coordinate_frequencies=config.coordinate_frequencies,
            geometry_feature_dim=int(config.decoder_geometry_feature_dim),
            plane_interpolation=str(config.plane_interpolation),
            continuous_mms_abi=str(config.continuous_mms_abi),
            geometry_mask_transition=float(config.decoder_geometry_mask_transition),
            streamfunction_output_scale=config.decoder_streamfunction_output_scale,
            pressure_output_scale=config.decoder_pressure_output_scale,
            separate_output_heads=bool(config.decoder_separate_heads),
            head_hidden_width=int(config.decoder_head_hidden_width),
            head_hidden_layers=int(config.decoder_head_hidden_layers),
            dtype=config.dtype,
            field_stack_decode=config.representation == "field_stack",
            field_stack_nt=int(config.xt_resolution[1]),
        )
        self.decoder = _TriplaneDecoder(torch_module, decoder_config)

    def encode(self, batch):
        return self.encoder(batch)

    def velocity_pressure(self, batch, coords, geometry_features):
        planes = self.encode(batch)
        return self.decoder.velocity_pressure(planes, coords, geometry_features)


class FactorizedTriplaneEncoder:
    @staticmethod
    def build(config: TriplaneAutoencoderConfig | dict[str, Any]):
        torch = _require_torch()
        cfg = config if isinstance(config, TriplaneAutoencoderConfig) else TriplaneAutoencoderConfig(**normalize_triplane_autoencoder_config_dict(dict(config)))
        return _FactorizedTriplaneEncoder(torch, cfg)


class _FactorizedTriplaneEncoder(_require_torch().nn.Module):
    def __init__(self, torch_module, config: TriplaneAutoencoderConfig):
        super().__init__()
        self.torch = torch_module
        self.config = config
        self.xy_branch = _make_branch(torch_module, _branch_input_channels(config, "xy"), config.encoder_width, config.plane_channels, config.encoder_blocks)
        self.xt_branch = _make_branch(torch_module, _branch_input_channels(config, "xt"), config.encoder_width, config.plane_channels, config.encoder_blocks)
        self.yt_branch = _make_branch(torch_module, _branch_input_channels(config, "yt"), config.encoder_width, config.plane_channels, config.encoder_blocks)
        dtype = torch_module.float64 if config.dtype == "float64" else torch_module.float32
        self.to(dtype=dtype)

    def forward(self, batch):
        torch = self.torch
        import torch.nn.functional as functional

        x = _ensure_5d(torch, batch)
        expected_channels = int(self.config.input_channels)
        if int(x.shape[1]) != expected_channels:
            raise ValueError(f"autoencoder input has {int(x.shape[1])} channels, expected {expected_channels}")
        if self.config.representation == "field_stack":
            return self._field_stack_planes(x)
        if self.config.encoder_summary_mode == "axis_unfold":
            xy = _axis_unfold(x, "xy", self.config)
            xt = _axis_unfold(x, "xt", self.config)
            yt = _axis_unfold(x, "yt", self.config)
        else:
            xy = _axis_summary(torch, x, dim=2, mode=self.config.encoder_summary_mode)
            xt = _axis_summary(torch, x, dim=4, mode=self.config.encoder_summary_mode).permute(0, 1, 3, 2)
            yt = _axis_summary(torch, x, dim=3, mode=self.config.encoder_summary_mode).permute(0, 1, 3, 2)
        p_xy = functional.interpolate(self.xy_branch(xy), size=tuple(self.config.xy_resolution), mode="bilinear", align_corners=True)
        p_xt = functional.interpolate(self.xt_branch(xt), size=tuple(self.config.xt_resolution), mode="bilinear", align_corners=True)
        p_yt = functional.interpolate(self.yt_branch(yt), size=tuple(self.config.yt_resolution), mode="bilinear", align_corners=True)
        return p_xy, p_xt, p_yt

    def _field_stack_planes(self, x):
        nt = int(x.shape[2])
        required = 3 * nt
        if int(self.config.plane_channels) < required:
            raise ValueError(f"field_stack representation requires at least {required} plane channels for nt={nt}")
        p_xy = self.torch.zeros(
            (x.shape[0], int(self.config.plane_channels), int(self.config.xy_resolution[0]), int(self.config.xy_resolution[1])),
            dtype=x.dtype,
            device=x.device,
        )
        p_xt = self.torch.zeros(
            (x.shape[0], int(self.config.plane_channels), int(self.config.xt_resolution[0]), int(self.config.xt_resolution[1])),
            dtype=x.dtype,
            device=x.device,
        )
        p_yt = self.torch.zeros(
            (x.shape[0], int(self.config.plane_channels), int(self.config.yt_resolution[0]), int(self.config.yt_resolution[1])),
            dtype=x.dtype,
            device=x.device,
        )
        if tuple(p_xy.shape[-2:]) != tuple(x.shape[-2:]):
            import torch.nn.functional as functional

            u_stack = functional.interpolate(x[:, 0], size=tuple(self.config.xy_resolution), mode="bilinear", align_corners=True)
            v_stack = functional.interpolate(x[:, 1], size=tuple(self.config.xy_resolution), mode="bilinear", align_corners=True)
            p_stack = functional.interpolate(x[:, 2], size=tuple(self.config.xy_resolution), mode="bilinear", align_corners=True)
        else:
            u_stack = x[:, 0]
            v_stack = x[:, 1]
            p_stack = x[:, 2]
        p_xy[:, 0:nt] = u_stack
        p_xy[:, nt : 2 * nt] = v_stack
        p_xy[:, 2 * nt : 3 * nt] = p_stack
        return p_xy, p_xt, p_yt


def _branch_input_channels(config: TriplaneAutoencoderConfig, branch: str) -> int:
    c = int(config.input_channels)
    mode = str(config.encoder_summary_mode)
    if mode == "mean":
        return c
    if mode == "axis_unfold":
        if branch == "xy":
            return c * int(config.xt_resolution[1])
        if branch == "xt":
            return c * int(config.xy_resolution[1])
        if branch == "yt":
            return c * int(config.xy_resolution[0])
    raise ValueError(f"unsupported encoder summary mode: {mode}")


def _axis_summary(torch, x, *, dim: int, mode: str):
    if mode == "mean":
        return torch.mean(x, dim=dim)
    raise ValueError(f"unsupported encoder summary mode: {mode}")


def _axis_unfold(x, branch: str, config: TriplaneAutoencoderConfig):
    b, c, t, nx, ny = x.shape
    if int(t) != int(config.xt_resolution[1]):
        raise ValueError(f"axis_unfold expected T={int(config.xt_resolution[1])}, got {int(t)}")
    if int(nx) != int(config.xy_resolution[0]) or int(ny) != int(config.xy_resolution[1]):
        raise ValueError(
            f"axis_unfold expected grid {(int(config.xy_resolution[0]), int(config.xy_resolution[1]))}, got {(int(nx), int(ny))}"
        )
    if branch == "xy":
        return x.reshape(b, c * t, nx, ny)
    if branch == "xt":
        return x.permute(0, 1, 4, 3, 2).reshape(b, c * ny, nx, t)
    if branch == "yt":
        return x.permute(0, 1, 3, 4, 2).reshape(b, c * nx, ny, t)
    raise ValueError(f"unknown triplane branch: {branch}")


def make_autoencoder_input(
    velocity,
    pressure,
    *,
    field_normalizer: FieldNormalizer,
    geometry=None,
    signed_distance=None,
    x=None,
    y=None,
    domain: tuple[float, float] | None = None,
    outer_boundary_mask=None,
    canonical_geometry=None,
):
    torch = _require_torch()
    if field_normalizer is None:
        raise ValueError("field_normalizer is required; fit it on the train split and store it with the checkpoint")
    vel = velocity if torch.is_tensor(velocity) else torch.as_tensor(velocity)
    prs = pressure if torch.is_tensor(pressure) else torch.as_tensor(pressure)
    if vel.ndim == 4:
        vel = vel[None, ...]
    if prs.ndim == 3:
        prs = prs[None, ...]
    if vel.ndim != 5 or vel.shape[2] != 2:
        raise ValueError(f"velocity must be [B,T,2,X,Y] or [T,2,X,Y], got {tuple(vel.shape)}")
    if prs.shape != (vel.shape[0], vel.shape[1], vel.shape[3], vel.shape[4]):
        raise ValueError(f"pressure {tuple(prs.shape)} does not match velocity {tuple(vel.shape)}")
    vel = field_normalizer.normalize_velocity(vel)
    prs = field_normalizer.normalize_pressure(prs)
    device = vel.device
    dtype = vel.dtype
    geom_stack = canonical_geometry
    if geom_stack is None:
        geom_stack = canonical_geometry_tensor(
            geometry=geometry,
            signed_distance=signed_distance,
            x=x,
            y=y,
            domain=domain,
            outer_boundary_mask=outer_boundary_mask,
            dtype=dtype,
            device=device,
        )
    else:
        geom_stack = torch.as_tensor(geom_stack, dtype=dtype, device=device)
        if geom_stack.ndim == 2:
            geom_stack = geom_stack[None, ...]
    expected_geom_shape = (CANONICAL_GEOMETRY_CHANNEL_COUNT, vel.shape[3], vel.shape[4])
    if tuple(geom_stack.shape) != expected_geom_shape:
        raise ValueError(f"canonical geometry tensor must have shape {expected_geom_shape}, got {tuple(geom_stack.shape)}")
    geom_stack = geom_stack[None, None, ...].expand(vel.shape[0], vel.shape[1], -1, -1, -1)
    fields = torch.cat([vel, prs[:, :, None, :, :], geom_stack], dim=2)
    return fields.permute(0, 2, 1, 3, 4).contiguous()


def canonical_geometry_tensor(
    *,
    geometry=None,
    signed_distance=None,
    x=None,
    y=None,
    domain: tuple[float, float] | None = None,
    outer_boundary_mask=None,
    dtype=None,
    device=None,
):
    torch = _require_torch()
    if signed_distance is None and geometry is not None:
        if isinstance(geometry, dict):
            signed_distance = geometry.get("obstacle_signed_distance")
        else:
            signed_distance = getattr(geometry, "obstacle_signed_distance", None)
    if outer_boundary_mask is None and geometry is not None:
        if isinstance(geometry, dict):
            outer_boundary_mask = geometry.get("outer_boundary_mask")
        else:
            outer_boundary_mask = getattr(geometry, "outer_boundary_mask", None)
    if signed_distance is None:
        raise ValueError("canonical geometry ABI requires obstacle_signed_distance for supported geometries")
    sdf = torch.as_tensor(signed_distance, dtype=dtype, device=device)
    if sdf.ndim != 2:
        raise ValueError(f"obstacle_signed_distance must be [X,Y], got {tuple(sdf.shape)}")
    shape = tuple(int(v) for v in sdf.shape)
    diagonal = _supported_domain_diagonal(geometry=geometry, x=x, y=y, domain=domain, shape=shape)
    wall_distance = _no_slip_wall_distance_tensor(
        geometry=geometry,
        x=x,
        y=y,
        domain=domain,
        outer_boundary_mask=outer_boundary_mask,
        shape=shape,
        dtype=sdf.dtype,
        device=sdf.device,
    )
    if outer_boundary_mask is not None:
        wall_mask = torch.as_tensor(outer_boundary_mask, dtype=torch.bool, device=wall_distance.device)
        wall_distance = wall_distance.masked_fill(wall_mask, 0.0)
    return torch.stack([sdf / float(diagonal), wall_distance / float(diagonal)], dim=0)


def _no_slip_wall_distance_tensor(
    *,
    geometry=None,
    x=None,
    y=None,
    domain: tuple[float, float] | None,
    outer_boundary_mask=None,
    shape: tuple[int, int],
    dtype,
    device,
):
    torch = _require_torch()
    axes = _infer_cell_center_axes(geometry=geometry, x=x, y=y, domain=domain, shape=shape)
    if axes is not None:
        x_centers, y_centers, x_min, x_max, y_min, y_max = axes
        x_t = torch.as_tensor(x_centers, dtype=dtype, device=device)
        y_t = torch.as_tensor(y_centers, dtype=dtype, device=device)
        wall_x = torch.minimum(x_t - float(x_min), float(x_max) - x_t)[:, None]
        wall_y = torch.minimum(y_t - float(y_min), float(y_max) - y_t)[None, :]
        return torch.minimum(wall_x, wall_y).clamp_min(0.0)
    if outer_boundary_mask is not None:
        return torch.as_tensor(_wall_distance_from_mask(outer_boundary_mask), dtype=dtype, device=device)
    raise ValueError("canonical geometry ABI v2 requires x/y, domain bounds, or outer_boundary_mask to derive no-slip wall distance")


def _infer_cell_center_axes(*, geometry=None, x=None, y=None, domain: tuple[float, float] | None, shape: tuple[int, int]):
    if geometry is not None:
        if x is None:
            x = geometry.get("x") if isinstance(geometry, dict) else getattr(geometry, "x", None)
        if y is None:
            y = geometry.get("y") if isinstance(geometry, dict) else getattr(geometry, "y", None)
    if x is not None and y is not None:
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        if xx.size == int(shape[0]) and yy.size == int(shape[1]):
            dx = float(np.mean(np.diff(xx))) if xx.size >= 2 else 1.0
            dy = float(np.mean(np.diff(yy))) if yy.size >= 2 else 1.0
            return xx, yy, float(xx[0] - 0.5 * dx), float(xx[-1] + 0.5 * dx), float(yy[0] - 0.5 * dy), float(yy[-1] + 0.5 * dy)
    cfg = None if geometry is None or isinstance(geometry, dict) else getattr(geometry, "config", None)
    if cfg is not None and all(hasattr(cfg, name) for name in ("x_min", "x_max", "y_min", "y_max")):
        x_min = float(cfg.x_min)
        x_max = float(cfg.x_max)
        y_min = float(cfg.y_min)
        y_max = float(cfg.y_max)
        x_centers = x_min + (np.arange(int(shape[0]), dtype=np.float64) + 0.5) * (x_max - x_min) / float(shape[0])
        y_centers = y_min + (np.arange(int(shape[1]), dtype=np.float64) + 0.5) * (y_max - y_min) / float(shape[1])
        return x_centers, y_centers, x_min, x_max, y_min, y_max
    if domain is not None:
        width, height = float(domain[0]), float(domain[1])
        if width > 0.0 and height > 0.0:
            x_centers = (np.arange(int(shape[0]), dtype=np.float64) + 0.5) * width / float(shape[0])
            y_centers = (np.arange(int(shape[1]), dtype=np.float64) + 0.5) * height / float(shape[1])
            return x_centers, y_centers, 0.0, width, 0.0, height
    return None


def _wall_distance_from_mask(outer_boundary_mask) -> np.ndarray:
    mask = np.asarray(outer_boundary_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"outer_boundary_mask must be [X,Y], got {mask.shape}")
    if not np.any(mask):
        raise ValueError("outer_boundary_mask has no no-slip wall points")
    try:
        from scipy import ndimage  # type: ignore

        return ndimage.distance_transform_edt(~mask).astype(np.float32)
    except Exception:  # noqa: BLE001
        wall = np.argwhere(mask)
        grid = np.indices(mask.shape).reshape(2, -1).T
        distances = np.sqrt(np.min(np.sum((grid[:, None, :] - wall[None, :, :]) ** 2, axis=2), axis=1))
        return distances.reshape(mask.shape).astype(np.float32)


def _supported_domain_diagonal(*, geometry=None, x=None, y=None, domain: tuple[float, float] | None, shape: tuple[int, int]) -> float:
    axes = _infer_cell_center_axes(geometry=geometry, x=x, y=y, domain=domain, shape=shape)
    if axes is not None:
        _x_centers, _y_centers, x_min, x_max, y_min, y_max = axes
        width = float(x_max - x_min)
        height = float(y_max - y_min)
        diagonal = float(np.sqrt(width * width + height * height))
        if diagonal > 0.0:
            return diagonal
    raise ValueError(
        "cannot infer supported-domain diagonal for canonical geometry; pass geometry with config bounds, x/y arrays, or domain=(width,height)"
    )


def autoencoder_loss_a0(pred_u, pred_v, pred_pressure, target_velocity, target_pressure, planes, config: AutoencoderLossConfig | None = None):
    torch = _require_torch()
    cfg = config or AutoencoderLossConfig()
    target = target_velocity if torch.is_tensor(target_velocity) else torch.as_tensor(target_velocity)
    target_p = target_pressure if torch.is_tensor(target_pressure) else torch.as_tensor(target_pressure)
    if target.ndim == 4:
        target = target[None, ...]
    if target_p.ndim == 3:
        target_p = target_p[None, ...]
    if target.ndim == 5:
        pred_vel = torch.stack([pred_u, pred_v], dim=2)
    elif target.ndim == 3:
        pred_vel = torch.stack([pred_u, pred_v], dim=-1)
    else:
        raise ValueError(f"target velocity must be [B,T,2,X,Y] or [B,N,2], got {tuple(target.shape)}")
    loss_u = torch.mean((pred_vel - target) ** 2)
    loss_p = gauge_centered_mse(pred_pressure, target_p)
    loss_z = plane_l2(planes)
    total = float(cfg.lambda_vel) * loss_u + float(cfg.lambda_p) * loss_p + float(cfg.lambda_z) * loss_z
    return {
        "loss": total,
        "velocity_mse": loss_u,
        "pressure_gauge_mse": loss_p,
        "latent_l2": loss_z,
    }


def normalized_mse(pred, target):
    torch = _require_torch()
    pred = pred if torch.is_tensor(pred) else torch.as_tensor(pred)
    target = target if torch.is_tensor(target) else torch.as_tensor(target)
    if pred.ndim <= 1:
        return torch.mean((pred - target) ** 2) / (torch.mean(target**2) + 1.0e-12)
    reduce_dims = tuple(range(1, pred.ndim))
    numerator = torch.mean((pred - target) ** 2, dim=reduce_dims)
    denominator = torch.mean(target**2, dim=reduce_dims) + 1.0e-12
    return torch.mean(numerator / denominator)


def gauge_centered_mse(pred, target):
    torch = _require_torch()
    pred = pred if torch.is_tensor(pred) else torch.as_tensor(pred)
    target = target if torch.is_tensor(target) else torch.as_tensor(target)
    if pred.ndim == 2:
        center_dims = (1,)
    else:
        center_dims = (-2, -1)
    pred_centered = pred - torch.mean(pred, dim=center_dims, keepdim=True)
    target_centered = target - torch.mean(target, dim=center_dims, keepdim=True)
    return normalized_mse(pred_centered, target_centered)


def plane_l2(planes) -> Any:
    torch = _require_torch()
    tensors = tuple(planes.as_tuple()) if hasattr(planes, "as_tuple") else tuple(planes)
    total = torch.zeros((), dtype=tensors[0].dtype, device=tensors[0].device)
    for plane in tensors:
        total = total + torch.mean(plane**2)
    return total / float(len(tensors))


def _make_branch(torch, in_channels: int, width: int, out_channels: int, blocks: int):
    layers = [torch.nn.Conv2d(in_channels, width, kernel_size=3, padding=1), torch.nn.SiLU()]
    for _ in range(max(0, int(blocks) - 1)):
        layers.extend([torch.nn.Conv2d(width, width, kernel_size=3, padding=1), torch.nn.SiLU()])
    layers.append(torch.nn.Conv2d(width, out_channels, kernel_size=1))
    return torch.nn.Sequential(*layers)


def _ensure_5d(torch, batch):
    x = batch if torch.is_tensor(batch) else torch.as_tensor(batch)
    if x.ndim != 5:
        raise ValueError(f"autoencoder input must be [B,C,T,X,Y], got {tuple(x.shape)}")
    return x
