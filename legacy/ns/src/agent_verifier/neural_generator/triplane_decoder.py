from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for triplane neural generator modules") from exc
    return torch


@dataclass(frozen=True)
class TriplaneDecoderConfig:
    plane_channels: int = 64
    hidden_width: int = 384
    hidden_layers: int = 7
    coordinate_frequencies: int = 6
    geometry_feature_dim: int = 2
    activation: str = "silu"
    dtype: str = "float32"
    hard_mask_alpha: float = 2.0
    obstacle_mask_epsilon: float = 0.04
    pressure_output_scale: float = 0.25
    streamfunction_output_scale: float = 0.01
    separate_output_heads: bool = True
    head_hidden_width: int = 384
    head_hidden_layers: int = 2
    field_stack_decode: bool = False
    field_stack_nt: int = 32

    plane_interpolation: str = "quintic_bspline"
    continuous_mms_abi: str = "continuous_triplane_mms_v1"
    geometry_mask_transition: float = 0.03

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriplaneBatch:
    p_xy: Any
    p_xt: Any
    p_yt: Any

    def as_tuple(self):
        return self.p_xy, self.p_xt, self.p_yt


class TriplaneDecoder:
    @staticmethod
    def build(config: TriplaneDecoderConfig | dict[str, Any] | None = None):
        torch = _require_torch()
        cfg = config if isinstance(config, TriplaneDecoderConfig) else TriplaneDecoderConfig(**dict(config or {}))
        return _TriplaneDecoder(torch, cfg)


class _TriplaneDecoder(_require_torch().nn.Module):
    def __init__(self, torch_module, config: TriplaneDecoderConfig):
        super().__init__()
        self.torch = torch_module
        self.config = config
        torch = torch_module
        coord_dim = 3 * (1 + 2 * int(config.coordinate_frequencies))
        in_dim = 3 * int(config.plane_channels) + coord_dim + int(config.geometry_feature_dim)
        layers = []
        last = in_dim
        for _ in range(int(config.hidden_layers)):
            layers.append(torch.nn.Linear(last, int(config.hidden_width)))
            last = int(config.hidden_width)
        self.layers = torch.nn.ModuleList(layers)
        if bool(config.separate_output_heads):
            head_width = int(config.head_hidden_width) if int(config.head_hidden_width) > 0 else int(config.hidden_width)
            self.psi_head_layers, self.psi_output = self._make_head(last, head_width, int(config.head_hidden_layers))
            self.pressure_head_layers, self.pressure_output = self._make_head(last, head_width, int(config.head_hidden_layers))
            self.output = None
        else:
            self.psi_head_layers = torch.nn.ModuleList()
            self.pressure_head_layers = torch.nn.ModuleList()
            self.psi_output = None
            self.pressure_output = None
            self.output = torch.nn.Linear(last, 2)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def _make_head(self, in_dim: int, hidden_width: int, hidden_layers: int):
        layers = []
        last = int(in_dim)
        for _ in range(max(0, int(hidden_layers))):
            layers.append(self.torch.nn.Linear(last, int(hidden_width)))
            last = int(hidden_width)
        return self.torch.nn.ModuleList(layers), self.torch.nn.Linear(last, 1)

    def _run_head(self, h, layers, output):
        for layer in layers:
            h = self.activation(layer(h))
        return output(h).squeeze(-1)

    def activation(self, x):
        if self.config.activation == "silu":
            return self.torch.nn.functional.silu(x)
        if self.config.activation == "tanh":
            return self.torch.tanh(x)
        raise ValueError(f"unsupported decoder activation: {self.config.activation}")

    def forward(self, planes: TriplaneBatch | tuple[Any, Any, Any], coords, geometry_features):
        p_xy, p_xt, p_yt = _plane_tuple(planes)
        coords = _ensure_batched_coords(self.torch, coords)
        if self.config.field_stack_decode:
            _u, _v, pressure, psi = self.field_stack_velocity_pressure((p_xy, p_xt, p_yt), coords)
            return psi, pressure
        geometry_features = _ensure_batched_features(self.torch, geometry_features, coords)
        features = self.sample_features((p_xy, p_xt, p_yt), coords)
        coord_encoding = self.coordinate_encoding(coords)
        h = self.torch.cat([features, coord_encoding, geometry_features.to(dtype=features.dtype)], dim=-1)
        for layer in self.layers:
            h = self.activation(layer(h))
        if bool(self.config.separate_output_heads):
            psi_unscaled = self._run_head(h, self.psi_head_layers, self.psi_output)
            pressure_unscaled = self._run_head(h, self.pressure_head_layers, self.pressure_output)
        else:
            raw = self.output(h)
            psi_unscaled = raw[..., 0]
            pressure_unscaled = raw[..., 1]
        psi_raw = float(self.config.streamfunction_output_scale) * psi_unscaled
        pressure_raw = float(self.config.pressure_output_scale) * pressure_unscaled
        if self.config.continuous_mms_abi == "continuous_triplane_mms_v1":
            pressure = self.smooth_obstacle_mask(geometry_features) * pressure_raw
            psi = self.smooth_geometry_mask(coords, geometry_features) * psi_raw
        else:
            pressure = pressure_raw
            sdf = geometry_features[..., 0]
            psi = self.hard_geometry_mask(coords, sdf) * psi_raw
        return psi, pressure

    def field_stack_velocity_pressure(self, planes: TriplaneBatch | tuple[Any, Any, Any], coords):
        p_xy, _p_xt, _p_yt = _plane_tuple(planes)
        p_xy = _ensure_batched_plane(self.torch, p_xy)
        coords = _ensure_batched_coords(self.torch, coords)
        p_xy = _expand_batch(p_xy, coords.shape[0])
        nt = int(self.config.field_stack_nt)
        if int(p_xy.shape[1]) < 3 * nt:
            raise ValueError(f"field_stack decoder needs at least {3 * nt} channels, got {int(p_xy.shape[1])}")
        sampled = _sample_cell_centered_plane(self.torch, p_xy[:, : 3 * nt], first=coords[..., 0], second=coords[..., 1])
        tau = (coords[..., 2].clamp(0.0, 1.0) * float(max(nt - 1, 1))).to(dtype=sampled.dtype)
        lower = self.torch.floor(tau).to(dtype=self.torch.long).clamp(min=0, max=max(nt - 1, 0))
        upper = (lower + 1).clamp(max=max(nt - 1, 0))
        weight = (tau - lower.to(dtype=tau.dtype)).clamp(0.0, 1.0)

        def interp(stack):
            low = self.torch.gather(stack, 2, lower[..., None]).squeeze(-1)
            high = self.torch.gather(stack, 2, upper[..., None]).squeeze(-1)
            return low * (1.0 - weight) + high * weight

        u = interp(sampled[..., 0:nt])
        v = interp(sampled[..., nt : 2 * nt])
        pressure = interp(sampled[..., 2 * nt : 3 * nt])
        psi = self.torch.zeros_like(u)
        return u, v, pressure, psi


    def sample_features(self, planes: tuple[Any, Any, Any], coords):
        p_xy, p_xt, p_yt = [_ensure_batched_plane(self.torch, p) for p in planes]
        coords = _ensure_batched_coords(self.torch, coords)
        b, n, _ = coords.shape
        p_xy, p_xt, p_yt = [_expand_batch(p, b) for p in (p_xy, p_xt, p_yt)]
        x = coords[..., 0].clamp(0.0, 1.0)
        y = coords[..., 1].clamp(0.0, 1.0)
        t = coords[..., 2].clamp(0.0, 1.0)
        f_xy = _sample_plane(self.torch, p_xy, first=x, second=y, mode=self.config.plane_interpolation)
        f_xt = _sample_plane(self.torch, p_xt, first=x, second=t, mode=self.config.plane_interpolation)
        f_yt = _sample_plane(self.torch, p_yt, first=y, second=t, mode=self.config.plane_interpolation)
        return self.torch.cat([f_xy, f_xt, f_yt], dim=-1).reshape(b, n, -1)

    def coordinate_encoding(self, coords):
        torch = self.torch
        pieces = [coords]
        for k in range(int(self.config.coordinate_frequencies)):
            scale = float(2**k) * np.pi
            pieces.append(torch.sin(scale * coords))
            pieces.append(torch.cos(scale * coords))
        return torch.cat(pieces, dim=-1)

    def hard_geometry_mask(self, coords, signed_distance):
        x = coords[..., 0]
        y = coords[..., 1]
        wall = (16.0 * x * (1.0 - x) * y * (1.0 - y)).clamp(min=0.0, max=1.0)
        obstacle = (signed_distance / max(float(self.config.obstacle_mask_epsilon), 1.0e-12)).clamp(min=0.0, max=1.0)
        return (wall * obstacle).clamp(min=0.0) ** float(self.config.hard_mask_alpha)

    def smooth_geometry_mask(self, coords, geometry_features):
        if int(geometry_features.shape[-1]) < 2:
            raise ValueError("continuous MMS requires canonical obstacle and no-slip wall distances")
        obstacle = self.smooth_obstacle_mask(geometry_features)
        transition = max(float(self.config.geometry_mask_transition), 1.0e-12)
        wall_distance = _c3_unit_ramp(geometry_features[..., 1] / transition)
        x = coords[..., 0]
        y = coords[..., 1]
        rectangular_boundary = (
            _c3_unit_ramp(x / transition)
            * _c3_unit_ramp((1.0 - x) / transition)
            * _c3_unit_ramp(y / transition)
            * _c3_unit_ramp((1.0 - y) / transition)
        )
        return obstacle * wall_distance * rectangular_boundary

    def smooth_obstacle_mask(self, geometry_features):
        transition = max(float(self.config.geometry_mask_transition), 1.0e-12)
        return _c3_unit_ramp(geometry_features[..., 0] / transition)

    def velocity_pressure(self, planes: TriplaneBatch | tuple[Any, Any, Any], coords, geometry_features, *, detach_coords: bool = True):
        coords = _ensure_batched_coords(self.torch, coords)
        if self.config.field_stack_decode:
            return self.field_stack_velocity_pressure(planes, coords)
        if detach_coords:
            coords = coords.clone().detach().requires_grad_(True)
        elif not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)
        psi, pressure = self.forward(planes, coords, geometry_features)
        grad_psi = self.torch.autograd.grad(psi.sum(), coords, create_graph=True, retain_graph=True)[0]
        u = grad_psi[..., 1]
        v = -grad_psi[..., 0]
        return u, v, pressure, psi


def make_query_grid(*, nt: int, nx: int, ny: int, device: str | None = None, dtype: str = "float32"):
    torch = _require_torch()
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    x = (torch.arange(nx, dtype=torch_dtype, device=device) + 0.5) / float(nx)
    y = (torch.arange(ny, dtype=torch_dtype, device=device) + 0.5) / float(ny)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch_dtype, device=device)
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="ij")
    return torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)


def geometry_features_from_arrays(
    coords,
    *,
    signed_distance=None,
    fluid_mask=None,
    obstacle_mask=None,
    outer_boundary_mask=None,
    canonical_geometry=None,
    interpolation: str = "bilinear",
):
    torch = _require_torch()
    coords = _ensure_batched_coords(torch, coords)
    device = coords.device
    dtype = coords.dtype
    if canonical_geometry is not None:
        canonical = torch.as_tensor(canonical_geometry, dtype=dtype, device=device)
        if canonical.ndim != 3 or int(canonical.shape[0]) != 2:
            raise ValueError(f"canonical_geometry must be [2,X,Y], got {tuple(canonical.shape)}")
        arrays = [canonical[0], canonical[1]]
    else:
        if any(value is None for value in (signed_distance, fluid_mask, obstacle_mask, outer_boundary_mask)):
            raise ValueError("legacy geometry features require signed_distance and all three masks")
        arrays = [
            torch.as_tensor(signed_distance, dtype=dtype, device=device),
            torch.as_tensor(fluid_mask, dtype=dtype, device=device),
            torch.as_tensor(obstacle_mask, dtype=dtype, device=device),
            torch.as_tensor(outer_boundary_mask, dtype=dtype, device=device),
        ]
    fields = torch.stack(arrays, dim=0)[None, ...]
    b = coords.shape[0]
    fields = _expand_batch(fields, b)
    x = coords[..., 0].clamp(0.0, 1.0)
    y = coords[..., 1].clamp(0.0, 1.0)
    return _sample_plane(torch, fields, first=x, second=y, mode=interpolation)


def _sample_plane(torch, plane, *, first, second, mode: str):
    if mode == "bilinear":
        return _grid_sample_plane(torch, plane, first=first, second=second)
    if mode == "c3_smoothstep":
        return _c3_sample_plane(torch, plane, first=first, second=second)
    if mode == "quintic_bspline":
        return _quintic_bspline_sample_plane(torch, plane, first=first, second=second)
    raise ValueError(f"unsupported triplane interpolation mode: {mode}")


def _c3_unit_ramp(value):
    s = value.clamp(min=0.0, max=1.0)
    # Seventh-order smoothstep. Extending it by constants outside [0,1]
    # keeps the value and its first three derivatives continuous.
    return s**4 * (35.0 - 84.0 * s + 70.0 * s**2 - 20.0 * s**3)


def _c3_sample_plane(torch, plane, *, first, second):
    plane = _ensure_batched_plane(torch, plane)
    b, c, height, width = plane.shape
    first_pos = (first * float(height) - 0.5).clamp(0.0, float(max(height - 1, 0)))
    second_pos = (second * float(width) - 0.5).clamp(0.0, float(max(width - 1, 0)))
    first0 = torch.floor(first_pos).to(dtype=torch.long).clamp(min=0, max=max(height - 1, 0))
    second0 = torch.floor(second_pos).to(dtype=torch.long).clamp(min=0, max=max(width - 1, 0))
    first1 = (first0 + 1).clamp(max=max(height - 1, 0))
    second1 = (second0 + 1).clamp(max=max(width - 1, 0))
    wf = _c3_unit_ramp(first_pos - first0.to(dtype=first_pos.dtype))[..., None]
    ws = _c3_unit_ramp(second_pos - second0.to(dtype=second_pos.dtype))[..., None]
    flat = plane.reshape(b, c, height * width)

    def take(ix, iy):
        linear = (ix * width + iy).clamp(min=0, max=max(height * width - 1, 0))
        gathered = torch.gather(flat, 2, linear[:, None, :].expand(b, c, linear.shape[1]))
        return gathered.transpose(1, 2)

    v00 = take(first0, second0)
    v10 = take(first1, second0)
    v01 = take(first0, second1)
    v11 = take(first1, second1)
    return v00 * (1.0 - wf) * (1.0 - ws) + v10 * wf * (1.0 - ws) + v01 * (1.0 - wf) * ws + v11 * wf * ws


def _quintic_bspline_weight(torch, distance):
    work = distance.to(dtype=torch.float64)
    result = torch.zeros_like(work)
    coefficients = (1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0)
    for k, coefficient in enumerate(coefficients):
        result = result + coefficient * torch.relu(work + 3.0 - float(k)) ** 5
    return (result / 120.0).to(dtype=distance.dtype)


def _quintic_bspline_sample_plane(torch, plane, *, first, second):
    plane = _ensure_batched_plane(torch, plane)
    b, c, height, width = plane.shape
    first_pos = first * float(height) - 0.5
    second_pos = second * float(width) - 0.5
    first_base = torch.floor(first_pos).to(dtype=torch.long)
    second_base = torch.floor(second_pos).to(dtype=torch.long)
    first_terms = []
    second_terms = []
    for offset in range(-2, 4):
        first_unclamped = first_base + offset
        second_unclamped = second_base + offset
        first_terms.append(
            (
                first_unclamped.clamp(min=0, max=max(height - 1, 0)),
                _quintic_bspline_weight(torch, first_pos - first_unclamped.to(dtype=first_pos.dtype)),
            )
        )
        second_terms.append(
            (
                second_unclamped.clamp(min=0, max=max(width - 1, 0)),
                _quintic_bspline_weight(torch, second_pos - second_unclamped.to(dtype=second_pos.dtype)),
            )
        )
    flat = plane.reshape(b, c, height * width)
    out = plane.new_zeros((b, first.shape[1], c))
    for first_index, first_weight in first_terms:
        for second_index, second_weight in second_terms:
            linear = (first_index * width + second_index).clamp(min=0, max=max(height * width - 1, 0))
            gathered = torch.gather(flat, 2, linear[:, None, :].expand(b, c, linear.shape[1])).transpose(1, 2)
            out = out + gathered * (first_weight * second_weight)[..., None]
    return out


def _sample_cell_centered_plane(torch, plane, *, first, second):
    b, c, height, width = plane.shape
    first_idx = (first.clamp(0.0, 1.0) * float(height) - 0.5).clamp(0.0, float(max(height - 1, 0)))
    second_idx = (second.clamp(0.0, 1.0) * float(width) - 0.5).clamp(0.0, float(max(width - 1, 0)))
    first0 = torch.floor(first_idx).to(dtype=torch.long)
    second0 = torch.floor(second_idx).to(dtype=torch.long)
    first1 = (first0 + 1).clamp(max=max(height - 1, 0))
    second1 = (second0 + 1).clamp(max=max(width - 1, 0))
    wf = (first_idx - first0.to(dtype=first_idx.dtype))[..., None]
    ws = (second_idx - second0.to(dtype=second_idx.dtype))[..., None]
    flat = plane.reshape(b, c, height * width)

    def take(ix, iy):
        linear = (ix * width + iy).clamp(min=0, max=max(height * width - 1, 0))
        gathered = torch.gather(flat, 2, linear[:, None, :].expand(b, c, linear.shape[1]))
        return gathered.transpose(1, 2)

    v00 = take(first0, second0)
    v10 = take(first1, second0)
    v01 = take(first0, second1)
    v11 = take(first1, second1)
    return v00 * (1.0 - wf) * (1.0 - ws) + v10 * wf * (1.0 - ws) + v01 * (1.0 - wf) * ws + v11 * wf * ws


def _grid_sample_plane(torch, plane, *, first, second):
    import torch.nn.functional as functional

    # Plane tensors are [B,C,first_axis,second_axis]. grid_sample expects
    # coordinates ordered as [width_axis, height_axis], hence [second, first].
    grid = torch.stack([2.0 * second - 1.0, 2.0 * first - 1.0], dim=-1)
    sampled = functional.grid_sample(
        plane,
        grid.reshape(grid.shape[0], grid.shape[1], 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[..., 0].transpose(1, 2)


def _plane_tuple(planes):
    if isinstance(planes, TriplaneBatch):
        return planes.p_xy, planes.p_xt, planes.p_yt
    return planes


def _ensure_batched_plane(torch, plane):
    tensor = plane if torch.is_tensor(plane) else torch.as_tensor(plane)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim != 4:
        raise ValueError(f"triplane tensor must have shape [B,C,H,W] or [C,H,W], got {tuple(tensor.shape)}")
    return tensor


def _ensure_batched_coords(torch, coords):
    tensor = coords if torch.is_tensor(coords) else torch.as_tensor(coords)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B,N,3] or [N,3], got {tuple(tensor.shape)}")
    return tensor


def _ensure_batched_features(torch, features, coords):
    tensor = features if torch.is_tensor(features) else torch.as_tensor(features)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    if tensor.ndim != 3:
        raise ValueError(f"geometry_features must have shape [B,N,G] or [N,G], got {tuple(tensor.shape)}")
    return _expand_batch(tensor, coords.shape[0])


def _expand_batch(tensor, batch_size: int):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError(f"batch size {tensor.shape[0]} cannot expand to {batch_size}")
