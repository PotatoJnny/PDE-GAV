from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from agent_verifier.vpfp.triplane import TriplaneBatch


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


def _require_torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for direct VPFP triplane diffusion") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE


@dataclass(frozen=True)
class DirectTriplaneDiffusionConfig:
    plane_channels: int = 32
    base_width: int = 96
    time_embedding_dim: int = 128
    condition_dim: int = 16
    timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    dtype: str = "float32"
    active_planes: tuple[str, ...] = ("p_xy", "p_xt", "p_yt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DirectTriplaneDiffusion:
    @staticmethod
    def build(config: DirectTriplaneDiffusionConfig | dict[str, Any] | None = None):
        torch = _require_torch()
        cfg = config if isinstance(config, DirectTriplaneDiffusionConfig) else DirectTriplaneDiffusionConfig(**dict(config or {}))
        return _DirectTriplaneDenoiser(torch, cfg)


class _DirectTriplaneDenoiser((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch_module, config: DirectTriplaneDiffusionConfig):
        super().__init__()
        self.torch = torch_module
        self.config = config
        torch = torch_module
        emb_dim = int(config.time_embedding_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.condition_mlp = torch.nn.Sequential(
            torch.nn.Linear(int(config.condition_dim), emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.active_planes = _active_planes(config.active_planes)
        self.xy_branch = _PlaneDenoiseBranch(torch, config.plane_channels, config.base_width, emb_dim) if "p_xy" in self.active_planes else None
        self.xt_branch = _PlaneDenoiseBranch(torch, config.plane_channels, config.base_width, emb_dim) if "p_xt" in self.active_planes else None
        self.yt_branch = _PlaneDenoiseBranch(torch, config.plane_channels, config.base_width, emb_dim) if "p_yt" in self.active_planes else None
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy: TriplaneBatch | tuple[Any, Any, Any], timesteps, condition=None):
        p_xy, p_xt, p_yt = _plane_tuple(noisy)
        torch = self.torch
        t = timesteps if torch.is_tensor(timesteps) else torch.as_tensor(timesteps, device=p_xy.device)
        if t.ndim == 0:
            t = t[None].expand(p_xy.shape[0])
        emb = timestep_embedding(torch, t.to(device=p_xy.device), int(self.config.time_embedding_dim)).to(dtype=p_xy.dtype)
        emb = self.time_mlp(emb)
        if condition is None:
            cond = torch.zeros((p_xy.shape[0], int(self.config.condition_dim)), dtype=p_xy.dtype, device=p_xy.device)
        else:
            cond = condition if torch.is_tensor(condition) else torch.as_tensor(condition, dtype=p_xy.dtype, device=p_xy.device)
            if cond.ndim == 1:
                cond = cond[None, :].expand(p_xy.shape[0], -1)
        emb = emb + self.condition_mlp(cond.to(dtype=p_xy.dtype, device=p_xy.device))
        return TriplaneBatch(
            p_xy=self.xy_branch(p_xy, emb) if self.xy_branch is not None else torch.zeros_like(p_xy),
            p_xt=self.xt_branch(p_xt, emb) if self.xt_branch is not None else torch.zeros_like(p_xt),
            p_yt=self.yt_branch(p_yt, emb) if self.yt_branch is not None else torch.zeros_like(p_yt),
        )


class _PlaneDenoiseBranch((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch, channels: int, width: int, embedding_dim: int):
        super().__init__()
        self.in_conv = torch.nn.Conv2d(channels, width, kernel_size=3, padding=1)
        self.block1 = _ResidualBlock(torch, width, embedding_dim)
        self.block2 = _ResidualBlock(torch, width, embedding_dim)
        self.out_conv = torch.nn.Conv2d(width, channels, kernel_size=3, padding=1)

    def forward(self, x, emb):
        h = self.in_conv(x)
        h = self.block1(h, emb)
        h = self.block2(h, emb)
        return self.out_conv(h)


class _ResidualBlock((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch, width: int, embedding_dim: int):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=8, num_channels=width)
        self.conv1 = torch.nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.emb = torch.nn.Linear(embedding_dim, width)
        self.norm2 = torch.nn.GroupNorm(num_groups=8, num_channels=width)
        self.conv2 = torch.nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.act = torch.nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


@dataclass
class DiffusionSchedule:
    betas: Any
    alphas: Any
    alpha_bars: Any


def make_schedule(config: DirectTriplaneDiffusionConfig, *, device=None, dtype=None) -> DiffusionSchedule:
    torch = _require_torch()
    betas = torch.linspace(float(config.beta_start), float(config.beta_end), int(config.timesteps), device=device, dtype=dtype or torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def diffusion_training_loss(model, clean: TriplaneBatch | tuple[Any, Any, Any], *, condition=None, schedule: DiffusionSchedule | None = None):
    torch = _require_torch()
    p_xy, p_xt, p_yt = _plane_tuple(clean)
    cfg = model.config
    active = _active_planes(getattr(model, "active_planes", cfg.active_planes))
    schedule = schedule or make_schedule(cfg, device=p_xy.device, dtype=p_xy.dtype)
    batch = p_xy.shape[0]
    t = torch.randint(0, int(cfg.timesteps), (batch,), dtype=torch.long, device=p_xy.device)
    noisy, eps = q_sample(TriplaneBatch(p_xy, p_xt, p_yt), t, schedule, active_planes=active)
    pred = model(noisy, t, condition=condition)
    losses = []
    if "p_xy" in active:
        losses.append(torch.mean((pred.p_xy - eps.p_xy) ** 2))
    if "p_xt" in active:
        losses.append(torch.mean((pred.p_xt - eps.p_xt) ** 2))
    if "p_yt" in active:
        losses.append(torch.mean((pred.p_yt - eps.p_yt) ** 2))
    if not losses:
        raise ValueError("direct triplane diffusion needs at least one active plane")
    return sum(losses) / float(len(losses))


def q_sample(clean: TriplaneBatch, timesteps, schedule: DiffusionSchedule, *, active_planes: tuple[str, ...] | None = None) -> tuple[TriplaneBatch, TriplaneBatch]:
    torch = _require_torch()
    active = _active_planes(active_planes or ("p_xy", "p_xt", "p_yt"))
    eps = TriplaneBatch(
        torch.randn_like(clean.p_xy) if "p_xy" in active else torch.zeros_like(clean.p_xy),
        torch.randn_like(clean.p_xt) if "p_xt" in active else torch.zeros_like(clean.p_xt),
        torch.randn_like(clean.p_yt) if "p_yt" in active else torch.zeros_like(clean.p_yt),
    )
    a_bar = schedule.alpha_bars[timesteps].to(dtype=clean.p_xy.dtype)

    def add_noise(x, noise, name):
        if name not in active:
            return x
        scale_clean = torch.sqrt(a_bar).reshape(-1, 1, 1, 1)
        scale_noise = torch.sqrt(1.0 - a_bar).reshape(-1, 1, 1, 1)
        return scale_clean * x + scale_noise * noise

    return (
        TriplaneBatch(add_noise(clean.p_xy, eps.p_xy, "p_xy"), add_noise(clean.p_xt, eps.p_xt, "p_xt"), add_noise(clean.p_yt, eps.p_yt, "p_yt")),
        eps,
    )


def sample_triplanes(
    model,
    *,
    batch_size: int,
    plane_shapes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    condition=None,
    steps: int | None = None,
    method: str = "ddim",
    eta: float = 0.0,
    start_timestep: int | None = None,
    x0_clip: float | None = None,
):
    torch = _require_torch()
    cfg = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    schedule = make_schedule(cfg, device=device, dtype=dtype)
    method = str(method).lower().replace("-", "_")
    if method not in {"ddim", "ddpm", "legacy_ddpm"}:
        raise ValueError(f"unsupported triplane diffusion sampler: {method}")
    sample_steps = max(1, min(int(steps or cfg.timesteps), int(cfg.timesteps)))
    if method == "ddpm" and sample_steps < int(cfg.timesteps):
        # Ancestral DDPM is only exact on adjacent timesteps. Use DDIM for skipped schedules.
        method = "ddim"
    indices = _sample_indices(torch, int(cfg.timesteps), sample_steps, start_timestep=start_timestep)
    shapes = [(batch_size, *shape) for shape in plane_shapes]
    active = _active_planes(cfg.active_planes)
    current = TriplaneBatch(
        torch.randn(shapes[0], device=device, dtype=dtype) if "p_xy" in active else torch.zeros(shapes[0], device=device, dtype=dtype),
        torch.randn(shapes[1], device=device, dtype=dtype) if "p_xt" in active else torch.zeros(shapes[1], device=device, dtype=dtype),
        torch.randn(shapes[2], device=device, dtype=dtype) if "p_yt" in active else torch.zeros(shapes[2], device=device, dtype=dtype),
    )
    for pos, idx in enumerate(indices):
        t = torch.full((batch_size,), idx, dtype=torch.long, device=device)
        eps = model(current, t, condition=condition)
        prev_idx = indices[pos + 1] if pos + 1 < len(indices) else -1
        if method == "ddim":
            current = _ddim_step(torch, current, eps, idx, prev_idx, schedule, active_planes=active, eta=float(eta), x0_clip=x0_clip)
        else:
            current = _ddpm_step(torch, current, eps, t, schedule, add_noise=idx > 0, active_planes=active)
    return current


def _sample_indices(torch, timesteps: int, sample_steps: int, *, start_timestep: int | None = None) -> list[int]:
    start = timesteps - 1 if start_timestep is None else max(0, min(int(start_timestep), timesteps - 1))
    if sample_steps >= start + 1:
        return list(range(start, -1, -1))
    values = torch.linspace(start, 0, steps=sample_steps).round().to(dtype=torch.long).cpu().tolist()
    indices: list[int] = []
    seen = set()
    for value in values:
        idx = int(value)
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if indices[-1] != 0:
        indices.append(0)
    return sorted(indices, reverse=True)


def timestep_embedding(torch, timesteps, dim: int):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def _ddpm_step(torch, current: TriplaneBatch, eps: TriplaneBatch, t, schedule: DiffusionSchedule, *, add_noise: bool, active_planes: tuple[str, ...]):
    beta = schedule.betas[t].reshape(-1, 1, 1, 1)
    alpha = schedule.alphas[t].reshape(-1, 1, 1, 1)
    alpha_bar = schedule.alpha_bars[t].reshape(-1, 1, 1, 1)

    active = _active_planes(active_planes)

    def step(x, e, name):
        if name not in active:
            return x
        mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * e) / torch.sqrt(alpha)
        if add_noise:
            return mean + torch.sqrt(beta) * torch.randn_like(x)
        return mean

    return TriplaneBatch(step(current.p_xy, eps.p_xy, "p_xy"), step(current.p_xt, eps.p_xt, "p_xt"), step(current.p_yt, eps.p_yt, "p_yt"))



def _ddim_step(
    torch,
    current: TriplaneBatch,
    eps: TriplaneBatch,
    idx: int,
    prev_idx: int,
    schedule: DiffusionSchedule,
    *,
    active_planes: tuple[str, ...],
    eta: float,
    x0_clip: float | None,
):
    active = _active_planes(active_planes)
    device = current.p_xy.device
    dtype = current.p_xy.dtype
    alpha_bar_t = schedule.alpha_bars[int(idx)].to(device=device, dtype=dtype).reshape(1, 1, 1, 1)
    if int(prev_idx) >= 0:
        alpha_bar_prev = schedule.alpha_bars[int(prev_idx)].to(device=device, dtype=dtype).reshape(1, 1, 1, 1)
    else:
        alpha_bar_prev = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    eta = max(0.0, float(eta))
    if eta > 0.0 and int(prev_idx) >= 0:
        sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=1.0e-12)) * torch.sqrt(
            torch.clamp(1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1.0e-12), min=0.0)
        )
    else:
        sigma = torch.zeros_like(alpha_bar_t)
    direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0))

    def step(x, e, name):
        if name not in active:
            return x
        pred_x0 = (x - torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0)) * e) / torch.sqrt(torch.clamp(alpha_bar_t, min=1.0e-12))
        if x0_clip is not None and float(x0_clip) > 0.0:
            pred_x0 = torch.clamp(pred_x0, -float(x0_clip), float(x0_clip))
        out = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_scale * e
        if eta > 0.0 and int(prev_idx) >= 0:
            out = out + sigma * torch.randn_like(x)
        return out

    return TriplaneBatch(
        step(current.p_xy, eps.p_xy, "p_xy"),
        step(current.p_xt, eps.p_xt, "p_xt"),
        step(current.p_yt, eps.p_yt, "p_yt"),
    )


def apply_triplane_normalization(planes, normalization: dict[str, Any] | None, *, inverse: bool = False) -> TriplaneBatch:
    if not normalization or not normalization.get("enabled", False):
        p_xy, p_xt, p_yt = _plane_tuple(planes)
        return TriplaneBatch(p_xy, p_xt, p_yt)
    torch = _require_torch()
    p_xy, p_xt, p_yt = _plane_tuple(planes)
    return TriplaneBatch(
        _transform_plane(torch, p_xy, normalization, "p_xy", inverse=inverse),
        _transform_plane(torch, p_xt, normalization, "p_xt", inverse=inverse),
        _transform_plane(torch, p_yt, normalization, "p_yt", inverse=inverse),
    )


def _transform_plane(torch, tensor, normalization: dict[str, Any], name: str, *, inverse: bool):
    plane_stats = dict(normalization.get("planes", {}).get(name, {}))
    if not plane_stats:
        return tensor
    mean = torch.as_tensor(plane_stats.get("mean"), dtype=tensor.dtype, device=tensor.device).reshape(1, -1, 1, 1)
    std = torch.as_tensor(plane_stats.get("std"), dtype=tensor.dtype, device=tensor.device).reshape(1, -1, 1, 1)
    if mean.shape[1] != tensor.shape[1] or std.shape[1] != tensor.shape[1]:
        raise ValueError(f"normalization stats for {name} have {mean.shape[1]} channels, tensor has {tensor.shape[1]}")
    if inverse:
        return tensor * std + mean
    return (tensor - mean) / std

def _active_planes(value) -> tuple[str, ...]:
    allowed = ("p_xy", "p_xt", "p_yt")
    active = tuple(str(item) for item in value)
    bad = [item for item in active if item not in allowed]
    if bad:
        raise ValueError(f"unsupported active triplane planes: {bad}")
    if not active:
        raise ValueError("at least one active triplane plane is required")
    return active

def _plane_tuple(planes):
    if isinstance(planes, TriplaneBatch):
        return planes.p_xy, planes.p_xt, planes.p_yt
    return planes
