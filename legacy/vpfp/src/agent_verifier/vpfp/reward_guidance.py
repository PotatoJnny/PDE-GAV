from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agent_verifier.neural_generator.triplane_decoder import TriplaneBatch
from agent_verifier.neural_generator.triplane_diffusion import make_schedule, timestep_embedding


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for VPFP reward guidance") from exc
    return torch


@dataclass(frozen=True)
class VPFPRewardGuidanceConfig:
    plane_channels: int = 32
    width: int = 64
    time_embedding_dim: int = 128
    condition_dim: int = 16
    dropout: float = 0.0
    active_planes: tuple[str, ...] = ("p_xy", "p_xt", "p_yt")
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VPFPRewardGuidance:
    @staticmethod
    def build(config: VPFPRewardGuidanceConfig | dict[str, Any] | None = None):
        cfg = config if isinstance(config, VPFPRewardGuidanceConfig) else VPFPRewardGuidanceConfig(**dict(config or {}))
        return _VPFPRewardGuidanceModel(_require_torch(), cfg)


class _VPFPRewardGuidanceModel(_require_torch().nn.Module):
    def __init__(self, torch_module, config: VPFPRewardGuidanceConfig):
        super().__init__()
        torch = torch_module
        self.torch = torch
        self.config = config
        width = int(config.width)
        emb_dim = int(config.time_embedding_dim)
        self.active_planes = _active_planes(config.active_planes)
        self.encoders = torch.nn.ModuleDict(
            {
                name: _PlaneRewardEncoder(torch, int(config.plane_channels), width)
                for name in self.active_planes
            }
        )
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
        fused_dim = 3 * width + 2 * emb_dim
        hidden = max(2 * width, emb_dim)
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.absolute_head = torch.nn.Linear(hidden, 1)
        self.centered_head = torch.nn.Linear(hidden, 1)
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        dtype = torch.float64 if str(config.dtype) == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, noisy: TriplaneBatch | tuple[Any, Any, Any], timesteps, condition=None) -> dict[str, Any]:
        torch = self.torch
        p_xy, p_xt, p_yt = _plane_tuple(noisy)
        batch = int(p_xy.shape[0])
        t = timesteps if torch.is_tensor(timesteps) else torch.as_tensor(timesteps, device=p_xy.device)
        if t.ndim == 0:
            t = t[None].expand(batch)
        cond = self._condition_tensor(condition, batch=batch, device=p_xy.device, dtype=p_xy.dtype)
        time_emb = timestep_embedding(torch, t.to(device=p_xy.device), int(self.config.time_embedding_dim)).to(dtype=p_xy.dtype)
        time_emb = self.time_mlp(time_emb)
        cond_emb = self.condition_mlp(cond)
        plane_features = []
        for name, plane in (("p_xy", p_xy), ("p_xt", p_xt), ("p_yt", p_yt)):
            if name in self.active_planes:
                plane_features.append(self.encoders[name](plane))
            else:
                plane_features.append(torch.zeros((batch, int(self.config.width)), device=plane.device, dtype=plane.dtype))
        features = self.interaction(torch.cat([*plane_features, time_emb, cond_emb], dim=1))
        absolute = self.absolute_head(features).reshape(-1)
        centered = self.centered_head(features).reshape(-1)
        condition_value = self.condition_head(cond_emb).reshape(-1)
        return {
            "absolute": absolute,
            "condition_centered": centered,
            "condition_value": condition_value,
            "decomposed_absolute": condition_value + centered,
        }

    def condition_value(self, condition) -> Any:
        torch = self.torch
        param = next(self.parameters())
        cond = condition if torch.is_tensor(condition) else torch.as_tensor(condition, dtype=param.dtype, device=param.device)
        if cond.ndim == 1:
            cond = cond[None, :]
        cond = self._condition_tensor(cond, batch=int(cond.shape[0]), device=param.device, dtype=param.dtype)
        return self.condition_head(self.condition_mlp(cond)).reshape(-1)

    def _condition_tensor(self, condition, *, batch: int, device, dtype):
        torch = self.torch
        if condition is None:
            return torch.zeros((batch, int(self.config.condition_dim)), dtype=dtype, device=device)
        cond = condition if torch.is_tensor(condition) else torch.as_tensor(condition, dtype=dtype, device=device)
        if cond.ndim == 1:
            cond = cond[None, :].expand(batch, -1)
        if tuple(cond.shape) != (batch, int(self.config.condition_dim)):
            raise ValueError(
                f"reward-guidance condition shape {tuple(cond.shape)} does not match "
                f"({batch}, {int(self.config.condition_dim)})"
            )
        return cond.to(device=device, dtype=dtype)


class _PlaneRewardEncoder(_require_torch().nn.Module):
    def __init__(self, torch, channels: int, width: int):
        super().__init__()
        groups = max(1, min(8, int(width)))
        while int(width) % groups:
            groups -= 1
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(int(channels), int(width), kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, int(width)),
            torch.nn.SiLU(),
            torch.nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, int(width)),
            torch.nn.SiLU(),
            torch.nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
            torch.nn.SiLU(),
        )

    def forward(self, plane):
        return self.net(plane).mean(dim=(-2, -1))


def build_reward_targets(
    rewards: Sequence[float] | np.ndarray,
    group_ids: Sequence[str],
    *,
    normalization: dict[str, float] | None = None,
    epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size != len(group_ids):
        raise ValueError("reward targets need one group id per one-dimensional reward")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("reward targets must be non-empty and finite")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        key = str(group_id)
        if not key:
            raise ValueError(f"missing condition group id at index {index}")
        groups[key].append(index)
    baselines = np.zeros_like(values)
    for indices in groups.values():
        baselines[indices] = float(np.mean(values[indices]))
    centered = values - baselines
    if normalization is None:
        mean = float(np.mean(values))
        std = max(float(np.std(values)), float(epsilon))
    else:
        mean = float(normalization["mean"])
        std = max(float(normalization["std"]), float(epsilon))
    return {
        "absolute": (values - mean) / std,
        "condition_value": (baselines - mean) / std,
        "condition_centered": centered / std,
        "raw_absolute": values,
        "raw_condition_value": baselines,
        "raw_condition_centered": centered,
        "normalization": {"mean": mean, "std": std},
        "group_count": len(groups),
        "singleton_group_count": sum(len(indices) == 1 for indices in groups.values()),
    }


def reward_guidance_loss(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    *,
    group_indices=None,
    mode: str = "combined",
    absolute_weight: float = 1.0,
    centered_weight: float = 1.0,
    condition_weight: float = 0.5,
    consistency_weight: float = 0.25,
    global_rank_weight: float = 0.1,
    within_rank_weight: float = 0.1,
):
    torch = _require_torch()
    resolved = str(mode)
    if resolved not in {"absolute", "condition_centered", "combined"}:
        raise ValueError(f"unsupported reward-guidance objective: {mode}")
    reference = outputs["absolute"]
    tensor_targets = {
        key: value.to(device=reference.device, dtype=reference.dtype)
        for key, value in targets.items()
        if hasattr(value, "to")
    }
    zero = reference.sum() * 0.0
    absolute_loss = torch.nn.functional.smooth_l1_loss(outputs["absolute"], tensor_targets["absolute"])
    centered_loss = torch.nn.functional.smooth_l1_loss(outputs["condition_centered"], tensor_targets["condition_centered"])
    condition_loss = torch.nn.functional.smooth_l1_loss(outputs["condition_value"], tensor_targets["condition_value"])
    consistency_loss = torch.nn.functional.smooth_l1_loss(outputs["absolute"], outputs["decomposed_absolute"])
    global_rank = pairwise_ranking_loss(outputs["absolute"], tensor_targets["absolute"])
    within_rank = pairwise_ranking_loss(
        outputs["condition_centered"],
        tensor_targets["condition_centered"],
        group_indices=group_indices,
    )
    loss = zero
    if resolved in {"absolute", "combined"}:
        loss = loss + float(absolute_weight) * absolute_loss + float(global_rank_weight) * global_rank
    if resolved in {"condition_centered", "combined"}:
        loss = loss + float(centered_weight) * centered_loss + float(within_rank_weight) * within_rank
    if resolved == "combined":
        loss = loss + float(condition_weight) * condition_loss + float(consistency_weight) * consistency_loss
    return loss, {
        "absolute_loss": absolute_loss.detach(),
        "condition_centered_loss": centered_loss.detach(),
        "condition_value_loss": condition_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
        "global_rank_loss": global_rank.detach(),
        "within_rank_loss": within_rank.detach(),
    }


def pairwise_ranking_loss(prediction, target, *, group_indices=None, min_target_delta: float = 1.0e-6):
    torch = _require_torch()
    prediction = prediction.reshape(-1)
    target = target.reshape(-1).to(device=prediction.device, dtype=prediction.dtype)
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    rows, cols = torch.triu_indices(prediction.numel(), prediction.numel(), offset=1, device=prediction.device)
    target_delta = target[rows] - target[cols]
    mask = torch.abs(target_delta) > float(min_target_delta)
    if group_indices is not None:
        groups = group_indices.to(device=prediction.device).reshape(-1)
        mask = mask & (groups[rows] == groups[cols])
    if not bool(torch.any(mask)):
        return prediction.sum() * 0.0
    signed_margin = torch.sign(target_delta[mask]) * (prediction[rows[mask]] - prediction[cols[mask]])
    return torch.nn.functional.softplus(-signed_margin).mean()


def reward_guidance_potential(
    outputs: dict[str, Any],
    *,
    mode: str,
    absolute_weight: float = 1.0,
    centered_weight: float = 1.0,
):
    resolved = str(mode)
    if resolved == "absolute":
        return float(absolute_weight) * outputs["absolute"]
    if resolved == "condition_centered":
        return float(centered_weight) * outputs["condition_centered"]
    if resolved == "combined":
        return float(absolute_weight) * outputs["absolute"] + float(centered_weight) * outputs["condition_centered"]
    raise ValueError(f"unsupported reward-guidance mode: {mode}")


def condition_sampling_probabilities(
    scores: Sequence[float] | np.ndarray,
    *,
    temperature: float = 1.0,
    max_density_ratio: float = 10.0,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("condition scores must be one-dimensional, non-empty, and finite")
    if float(temperature) <= 0.0:
        raise ValueError("condition sampling temperature must be positive")
    if float(max_density_ratio) < 1.0:
        raise ValueError("condition max density ratio must be at least one")
    logits = (values - float(np.max(values))) / float(temperature)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    probabilities = weights / float(np.sum(weights))
    cap = min(1.0, float(max_density_ratio) / float(values.size))
    if cap >= 1.0:
        return probabilities
    free = np.ones(values.size, dtype=bool)
    result = np.zeros(values.size, dtype=np.float64)
    while np.any(free):
        remaining = 1.0 - float(np.sum(result[~free]))
        free_weights = probabilities[free]
        candidate = remaining * free_weights / max(float(np.sum(free_weights)), 1.0e-300)
        over = candidate > cap + 1.0e-15
        if not np.any(over):
            result[free] = candidate
            break
        free_indices = np.flatnonzero(free)
        capped_indices = free_indices[over]
        result[capped_indices] = cap
        free[capped_indices] = False
    result /= float(np.sum(result))
    return result


def reward_guided_epsilon(
    reward_model,
    current: TriplaneBatch,
    timesteps,
    base_epsilon: TriplaneBatch,
    schedule,
    *,
    condition=None,
    mode: str,
    guidance_strength: float = 1.0,
    absolute_weight: float = 1.0,
    centered_weight: float = 1.0,
    gradient_normalization: str = "score_rms",
    max_score_ratio: float = 1.0,
    active_planes: Sequence[str] = ("p_xy", "p_xt", "p_yt"),
) -> tuple[TriplaneBatch, dict[str, float]]:
    torch = _require_torch()
    active = _active_planes(active_planes)
    requested = {}
    for name, value in _plane_items(current):
        requested[name] = value.detach().requires_grad_(name in active)
    guided_input = TriplaneBatch(requested["p_xy"], requested["p_xt"], requested["p_yt"])
    with torch.enable_grad():
        outputs = reward_model(guided_input, timesteps, condition=condition)
        potential = reward_guidance_potential(
            outputs,
            mode=mode,
            absolute_weight=float(absolute_weight),
            centered_weight=float(centered_weight),
        )
        active_tensors = [requested[name] for name in active]
        active_gradients = torch.autograd.grad(potential.sum(), active_tensors, allow_unused=True)
    gradients = {name: gradient for name, gradient in zip(active, active_gradients)}
    alpha_bar = schedule.alpha_bars[timesteps].to(device=current.p_xy.device, dtype=current.p_xy.dtype).reshape(-1, 1, 1, 1)
    sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1.0e-12))
    diagnostics: dict[str, float] = {
        "potential_mean": float(potential.detach().mean().cpu()),
        "absolute_mean": float(outputs["absolute"].detach().mean().cpu()),
        "condition_centered_mean": float(outputs["condition_centered"].detach().mean().cpu()),
        "condition_value_mean": float(outputs["condition_value"].detach().mean().cpu()),
    }
    guided_planes = {}
    for name, eps in _plane_items(base_epsilon):
        if name not in active:
            guided_planes[name] = eps.detach()
            continue
        gradient = gradients.get(name)
        if gradient is None:
            gradient = torch.zeros_like(eps)
        gradient = gradient.detach().to(dtype=eps.dtype, device=eps.device)
        base_score = -eps.detach() / sigma
        grad_rms = _batch_rms(torch, gradient)
        score_rms = _batch_rms(torch, base_score)
        resolved_normalization = str(gradient_normalization)
        if resolved_normalization == "score_rms":
            gradient = gradient * score_rms / torch.clamp(grad_rms, min=1.0e-12)
        elif resolved_normalization != "none":
            raise ValueError(f"unsupported reward-guidance gradient normalization: {gradient_normalization}")
        addition = float(guidance_strength) * gradient
        addition_rms = _batch_rms(torch, addition)
        if float(max_score_ratio) > 0.0:
            limit = float(max_score_ratio) * score_rms
            addition = addition * torch.minimum(torch.ones_like(limit), limit / torch.clamp(addition_rms, min=1.0e-12))
            addition_rms = _batch_rms(torch, addition)
        guided_planes[name] = (eps.detach() - sigma * addition).detach()
        diagnostics[f"{name}_raw_gradient_rms"] = float(grad_rms.mean().detach().cpu())
        diagnostics[f"{name}_score_addition_ratio"] = float(
            (addition_rms / torch.clamp(score_rms, min=1.0e-12)).mean().detach().cpu()
        )
    return TriplaneBatch(guided_planes["p_xy"], guided_planes["p_xt"], guided_planes["p_yt"]), diagnostics



def sample_reward_guided_triplanes(
    base_model,
    reward_model,
    *,
    batch_size: int,
    plane_shapes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    condition=None,
    steps: int | None = None,
    method: str = "ddim",
    eta: float = 0.0,
    guidance_mode: str = "combined",
    guidance_strength: float = 1.0,
    absolute_weight: float = 1.0,
    centered_weight: float = 1.0,
    gradient_normalization: str = "score_rms",
    max_score_ratio: float = 1.0,
    guidance_min_timestep: int = 0,
    guidance_max_timestep: int | None = None,
    absolute_guidance_max_timestep: int | None = None,
    centered_guidance_max_timestep: int | None = None,
    initial_noise: TriplaneBatch | None = None,
) -> tuple[TriplaneBatch, dict[str, Any]]:
    # This reverse loop is independent from the protected DDPO implementation.
    torch = _require_torch()
    config = base_model.config
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    schedule = make_schedule(config, device=device, dtype=dtype)
    active = _active_planes(config.active_planes)
    resolved_method = str(method).lower()
    if resolved_method not in {"ddim", "ddpm"}:
        raise ValueError(f"unsupported reward-guidance sampler: {method}")
    indices = _sample_indices(int(config.timesteps), int(steps or config.timesteps))
    if resolved_method == "ddpm" and not _adjacent_indices(indices):
        raise ValueError("reward-guided DDPM requires adjacent reverse timesteps; use DDIM for skipped schedules")
    shapes = [(int(batch_size), *shape) for shape in plane_shapes]
    if initial_noise is None:
        current = TriplaneBatch(
            torch.randn(shapes[0], device=device, dtype=dtype) if "p_xy" in active else torch.zeros(shapes[0], device=device, dtype=dtype),
            torch.randn(shapes[1], device=device, dtype=dtype) if "p_xt" in active else torch.zeros(shapes[1], device=device, dtype=dtype),
            torch.randn(shapes[2], device=device, dtype=dtype) if "p_yt" in active else torch.zeros(shapes[2], device=device, dtype=dtype),
        )
    else:
        current = TriplaneBatch(
            initial_noise.p_xy.detach().to(device=device, dtype=dtype).clone(),
            initial_noise.p_xt.detach().to(device=device, dtype=dtype).clone(),
            initial_noise.p_yt.detach().to(device=device, dtype=dtype).clone(),
        )
    diagnostics: list[dict[str, float | int]] = []
    max_timestep = int(config.timesteps) - 1 if guidance_max_timestep is None else int(guidance_max_timestep)
    absolute_max_timestep = (
        max_timestep
        if absolute_guidance_max_timestep is None
        else min(max_timestep, int(absolute_guidance_max_timestep))
    )
    centered_max_timestep = (
        max_timestep
        if centered_guidance_max_timestep is None
        else min(max_timestep, int(centered_guidance_max_timestep))
    )
    resolved_guidance_mode = str(guidance_mode)
    if resolved_guidance_mode not in {"absolute", "condition_centered", "combined"}:
        raise ValueError(f"unsupported reward-guidance mode: {guidance_mode}")
    for position, index in enumerate(indices):
        timesteps = torch.full((int(batch_size),), int(index), dtype=torch.long, device=device)
        with torch.no_grad():
            base_epsilon = base_model(current, timesteps, condition=condition)
        inside_global_window = (
            float(guidance_strength) != 0.0
            and int(guidance_min_timestep) <= int(index) <= max_timestep
        )
        absolute_active = (
            inside_global_window
            and resolved_guidance_mode in {"absolute", "combined"}
            and float(absolute_weight) != 0.0
            and int(index) <= absolute_max_timestep
        )
        centered_active = (
            inside_global_window
            and resolved_guidance_mode in {"condition_centered", "combined"}
            and float(centered_weight) != 0.0
            and int(index) <= centered_max_timestep
        )
        use_guidance = absolute_active or centered_active
        if use_guidance:
            epsilon, step_diagnostics = reward_guided_epsilon(
                reward_model,
                current,
                timesteps,
                base_epsilon,
                schedule,
                condition=condition,
                mode=resolved_guidance_mode,
                guidance_strength=float(guidance_strength),
                absolute_weight=float(absolute_weight) if absolute_active else 0.0,
                centered_weight=float(centered_weight) if centered_active else 0.0,
                gradient_normalization=str(gradient_normalization),
                max_score_ratio=float(max_score_ratio),
                active_planes=active,
            )
            diagnostics.append(
                {
                    "timestep": int(index),
                    "absolute_head_active": int(absolute_active),
                    "centered_head_active": int(centered_active),
                    **step_diagnostics,
                }
            )
        else:
            epsilon = base_epsilon
        previous_index = indices[position + 1] if position + 1 < len(indices) else -1
        if resolved_method == "ddim":
            current = _ddim_step(
                torch,
                current,
                epsilon,
                index=int(index),
                previous_index=int(previous_index),
                schedule=schedule,
                active_planes=active,
                eta=float(eta),
            )
        else:
            current = _ddpm_step(
                torch,
                current,
                epsilon,
                timesteps,
                schedule,
                active_planes=active,
                add_noise=int(index) > 0,
            )
    summary = _aggregate_guidance_diagnostics(
        diagnostics,
        mode=resolved_guidance_mode,
        strength=float(guidance_strength),
    )
    summary.update(
        {
            "guidance_min_timestep": int(guidance_min_timestep),
            "guidance_max_timestep": int(max_timestep),
            "absolute_guidance_max_timestep": int(absolute_max_timestep),
            "centered_guidance_max_timestep": int(centered_max_timestep),
            "absolute_guided_step_count": sum(int(row.get("absolute_head_active", 0)) for row in diagnostics),
            "centered_guided_step_count": sum(int(row.get("centered_head_active", 0)) for row in diagnostics),
        }
    )
    return current, summary


def _sample_indices(timesteps: int, sample_steps: int) -> list[int]:
    torch = _require_torch()
    count = max(1, min(int(sample_steps), int(timesteps)))
    if count >= int(timesteps):
        return list(range(int(timesteps) - 1, -1, -1))
    values = torch.linspace(int(timesteps) - 1, 0, steps=count).round().to(dtype=torch.long).tolist()
    indices: list[int] = []
    for value in values:
        index = int(value)
        if not indices or index != indices[-1]:
            indices.append(index)
    if indices[-1] != 0:
        indices.append(0)
    return indices


def _adjacent_indices(indices: Sequence[int]) -> bool:
    return all(int(right) == int(left) - 1 for left, right in zip(indices[:-1], indices[1:]))


def _ddpm_step(torch, current, epsilon, timesteps, schedule, *, active_planes, add_noise: bool):
    beta = schedule.betas[timesteps].reshape(-1, 1, 1, 1)
    alpha = schedule.alphas[timesteps].reshape(-1, 1, 1, 1)
    alpha_bar = schedule.alpha_bars[timesteps].reshape(-1, 1, 1, 1)

    def step(x, eps, name):
        if name not in active_planes:
            return x.detach()
        mean = (x - beta / torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1.0e-12)) * eps) / torch.sqrt(
            torch.clamp(alpha, min=1.0e-12)
        )
        if add_noise:
            mean = mean + torch.sqrt(beta) * torch.randn_like(x)
        return mean.detach()

    return TriplaneBatch(
        step(current.p_xy, epsilon.p_xy, "p_xy"),
        step(current.p_xt, epsilon.p_xt, "p_xt"),
        step(current.p_yt, epsilon.p_yt, "p_yt"),
    )


def _ddim_step(torch, current, epsilon, *, index: int, previous_index: int, schedule, active_planes, eta: float):
    device = current.p_xy.device
    dtype = current.p_xy.dtype
    alpha_bar = schedule.alpha_bars[int(index)].to(device=device, dtype=dtype).reshape(1, 1, 1, 1)
    previous_alpha_bar = (
        schedule.alpha_bars[int(previous_index)].to(device=device, dtype=dtype).reshape(1, 1, 1, 1)
        if int(previous_index) >= 0
        else torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    )
    eta = max(0.0, float(eta))
    if eta > 0.0 and int(previous_index) >= 0:
        sigma = eta * torch.sqrt((1.0 - previous_alpha_bar) / torch.clamp(1.0 - alpha_bar, min=1.0e-12)) * torch.sqrt(
            torch.clamp(1.0 - alpha_bar / torch.clamp(previous_alpha_bar, min=1.0e-12), min=0.0)
        )
    else:
        sigma = torch.zeros_like(alpha_bar)
    direction = torch.sqrt(torch.clamp(1.0 - previous_alpha_bar - sigma * sigma, min=0.0))

    def step(x, eps, name):
        if name not in active_planes:
            return x.detach()
        predicted_clean = (x - torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0)) * eps) / torch.sqrt(
            torch.clamp(alpha_bar, min=1.0e-12)
        )
        result = torch.sqrt(previous_alpha_bar) * predicted_clean + direction * eps
        if eta > 0.0 and int(previous_index) >= 0:
            result = result + sigma * torch.randn_like(x)
        return result.detach()

    return TriplaneBatch(
        step(current.p_xy, epsilon.p_xy, "p_xy"),
        step(current.p_xt, epsilon.p_xt, "p_xt"),
        step(current.p_yt, epsilon.p_yt, "p_yt"),
    )


def _aggregate_guidance_diagnostics(rows: list[dict[str, float | int]], *, mode: str, strength: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "guidance_mode": str(mode),
        "guidance_strength": float(strength),
        "guided_step_count": len(rows),
    }
    if not rows:
        return summary
    keys = sorted({key for row in rows for key in row if key != "timestep"})
    summary["timestep_min"] = min(int(row["timestep"]) for row in rows)
    summary["timestep_max"] = max(int(row["timestep"]) for row in rows)
    summary["mean"] = {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in keys
    }
    return summary

def load_reward_guidance_checkpoint(path: str | Path, *, device="cpu"):
    torch = _require_torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = VPFPRewardGuidanceConfig(**payload["model_config"])
    model = VPFPRewardGuidance.build(config).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _batch_rms(torch, value):
    return torch.sqrt(torch.mean(value * value, dim=tuple(range(1, value.ndim)), keepdim=True) + 1.0e-20)


def _active_planes(values: Sequence[str]) -> tuple[str, ...]:
    allowed = ("p_xy", "p_xt", "p_yt")
    active = tuple(name for name in allowed if name in {str(value) for value in values})
    if not active:
        raise ValueError("reward guidance needs at least one active plane")
    return active


def _plane_tuple(value):
    if isinstance(value, TriplaneBatch):
        return value.p_xy, value.p_xt, value.p_yt
    return value


def _plane_items(value):
    p_xy, p_xt, p_yt = _plane_tuple(value)
    return (("p_xy", p_xy), ("p_xt", p_xt), ("p_yt", p_yt))
