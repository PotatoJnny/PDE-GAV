"""Request-local CUDA stage replay for the isolated NS Controller process.

The caller serializes Generator operations in its isolated process. A shared NS
lock also excludes concurrent truth queries during warmup/capture. This module
is an optional NS implementation detail, not a general execution service.
Unsupported models or tensor layouts return None for the existing eager path.
Actual CUDA errors propagate; they are never treated as a successful fallback.
"""
from __future__ import annotations

from .continuous_backend import _NS_CUDA_CAPTURE_LOCK


def _eligible(torch, model, initial, guidance) -> bool:
    if (
        type(initial) is not torch.Tensor
        or initial.device.type != "cuda"
        or initial.dtype != torch.float32
        or initial.ndim != 2
        or int(initial.shape[0]) < 1
        or not initial.is_contiguous()
        or torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    # A custom GPU model may be used without the optional legacy package.
    if type(model).__module__ != "agent_verifier.neural_generator.structured_ns_distribution":
        return False
    from agent_verifier.neural_generator.structured_ns_distribution import (
        _StructuredCoefficientFlowBlock,
        _StructuredCoefficientRectifiedFlow,
    )
    from agent_verifier.neural_generator.structured_ns_guidance import (
        StructuredRewardGradient,
        _RewardBlock,
        _StructuredRewardModel,
    )
    if type(model) is not _StructuredCoefficientRectifiedFlow:
        return False
    if initial.shape[1] != model.config.feature_dim:
        return False
    if guidance is not None and (
        type(guidance) is not StructuredRewardGradient
        or type(guidance.model) is not _StructuredRewardModel
        or guidance.model.config.feature_dim != initial.shape[1]
        or "_tensor_update" in vars(guidance)
        or "_commit_diagnostics" in vars(guidance)
        or "__call__" in vars(guidance)
    ):
        return False
    allowed = {
        _StructuredCoefficientRectifiedFlow, _StructuredCoefficientFlowBlock,
        _StructuredRewardModel, _RewardBlock, torch.nn.ModuleList,
        torch.nn.Sequential, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.SiLU,
    }
    # Global hooks, like per-module hooks, would otherwise execute only during
    # warmup/capture and disappear on replay.
    hook_names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                  "_backward_pre_hooks")
    module_namespace = torch.nn.modules.module
    if any(getattr(module_namespace, "_global" + name, {}) for name in hook_names):
        return False
    models = (model,) if guidance is None else (model, guidance.model)
    for current in models:
        for layer in current.modules():
            if (
                type(layer) not in allowed or layer.training
                or any(name in vars(layer) for name in ("forward", "_call_impl", "_wrapped_call_impl"))
                or getattr(layer, "_compiled_call_impl", None) is not None
                or any(getattr(layer, name, {}) for name in hook_names)
            ):
                return False
        for parameter in current.parameters():
            if parameter.device != initial.device or parameter.dtype != torch.float32:
                return False
            if current is not model and parameter.requires_grad:
                return False
    return True


class _FlowStageGraph:
    def __init__(self, torch, model, values, guidance, guidance_scale):
        self.torch = torch
        self.guidance = guidance
        self.state = values.clone()
        self.time = torch.zeros(
            values.shape[0], device=values.device, dtype=values.dtype,
        )

        def compute():
            velocity = model(self.state, self.time)
            if guidance is None:
                return velocity, None
            tensors = guidance._tensor_update(self.state, self.time, velocity)
            return velocity + guidance_scale * tensors[0], tensors

        with _NS_CUDA_CAPTURE_LOCK:
            self.graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=values.device)
            current = torch.cuda.current_stream(values.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compute()
            current.wait_stream(stream)
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = compute()

    def __call__(self, values, time):
        self.state.copy_(values)
        self.time.copy_(time)
        self.graph.replay()
        velocity, tensors = self.output
        if self.guidance is not None:
            # Includes the immediate finite check before this stage's result
            # is returned. Reductions allocate new scalars, so the original
            # diagnostic accumulator cannot alias graph output storage.
            self.guidance._commit_diagnostics(int(values.shape[0]), tensors)
        # Every replay overwrites its static outputs. Preserve the ordinary
        # forward contract, including the first velocity in a Heun step.
        return velocity.clone()


def sample_flow_cuda_graph(
    model, initial_values, *, steps, method, guidance=None, guidance_scale=0.0,
):
    """Sample with unchanged Euler/Heun arithmetic, or return None if ineligible.

    Initial noise is supplied by the caller; this function makes no RNG calls.
    Exactly one graph is captured per sampling call, with no cache shared across
    requests, models, streams, or batch sizes. The caller guarantees an isolated
    Controller process and keeps the frozen models unchanged until return.
    """
    import torch

    active_guidance = guidance if guidance is not None and float(guidance_scale) != 0.0 else None
    if method not in {"euler", "heun"} or int(steps) < 1:
        return None
    if not _eligible(torch, model, initial_values, active_guidance):
        return None
    with torch.cuda.device(initial_values.device):
        values = initial_values.clone()
        stage = _FlowStageGraph(torch, model, values, active_guidance, float(guidance_scale))
        step_size = 1.0 / int(steps)
        try:
            for index in range(int(steps)):
                time = torch.full(
                    (int(values.shape[0]),), index / int(steps),
                    device=values.device, dtype=values.dtype,
                )
                velocity = stage(values, time)
                if method == "euler" or index + 1 == int(steps):
                    values = values + step_size * velocity
                    continue
                provisional = values + step_size * velocity
                next_time = torch.full_like(time, (index + 1) / int(steps))
                next_velocity = stage(provisional, next_time)
                values = values + 0.5 * step_size * (velocity + next_velocity)
            return values
        finally:
            # The graph and its private tensors are request-local. Finish
            # queued consumers before releasing their storage on either exit.
            torch.cuda.current_stream(initial_values.device).synchronize()
