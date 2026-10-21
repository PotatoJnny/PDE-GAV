"""Detached activation observations; no changes to Guidance training or loss."""

from copy import deepcopy
import math

import torch


def tensor_stats(tensor):
    """Return JSON-safe statistics over finite entries of one real tensor.

    Nonfinite entries are counted explicitly, never treated as zero. Reductions
    use detached FP64 values and a scaled RMS so small FP32 gradients do not
    disappear through squaring. Empty/all-nonfinite tensors have null summaries.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.is_complex() or tensor.layout != torch.strided:
        raise TypeError("diagnostics require one real strided tensor")
    with torch.no_grad():
        value = tensor.detach().to(dtype=torch.float64)
        finite = torch.isfinite(value)
        count = value.numel()
        finite_count = int(finite.sum().item())
        result = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype), "count": count,
            "finite_count": finite_count, "nonfinite_count": count - finite_count,
            "nan_count": int(torch.isnan(value).sum().item()),
            "posinf_count": int(torch.isposinf(value).sum().item()),
            "neginf_count": int(torch.isneginf(value).sum().item()),
            "all_finite": finite_count == count,
            "statistics_scope": "finite_entries_only",
            "min": None, "max": None, "mean": None, "rms": None,
        }
        if finite_count:
            values = value[finite]
            result["min"] = float(values.min().item())
            result["max"] = float(values.max().item())
            scale = float(values.abs().max().item())
            if scale == 0.:
                result.update(mean=0., rms=0.)
            else:
                normalized = values / scale
                mean = float(normalized.mean().item()) * scale
                rms = math.sqrt(float(normalized.square().mean().item())) * scale
                # Even scalar reduction overflow must remain strict JSON-safe.
                result["mean"] = mean if math.isfinite(mean) else None
                result["rms"] = rms if math.isfinite(rms) else None
        return result


class ActivationProbe:
    """Observe only the latest forward of a compatible Guidance network.

    ``head.input`` is the pooled representation; ``head.output`` is the pair of
    raw logits. Block inputs refer to the first (hidden-field) argument. Hooks
    return None and store only scalar dictionaries, never tensors or graphs.
    Partial forwards are retained if the model raises, without stale activations
    from the previous call. The probe neither changes train/eval mode nor RNG.
    """

    def __init__(self, model):
        self.model = model
        self._handles = []
        self._active = False
        self._forward_calls = 0
        self._activations = {}

    def _begin_forward(self, _module, _inputs):
        self._forward_calls += 1
        self._activations = {}

    def _before(self, name):
        def observe(_module, inputs):
            self._activations[name] = {"input": tensor_stats(inputs[0])}
        return observe

    def _after(self, name):
        def observe(_module, _inputs, output):
            self._activations[name]["output"] = tensor_stats(output)
        return observe

    def __enter__(self):
        if self._active:
            raise RuntimeError("ActivationProbe context is already active")
        modules = [("input", self.model.input), ("initial_input", self.model.initial_input)]
        modules.extend(("blocks." + str(index), block) for index, block in enumerate(self.model.blocks))
        modules.append(("head", self.model.head))
        self._active = True
        self._forward_calls = 0
        self._activations = {}
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._begin_forward))
            for name, module in modules:
                self._handles.append(module.register_forward_pre_hook(self._before(name)))
                self._handles.append(module.register_forward_hook(self._after(name)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def snapshot(self):
        return deepcopy({"forward_calls": self._forward_calls, "activations": self._activations})

    def __exit__(self, _exception_type, _exception, _traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        return False
