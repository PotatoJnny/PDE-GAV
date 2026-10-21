"""Versioned amplitude controls for the native VPFP EDM guidance experiment.

The controlled quantity is the future-white ODE drift (x-D)/sigma, NOT D.
RMS ratios in these coordinates are not percentages of physical perturbation.
The numerical floors below do not establish learned-direction reliability.
State-dependent gain, clipping and a finite Heun solve do not define a claimed
exact exponentially tilted sampling law. No initial coordinates are modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from types import SimpleNamespace

import torch

from . import solver_error_guidance as legacy


SCHEMA = "vpfp_ic40_drift_rms_guidance_v1"


@dataclass(frozen=True)
class RescalingPolicy:
    mode: str = "natural"
    fixed_gain: float = 1.
    target_ratio: float = .1
    gain_max: float = 1.e6
    max_flow_ratio: float = .25
    minimum_gradient_rms: float = 1.e-12
    minimum_base_drift_rms: float = 1.e-8
    random_direction: bool = False
    random_seed: int = 1720260908

    def __post_init__(self):
        if self.mode not in ("natural", "fixed", "relative"):
            raise ValueError("rescaling mode must be natural, fixed or relative")
        for name in ("fixed_gain", "target_ratio", "gain_max", "max_flow_ratio",
                     "minimum_gradient_rms", "minimum_base_drift_rms"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name != "fixed_gain" and value == 0)):
                raise ValueError("invalid rescaling policy " + name)
        if type(self.random_direction) is not bool:
            raise ValueError("random_direction must be a boolean")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2 ** 63:
            raise ValueError("random_seed must be a signed-64-bit nonnegative integer")
        if self.mode == "natural" and (self.random_direction or self.fixed_gain != 1.):
            raise ValueError("natural preserves the original unit-strength learned guidance")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("unexpected rescaling policy fields")
        return cls(**value)


def _rms(values):
    """Detached FP64 scaled reduction: tiny FP32 gradients do not underflow."""
    values = values.detach().double()
    peak = float(values.abs().max())
    if not math.isfinite(peak):
        raise FloatingPointError("nonfinite guidance diagnostic input")
    return peak * float((values / peak).square().mean().sqrt()) if peak else 0.


def _ratio(numerator, denominator):
    if denominator == 0.:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _cosine(a, b):
    a, b = a.detach().double(), b.detach().double()
    a_rms, b_rms = _rms(a), _rms(b)
    if a_rms == 0. or b_rms == 0.:
        return None
    return max(-1., min(1., float(((a / a_rms) * (b / b_rms)).mean())))


@torch.no_grad()
def apply_rescaled_guidance(noisy, baseline, gradient, sigma, policy, *,
                            max_gradient_norm=1., random_direction=None, enabled=True):
    """Apply one B1 native-precision correction and return detached scalars.

    Rescaling acts on an ALREADY differentiated guide potential. The random
    control replaces only its direction, matching the learned clipped RMS and
    scalar gain at this same local state; paths need not subsequently coincide.
    Natural mode intentionally keeps the original FP32 norm/clip arithmetic.
    """
    if not isinstance(policy, RescalingPolicy):
        raise TypeError("a frozen RescalingPolicy is required")
    policy.__post_init__()
    if (not isinstance(noisy, torch.Tensor) or noisy.ndim != 2 or noisy.shape[0] != 1
            or noisy.shape[1] == 0 or noisy.dtype not in (torch.float32, torch.float64)):
        raise ValueError("one nonempty FP32/FP64 future-white row required")
    for value in (baseline, gradient):
        if (not isinstance(value, torch.Tensor) or value.shape != noisy.shape
                or value.dtype != noisy.dtype or value.device != noisy.device):
            raise ValueError("state, denoiser and gradient must have matching shape/device/dtype")
    if (not isinstance(sigma, torch.Tensor) or sigma.shape != (1,)
            or sigma.dtype != noisy.dtype or sigma.device != noisy.device
            or not bool(torch.isfinite(sigma).all()) or float(sigma[0]) <= 0):
        raise ValueError("one finite positive native-precision sigma required")
    if (isinstance(max_gradient_norm, bool) or not math.isfinite(max_gradient_norm)
            or max_gradient_norm <= 0 or type(enabled) is not bool):
        raise ValueError("positive finite gradient clip and boolean enabled required")
    if any(not bool(torch.isfinite(value).all()) for value in (noisy, baseline, gradient)):
        raise FloatingPointError("nonfinite state, denoiser or guidance gradient")
    noisy, baseline, gradient = noisy.detach(), baseline.detach(), gradient.detach()
    s = float(sigma[0])
    residual = noisy - baseline
    drift = residual / sigma[:, None]
    residual_rms, drift_rms, raw_rms = _rms(residual), _rms(drift), _rms(gradient)
    norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    raw_l2_32 = float(norm.max())
    raw_l2_64 = raw_rms * math.sqrt(gradient.numel())
    if not math.isfinite(raw_l2_64):
        raise FloatingPointError("nonfinite FP64 gradient norm")
    # Exact original natural branch: this intentionally retains its norm dtype.
    if policy.mode == "natural":
        if not math.isfinite(raw_l2_32):
            raise FloatingPointError("nonfinite original guidance gradient norm")
        clip_scale = torch.where(norm > max_gradient_norm,
            max_gradient_norm / norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.ones_like(norm))
        clipped = gradient * clip_scale
        gradient_clipped = bool(torch.any(norm > max_gradient_norm))
    else:
        clipped = gradient * (min(1., max_gradient_norm / raw_l2_64) if raw_l2_64 else 1.)
        gradient_clipped = raw_l2_64 > max_gradient_norm
    clipped_rms = _rms(clipped)
    direction = clipped
    skip = None if enabled else "outside_sigma_window"
    requested_gain, gain = 1., 1.
    gain_capped = flow_capped = False
    if enabled and policy.mode != "natural":
        if raw_rms < policy.minimum_gradient_rms or clipped_rms == 0.:
            skip = "near_zero_gradient"
        elif drift_rms < policy.minimum_base_drift_rms:
            skip = "near_zero_base_drift"
        if not skip:
            if policy.mode == "fixed":
                requested_gain = policy.fixed_gain
            else:
                # Log arithmetic avoids overflowing a gain that will be capped.
                log_gain = (math.log(policy.target_ratio) + math.log(residual_rms)
                            - 2. * math.log(s) - math.log(clipped_rms))
                requested_gain = math.exp(log_gain) if log_gain < 709. else None
            gain_capped = requested_gain is None or requested_gain > policy.gain_max
            gain = policy.gain_max if gain_capped else requested_gain
            # Bound relative ODE drift, not denoiser amplitude. No positive
            # floor is substituted for a vanishing base drift.
            log_allowed_gain = (math.log(policy.max_flow_ratio) + math.log(drift_rms)
                                - math.log(s) - math.log(clipped_rms))
            allowed_gain = math.exp(log_allowed_gain) if log_allowed_gain < 709. else math.inf
            flow_capped = gain > allowed_gain
            gain = min(gain, allowed_gain)
            if gain == 0.:
                skip = "zero_gain"
    if enabled and policy.random_direction and not skip:
        if (not isinstance(random_direction, torch.Tensor) or random_direction.shape != noisy.shape
                or random_direction.dtype != noisy.dtype or random_direction.device != noisy.device):
            raise ValueError("random control requires one matching independently sampled direction")
        random_rms = _rms(random_direction)
        if random_rms == 0.:
            skip = "degenerate_random_direction"
        else:
            direction = (random_direction.detach().double() * (clipped_rms / random_rms)).to(noisy.dtype)
    if skip:
        gain = 0.
        displacement = torch.zeros_like(baseline)
        guided = baseline
    else:
        # Keep sigma squared exactly once and the original multiplication order.
        displacement = gain * sigma[:, None].square() * direction
        guided = baseline + displacement
    if not bool(torch.isfinite(guided).all()) or not bool(torch.isfinite(displacement).all()):
        raise FloatingPointError("nonfinite rescaled guided denoiser")
    actual_delta = guided - baseline
    delta_drift = -displacement / sigma[:, None]
    actual_drift_delta = (noisy - guided) / sigma[:, None] - drift
    applied_drift_rms, actual_drift_rms = _rms(delta_drift), _rms(actual_drift_delta)
    record = {
        "sigma": s, "mode": policy.mode, "random_direction": policy.random_direction,
        "base_residual_rms": residual_rms, "base_drift_rms": drift_rms,
        "raw_gradient_rms": raw_rms, "raw_gradient_l2_fp64": raw_l2_64,
        "raw_gradient_l2_native": raw_l2_32 if math.isfinite(raw_l2_32) else None,
        "clipped_gradient_rms": clipped_rms,
        "clipped_gradient_l2_fp64": clipped_rms * math.sqrt(gradient.numel()),
        "gradient_clipped": gradient_clipped,
        "requested_gain": requested_gain, "requested_gain_overflow": requested_gain is None,
        "effective_gain": gain, "gain_capped": gain_capped, "flow_capped": flow_capped,
        "skip_reason": skip,
        "raw_delta_denoised_rms": _rms(sigma[:, None].double().square() * gradient.double()),
        "raw_delta_drift_rms": _rms(-sigma[:, None].double() * gradient.double()),
        "applied_delta_denoised_rms": _rms(displacement),
        "applied_delta_drift_rms": applied_drift_rms,
        "applied_flow_ratio": _ratio(applied_drift_rms, drift_rms),
        "actual_delta_denoised_rms": _rms(actual_delta),
        "actual_delta_denoised_over_sigma_rms": _rms(actual_delta / sigma[:, None]),
        "actual_delta_drift_rms": actual_drift_rms,
        "actual_flow_ratio": _ratio(actual_drift_rms, drift_rms),
        "actual_changed_coordinates": int(torch.count_nonzero(actual_delta)),
        "guidance_base_drift_cosine": _cosine(delta_drift, drift),
        "numerical_floors_are_reliability_test": False,
    }
    return guided, record


class RescaledIC40Sampler(legacy.GuidedIC40Sampler):
    """Reuse native Heun and all inherited answer-first draw/batch operations."""

    def __init__(self, base, guide, policy, *, max_gradient_norm=1., sigma_min=.002, sigma_max=2.):
        if not isinstance(policy, RescalingPolicy):
            raise TypeError("a frozen RescalingPolicy is required")
        policy.__post_init__()
        self.policy = policy
        # Unit strength is natural exponential guidance. A separate policy gain
        # is explicit controlled sampling, NOT an extra target beta.
        super().__init__(base, guide, strength=1., max_gradient_norm=max_gradient_norm,
                         sigma_min=sigma_min, sigma_max=sigma_max)
        self.identity.update(kind="frozen_IC40_rescaled_solver_error_guidance")
        self.identity["guidance"].update(rescaling_schema=SCHEMA, rescaling_policy=policy.to_dict(),
            gain_semantics="sampling_control_not_target_beta",
            controlled_quantity="RMS(delta_ODE_drift)/RMS((x-D_base)/sigma)",
            physical_percentage_claimed=False, exact_tilted_law_claimed=False,
            numerical_floors_are_reliability_test=False,
            score_correction="+policy_gain*clipped_white_coordinate_gradient")
        self._identity_sha = legacy._json_sha(self.identity)
        self._settings_sha = self._settings_digest()

    def _settings_digest(self):
        return legacy._json_sha({"legacy_settings_sha256": super()._settings_digest(),
                                 "rescaling_schema": SCHEMA, "policy": self.policy.to_dict()})

    def sample_future(self, initial_context, conditions, future_seeds):
        self.assert_frozen()
        if torch.is_inference_mode_enabled():
            raise ValueError("reward gradients require no_grad, not torch.inference_mode")
        model, cfg = self.base.model, self.guide.config.base_config
        seeds = legacy._seeds(future_seeds)
        context = legacy._context(initial_context, cfg, prototype=next(model.parameters())).detach()
        conditions = torch.as_tensor(conditions, dtype=context.dtype, device=context.device)
        model._conditions(conditions, len(seeds), context)
        if len(context) != len(seeds):
            raise ValueError("one full clean IC per future seed required")
        stats = {"schema": SCHEMA, "policy": self.policy.to_dict(), "records": [],
            "denoise_calls": 0, "guided_calls": 0, "nonzero_gradient_calls": 0,
            "clipped_calls": 0, "maximum_raw_gradient_l2": 0., "maximum_applied_gradient_l2": 0.,
            "maximum_denoised_displacement_l2": 0., "sum_denoised_displacement_l2": 0.,
            "strength": 1., "gain_capped_calls": 0, "flow_capped_calls": 0, "skipped_calls": 0}
        rows = []
        try:
            for i, seed in enumerate(seeds):
                clean_ic = context[i:i+1]
                private_seed = int.from_bytes(sha256(
                    f"{SCHEMA}:{self.policy.random_seed}:{seed}".encode()).digest()[:8], "big") % (2 ** 63)
                random = torch.Generator(device=context.device).manual_seed(private_seed)
                call_index = 0

                def denoise(noisy, sigma, physical):
                    nonlocal call_index
                    with torch.no_grad():
                        baseline = model.denoise(noisy, sigma, physical, initial_context=clean_ic)
                    enabled = self.sigma_min <= float(sigma[0]) <= self.sigma_max
                    if enabled:
                        with torch.enable_grad():
                            state = noisy.detach().requires_grad_(True)
                            prediction = self.guide(state, sigma, physical, initial_context=clean_ic)
                            potential = getattr(self.guide, "score_potential", None)
                            utility = (potential(prediction) if callable(potential)
                                       else prediction["reward"] * prediction["valid_probability"])
                            gradient = torch.autograd.grad(utility.sum(), state, create_graph=False)[0]
                    else:
                        gradient = torch.zeros_like(noisy)
                    direction = (torch.randn(noisy.shape, dtype=noisy.dtype, device=noisy.device,
                        generator=random) if enabled and self.policy.random_direction else None)
                    guided, record = apply_rescaled_guidance(noisy, baseline, gradient, sigma, self.policy,
                        max_gradient_norm=self.max_gradient_norm, random_direction=direction, enabled=enabled)
                    record.update(root_seed=seed, call_index=call_index,
                        heun_stage="predictor" if call_index % 2 == 0 else "corrector")
                    stats["records"].append(record)
                    call_index += 1
                    stats["denoise_calls"] += 1
                    if enabled:
                        stats["guided_calls"] += 1
                        stats["nonzero_gradient_calls"] += int(record["raw_gradient_l2_fp64"] > 0.)
                        stats["clipped_calls"] += int(record["gradient_clipped"])
                        stats["maximum_raw_gradient_l2"] = max(stats["maximum_raw_gradient_l2"], record["raw_gradient_l2_fp64"])
                        stats["maximum_applied_gradient_l2"] = max(stats["maximum_applied_gradient_l2"], record["clipped_gradient_l2_fp64"])
                        length = record["applied_delta_denoised_rms"] * math.sqrt(gradient.numel())
                        stats["maximum_denoised_displacement_l2"] = max(stats["maximum_denoised_displacement_l2"], length)
                        stats["sum_denoised_displacement_l2"] += length
                    stats["gain_capped_calls"] += int(record["gain_capped"])
                    stats["flow_capped_calls"] += int(record["flow_capped"])
                    stats["skipped_calls"] += int(record["skip_reason"] is not None)
                    return guided

                view = SimpleNamespace(config=cfg, parameters=model.parameters, _noise=model._noise,
                    _conditions=model._conditions, denoise=denoise)
                rows.append(legacy.VPFPStructuredEDM.sample(view, conditions[i:i+1], (seed,),
                    steps=self.base.steps, churn=self.base.churn))
            return torch.cat(rows, 0)
        finally:
            self.last_report = stats
            self.assert_frozen()
