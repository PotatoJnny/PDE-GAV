"""Actual solver-error guidance for a frozen initial-conditioned (IC40) EDM.

The separate noise-conditioned guide sees all future Schur-white values,
the complete clean initial context, sigma and physical nu/T. Numerical
failures train a separate classifier, NEVER large-error reward labels.
Sampling reuses native Heun with D_guided=D_base+strength*sigma^2*grad(utility).
This clipped reward-gradient heuristic is not a certified tilted physical law.
No base weights/normalization are trained; no initial coordinates are diffused.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .initial_conditioned_diffusion import VPFPInitialConditionedBatchSampler, VPFPInitialConditionedEDMConfig, _context
from .known_initial_prior import _json_sha
from .structured_diffusion import VPFPStructuredEDM, _SpatiotemporalBlock, _coefficients, _seeds, _sha
from .value_control_diffusion import VPFPValueFourierTransform

SCHEMA = "vpfp_frozen_ic_solver_error_guidance_v1"


@dataclass(frozen=True)
class SolverErrorGuidanceConfig:
    base_config: VPFPInitialConditionedEDMConfig
    base_checkpoint_sha256: str
    normalizer_sha256: str
    width: int = 32
    depth: int = 2
    error_scale: float = 1.
    validity_weight: float = .25
    schema: str = SCHEMA + "_config"

    def __post_init__(self):
        if not isinstance(self.base_config, VPFPInitialConditionedEDMConfig):
            raise TypeError("complete native initial-conditioned config required")
        _sha(self.base_checkpoint_sha256, "frozen IC checkpoint")
        _sha(self.normalizer_sha256, "frozen IC normalizer")
        if (type(self.width) is not int or self.width < 2 or type(self.depth) is not int or self.depth < 1
                or not math.isfinite(self.error_scale) or self.error_scale <= 0
                or not math.isfinite(self.validity_weight) or self.validity_weight <= 0
                or self.schema != SCHEMA + "_config"):
            raise ValueError("invalid solver-error guidance config")

    def to_dict(self):
        return {"base_config": self.base_config.to_dict(), "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "normalizer_sha256": self.normalizer_sha256, "width": self.width, "depth": self.depth,
            "error_scale": self.error_scale, "validity_weight": self.validity_weight, "schema": self.schema}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("unexpected guidance config fields")
        return cls(**dict(payload, base_config=VPFPInitialConditionedEDMConfig.from_dict(payload["base_config"])))


def validate_root_partition(train_root_ids, validation_root_ids):
    """Noising replicas/descendants stay in their original physical-root split."""
    train, validation = tuple(train_root_ids), tuple(validation_root_ids)
    if (not train or not validation or any(not isinstance(x, str) or not x.strip() for x in (*train, *validation))
            or set(train) & set(validation)):
        raise ValueError("nonempty disjoint physical-root train/validation partitions required")
    return {"train_root_ids": list(train), "validation_root_ids": list(validation),
        "independent_train_root_count": len(set(train)), "independent_validation_root_count": len(set(validation)),
        "independence_unit": "physical root before noise replication or descendants"}


def bounded_error_targets(macro_errors, numerically_valid, *, prototype, error_scale=1.):
    """Bounded monotone log1p error; invalid placeholders are regression-masked."""
    errors = torch.as_tensor(macro_errors, device=prototype.device, dtype=torch.float64).detach()
    valid = torch.as_tensor(numerically_valid, device=prototype.device)
    if (errors.ndim != 1 or valid.shape != errors.shape or valid.dtype != torch.bool
            or not math.isfinite(error_scale) or error_scale <= 0):
        raise ValueError("one error and explicit boolean numerical validity per root required")
    if not torch.isfinite(errors[valid]).all() or torch.any(errors[valid] < 0):
        raise ValueError("valid solver errors must be finite and nonnegative")
    target = torch.zeros_like(errors)
    positive = errors[valid]
    log_error = torch.where(positive > 0, positive.log(), positive.new_full(positive.shape, -torch.inf))
    log1p = torch.logaddexp(torch.zeros_like(positive), log_error - math.log(error_scale))
    target[valid] = log1p / (1 + log1p)
    return target.to(prototype), valid


class SolverErrorRegressor(nn.Module):
    """Full-field reward/validity network, with no owned base model parameters."""
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, SolverErrorGuidanceConfig):
            raise TypeError("explicit solver-error config required")
        self.config = config
        base, width = config.base_config, config.width
        self.transform = VPFPValueFourierTransform(base)
        self.input = nn.Conv3d(base.field_channels, width, 1)
        self.initial_input = nn.Conv2d(base.field_channels, width, 1)
        self.embedding = nn.Sequential(nn.Linear(18, width), nn.SiLU(), nn.Linear(width, width))
        self.time_position = nn.Linear(3, width)
        self.blocks = nn.ModuleList(_SpatiotemporalBlock(width,
            base.temporal_dilation_cycle[i % len(base.temporal_dilation_cycle)]) for i in range(config.depth))
        self.head = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 2))

    def _initial_field(self, context):
        cfg = self.config.base_config
        rho = context[:, :cfg.rho_modes]
        kinetic = context[:, cfg.rho_modes:].reshape(-1, cfg.hermite_channels, cfg.kinetic_modes)
        return torch.cat((torch.einsum("bm,mxy->bxy", rho, self.transform.rho_basis.to(context))[:, None],
            torch.einsum("bhm,mxy->bhxy", kinetic, self.transform.kinetic_basis.to(context))), 1)

    def forward(self, noisy_future_white, sigma, conditions, *, initial_context):
        cfg, parameter = self.config.base_config, next(self.parameters())
        noisy = _coefficients(noisy_future_white, cfg, device=parameter.device, dtype=parameter.dtype)
        if not len(noisy):
            raise ValueError("guidance needs a nonempty batch")
        context = _context(initial_context, cfg, prototype=noisy)
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        if (sigma.shape != (len(noisy),) or len(context) != len(noisy)
                or not torch.isfinite(sigma).all() or torch.any(sigma <= 0)):
            raise ValueError("positive sigma and complete clean IC for every future state required")
        condition = VPFPStructuredEDM._conditions(SimpleNamespace(config=cfg), conditions, len(noisy), noisy)
        field = self.transform.to_field(noisy / (sigma.square()+cfg.sigma_data**2).sqrt()[:, None])
        frequency = torch.exp(torch.linspace(0., math.log(100.), 8, dtype=noisy.dtype, device=noisy.device))
        phase = sigma.log()[:, None]*frequency[None]
        embedding = self.embedding(torch.cat((phase.sin(), phase.cos(), condition), 1))
        tau = torch.linspace(0., 1., cfg.full_config.control_count, dtype=noisy.dtype, device=noisy.device)[1:]
        position = self.time_position(torch.stack((tau, (math.pi*tau).sin(), (math.pi*tau).cos()), 1))
        hidden = self.input(field)+self.initial_input(self._initial_field(context))[:, :, None]
        hidden = hidden+position.T[None, :, :, None, None]
        for block in self.blocks:
            hidden = block(hidden, embedding)
        hidden = F.silu(hidden)
        pooled = torch.cat((hidden.mean((2, 3, 4)), hidden.square().mean((2, 3, 4))), 1)
        logits = self.head(pooled)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite solver-error prediction")
        return {"reward": logits[:, 0].sigmoid(), "validity_logit": logits[:, 1],
            "valid_probability": logits[:, 1].sigmoid()}

    def loss(self, clean_future_white, conditions, initial_context, macro_errors, numerically_valid,
             *, generator=None, sigma=None, noise=None):
        cfg, parameter = self.config.base_config, next(self.parameters())
        clean = _coefficients(clean_future_white, cfg, device=parameter.device, dtype=parameter.dtype).detach()
        context = _context(initial_context, cfg, prototype=clean).detach()
        if sigma is None:
            sigma = (torch.randn(len(clean), dtype=clean.dtype, device=clean.device, generator=generator)
                * cfg.log_sigma_std+cfg.log_sigma_mean).exp().clamp(cfg.sigma_min, cfg.sigma_max)
        sigma = torch.as_tensor(sigma, dtype=clean.dtype, device=clean.device)
        if (sigma.shape != (len(clean),) or not torch.isfinite(sigma).all()
                or torch.any(sigma < cfg.sigma_min) or torch.any(sigma > cfg.sigma_max)):
            raise ValueError("guide training sigma leaves the frozen EDM interval")
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator) if noise is None else _coefficients(
            noise, cfg, device=clean.device, dtype=clean.dtype)
        if noise.shape != clean.shape:
            raise ValueError("guidance noise batch differs")
        targets, valid = bounded_error_targets(macro_errors, numerically_valid, prototype=clean, error_scale=self.config.error_scale)
        if len(targets) != len(clean):
            raise ValueError("solver label and state counts differ")
        prediction = self(clean+sigma[:, None]*noise.detach(), sigma, conditions, initial_context=context)
        # Index BEFORE arithmetic; invalid NaN labels never enter this loss.
        reward_loss = (prediction["reward"][valid]-targets[valid]).square().mean() if bool(valid.any()) else prediction["reward"].sum()*0
        validity_loss = F.binary_cross_entropy_with_logits(prediction["validity_logit"], valid.to(clean))
        loss = reward_loss+self.config.validity_weight*validity_loss
        return loss, {"reward_loss": reward_loss.detach(), "validity_loss": validity_loss.detach(),
            "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()), "sigma": sigma.detach(),
            "reward_target": targets.detach(), "predicted_reward": prediction["reward"].detach()}

    def architecture_report(self):
        return {"schema": SCHEMA, "config": self.config.to_dict(),
            "input": "complete_future_Schur_white_and_complete_clean_parent_scaled_IC_plus_sigma_nu_T",
            "input_is_physical_amplitude": False, "base_denoiser_used_in_reward_training": False,
            "reward_target": "log1p(macro_error/error_scale)/(1+log1p(macro_error/error_scale))",
            "invalid_labels": "excluded_from_reward_regression_separate_validity_BCE",
            "sigma_law": "same_clipped_log_normal_as_frozen_EDM", "base_weights_updated": False,
            "parameters": sum(p.numel() for p in self.parameters()), "scientific_admission": False}


def make_solver_error_regressor(config, *, parameter_dtype, device="cpu"):
    """Explicit learned-parameter precision; preserve native FP64 Fourier buffers.

    Layer construction must use the requested dtype, not initialize in an
    ambient dtype and cast afterwards. This scoped construction restores the
    caller's global default even on failure and leaves the old FP32 RNG path
    unchanged. It is used outside concurrent model-construction sections.
    """
    if parameter_dtype not in (torch.float32, torch.float64):
        raise ValueError("guidance parameters require explicit float32 or float64")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter_dtype)
        model = SolverErrorRegressor(config)
    finally:
        torch.set_default_dtype(previous)
    return model.to(device=device)


def _validated_guide_state_dtype(state):
    """Validate all stored tensor dtypes before choosing construction precision."""
    fixed = {"transform.rho_basis", "transform.kinetic_basis"}
    if (not isinstance(state, dict) or not state or not fixed.issubset(state)
            or any(not isinstance(name, str) for name in state)):
        raise ValueError("guidance model state requires its named tensor inventory")
    learned_dtypes = set()
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype not in (torch.float32, torch.float64) or value.requires_grad
                or not torch.isfinite(value).all()):
            raise ValueError("invalid guidance model tensor: " + name)
        if name in fixed:
            if value.dtype != torch.float64:
                raise ValueError("immutable guidance Fourier buffers require native float64")
        else:
            learned_dtypes.add(value.dtype)
    if len(learned_dtypes) != 1:
        raise ValueError("guidance learned tensors require one consistent floating dtype")
    return next(iter(learned_dtypes))


def _validate_guide_state(model, state):
    """No load_state_dict implicit casts, including on training continuation."""
    dtype = _validated_guide_state_dtype(state)
    if dtype != next(model.parameters()).dtype:
        raise ValueError("guidance stored parameter dtype differs from the frozen base precision")
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError("guidance model tensor keys differ")
    for name, target in expected.items():
        tensor = state[name]
        if tensor.dtype != target.dtype or tensor.shape != target.shape:
            raise ValueError("invalid guidance model tensor: " + name)
        if name.startswith("transform.") and not torch.equal(tensor.cpu(), target.cpu()):
            raise ValueError("immutable complete Fourier guidance geometry changed")


def _module_sha(module):
    digest = sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(b"\0"); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _draw_inputs(base_sampler, seeds, conditions):
    if not isinstance(base_sampler, VPFPInitialConditionedBatchSampler):
        raise TypeError("strict native initial-conditioned sampler required")
    if base_sampler.model.training or any(p.requires_grad for p in base_sampler.model.parameters()):
        raise ValueError("native base must be eval/frozen")
    seeds = _seeds(seeds)
    stream = lambda tag, seed: int.from_bytes(sha256(f"vpfp-initial-conditioned-{tag}:{seed}".encode()).digest()[:8], "little") % 2**63
    prior_seeds, future_seeds = [stream("prior", s) for s in seeds], [stream("future", s) for s in seeds]
    # Native KnownInitialPrior accepts host NumPy conditions, including on GPU.
    if conditions is not None:
        conditions = conditions.detach().cpu().numpy() if isinstance(conditions, torch.Tensor) else conditions
        conditions = np.asarray(conditions, dtype=np.float32)
    initial = base_sampler.prior.sample(prior_seeds, conditions)
    parameter = next(base_sampler.model.parameters())
    context = base_sampler.normalizer.encode_initial(initial["initial_controls"].to(parameter))
    physical = torch.as_tensor(initial["equation_conditions"], dtype=parameter.dtype, device=parameter.device)
    if conditions is not None and not np.array_equal(physical.detach().cpu().numpy().astype(np.float32).view(np.uint32), conditions.view(np.uint32)):
        raise ValueError("known initial prior changed requested physical FP32 condition bits")
    return {"initial_context": context, "equation_conditions": physical, "known_prior_seeds": prior_seeds,
        "future_noise_seeds": future_seeds, "root_seeds": list(seeds)}


@torch.no_grad()
def draw_frozen_ic40(base_sampler, seeds, conditions=None):
    """Single actual base draw with clean guide-training coordinates; no guide."""
    model_sha, norm_sha = _module_sha(base_sampler.model), _json_sha(base_sampler.normalizer.to_payload())
    draw = _draw_inputs(base_sampler, seeds, conditions)
    future = base_sampler.model.sample(draw["initial_context"], draw["equation_conditions"], draw["future_noise_seeds"],
        steps=base_sampler.steps, churn=base_sampler.churn)
    draw.update(future_white=future, full=base_sampler.normalizer.decode((draw["initial_context"], future)))
    return draw


def native_batch_from_draw(draw, *, guidance_report=None):
    """Adapt a completed draw WITHOUT sampling again; safe for a resident replay."""
    full, physical = draw["full"], draw["equation_conditions"]
    seeds = _seeds(draw["root_seeds"])
    if (not torch.isfinite(full).all() or full.ndim != 2 or len(full) != len(seeds)
            or physical.shape != (len(seeds), 2) or not torch.isfinite(physical).all()):
        raise ValueError("invalid completed IC draw")
    return {"schema": "vpfp_learned_answer_batch_v1", "latents": full.detach().cpu().numpy().astype(np.float32),
        "equation_conditions": physical.detach().cpu().numpy().astype(np.float32),
        "equation_seeds": np.asarray(seeds, dtype=np.int64), "diagnostics": {"pde_solves": 0,
            "learned_generation": True, "known_prior_seeds": draw["known_prior_seeds"],
            "future_noise_seeds": draw["future_noise_seeds"], "guidance": deepcopy(guidance_report),
            "scientific_qualification": "not_established_by_sampling_or_guidance"}}


class GuidedIC40Sampler:
    """Freeze a trained guide and reuse the unmodified native Heun algorithm.

    The base and guide stay in evaluation mode with gradients disabled on
    their parameters. Sampling does not rescan or hash their tensors.
    """
    def __init__(self, base_sampler, guide, *, strength=1., max_gradient_norm=1., sigma_min=None, sigma_max=None):
        if not isinstance(base_sampler, VPFPInitialConditionedBatchSampler) or not isinstance(guide, SolverErrorRegressor):
            raise TypeError("native IC sampler and solver-error regressor required")
        cfg = guide.config
        if (base_sampler.model.config != cfg.base_config
                or base_sampler.identity["checkpoint_sha256"] != cfg.base_checkpoint_sha256
                or _json_sha(base_sampler.normalizer.to_payload()) != cfg.normalizer_sha256):
            raise ValueError("guide and frozen IC checkpoint/normalizer bindings differ")
        for name, value in (("strength", strength), ("max_gradient_norm", max_gradient_norm)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name == "max_gradient_norm" and value == 0):
                raise ValueError("invalid sampling guidance "+name)
        lo = cfg.base_config.sigma_min if sigma_min is None else sigma_min
        hi = cfg.base_config.sigma_max if sigma_max is None else sigma_max
        if (not math.isfinite(lo) or not math.isfinite(hi) or lo < cfg.base_config.sigma_min
                or hi > cfg.base_config.sigma_max or lo > hi):
            raise ValueError("guidance sigma window must stay within its training interval")
        if base_sampler.model.training or any(p.requires_grad for p in base_sampler.model.parameters()):
            raise ValueError("IC40 denoiser must already be eval/frozen")
        if next(guide.parameters()).device != next(base_sampler.model.parameters()).device or next(guide.parameters()).dtype != next(base_sampler.model.parameters()).dtype:
            raise ValueError("guide/base device and dtype differ")
        self.base, self.guide = base_sampler, guide.eval().requires_grad_(False)
        self.strength, self.max_gradient_norm = float(strength), float(max_gradient_norm)
        expected_strength = getattr(guide, "expected_sampling_strength", None)
        if expected_strength is not None and self.strength not in (0., expected_strength):
            raise ValueError("exponential guidance already includes beta; external strength must be one (or zero baseline)")
        self.sigma_min, self.sigma_max = float(lo), float(hi)
        self._base_sha, self._guide_sha = _module_sha(base_sampler.model), _module_sha(guide)
        self._normalizer_sha = _json_sha(base_sampler.normalizer.to_payload())
        self.identity = deepcopy(base_sampler.identity)
        self.identity.update(kind="frozen_IC40_solver_error_gradient_guidance", base_sampling_identity=deepcopy(base_sampler.identity),
            guidance={"schema": SCHEMA, "guide_state_sha256": self._guide_sha, "config": cfg.to_dict(),
                "strength": self.strength, "gradient_clip_l2": self.max_gradient_norm,
                "sigma_window": [self.sigma_min, self.sigma_max],
                "utility": "P(valid|noisy,IC,nu,T)*E(bounded_error|valid,noisy,IC,nu,T)",
                "score_correction": "+strength*clipped_white_coordinate_gradient",
                "reverse_integrator": "unmodified_native_Heun_and_churn",
                "exact_tilted_law_claimed": False, "scientific_admission": False})
        if callable(getattr(guide, "score_potential", None)):
            self.identity["guidance"].update(utility="log_joint_valid_exponential_reward_desirability",
                training_beta=guide.beta, beta_applied_once=True,
                approximation="finite_Heun_with_existing_sigma_window_and_gradient_clip")
            if getattr(guide, "objective", None) == "standardized_mean_reward":
                self.identity["guidance"]["utility"] = "beta_std*E(standardized_reward|valid,noisy,IC,nu,T)"
            if hasattr(guide, "beta_std"):
                self.identity["guidance"].update(training_objective=guide.objective,
                    training_beta_std=guide.beta_std, effective_raw_reward_beta=guide.beta,
                    auxiliary_validity_enters_score=False,
                    sampling_gain_separate_from_training_beta=True)
        self.last_report = {}
        self._identity_sha = _json_sha(self.identity)
        self._settings_sha = self._settings_digest()

    def _settings_digest(self):
        settings = {"guide_config": self.guide.config.to_dict(), "base_config": self.base.model.config.to_dict(),
            "steps": self.base.steps, "churn": self.base.churn, "base_identity": self.base.identity,
            "strength": self.strength, "clip": self.max_gradient_norm, "sigma_window": [self.sigma_min, self.sigma_max]}
        runtime_settings = getattr(self.guide, "runtime_settings", None)
        if callable(runtime_settings):
            settings["guide_runtime_settings"] = runtime_settings()
        return _json_sha(settings)

    def assert_frozen(self):
        expected_strength = getattr(self.guide, "expected_sampling_strength", None)
        if expected_strength is not None and self.strength not in (0., expected_strength):
            raise ValueError("exponential guidance external strength changed")
        if self.base.model.training or self.guide.training or any(p.requires_grad for p in self.base.model.parameters()) or any(p.requires_grad for p in self.guide.parameters()):
            raise ValueError("frozen base, guide or normalization changed")

    def sample_future(self, initial_context, conditions, future_seeds):
        self.assert_frozen()
        if self.strength and torch.is_inference_mode_enabled():
            raise ValueError("reward gradients require no_grad, not torch.inference_mode")
        model, cfg = self.base.model, self.guide.config.base_config
        seeds = _seeds(future_seeds)
        context = _context(initial_context, cfg, prototype=next(model.parameters())).detach()
        conditions = torch.as_tensor(conditions, dtype=context.dtype, device=context.device)
        model._conditions(conditions, len(seeds), context)
        if len(context) != len(seeds):
            raise ValueError("one full clean IC per future seed required")
        stats = {"denoise_calls": 0, "guided_calls": 0, "nonzero_gradient_calls": 0,
            "clipped_calls": 0, "maximum_raw_gradient_l2": 0., "maximum_applied_gradient_l2": 0.,
            "maximum_denoised_displacement_l2": 0., "sum_denoised_displacement_l2": 0., "strength": self.strength}
        rows = []
        try:
            for i, seed in enumerate(seeds):
                clean_ic = context[i:i+1]
                def denoise(noisy, sigma, physical):
                    with torch.no_grad():
                        baseline = model.denoise(noisy, sigma, physical, initial_context=clean_ic)
                    stats["denoise_calls"] += 1
                    if self.strength == 0 or not self.sigma_min <= float(sigma[0]) <= self.sigma_max:
                        return baseline
                    with torch.enable_grad():
                        state = noisy.detach().requires_grad_(True)
                        prediction = self.guide(state, sigma, physical, initial_context=clean_ic)
                        potential = getattr(self.guide, "score_potential", None)
                        utility = (potential(prediction) if callable(potential)
                                   else prediction["reward"]*prediction["valid_probability"])
                        gradient = torch.autograd.grad(utility.sum(), state, create_graph=False)[0]
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError("nonfinite solver-error reward gradient")
                    norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
                    if not torch.isfinite(norm).all():
                        raise FloatingPointError("nonfinite reward gradient norm")
                    scale = torch.where(norm > self.max_gradient_norm, self.max_gradient_norm/norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.ones_like(norm))
                    applied = gradient.detach()*scale
                    stats["guided_calls"] += 1
                    stats["nonzero_gradient_calls"] += int(bool(torch.any(norm > 0)))
                    stats["clipped_calls"] += int(bool(torch.any(norm > self.max_gradient_norm)))
                    stats["maximum_raw_gradient_l2"] = max(stats["maximum_raw_gradient_l2"], float(norm.max()))
                    stats["maximum_applied_gradient_l2"] = max(stats["maximum_applied_gradient_l2"], float(torch.linalg.vector_norm(applied, dim=1).max()))
                    displacement = self.strength*sigma[:, None].square()*applied
                    length = float(torch.linalg.vector_norm(displacement, dim=1).max())
                    if not math.isfinite(length):
                        raise FloatingPointError("nonfinite guided denoised displacement norm")
                    stats["maximum_denoised_displacement_l2"] = max(stats["maximum_denoised_displacement_l2"], length)
                    stats["sum_denoised_displacement_l2"] += length
                    guided = baseline+displacement
                    if not torch.isfinite(guided).all():
                        raise FloatingPointError("nonfinite guided denoised future")
                    return guided
                view = SimpleNamespace(config=cfg, parameters=model.parameters, _noise=model._noise,
                    _conditions=model._conditions, denoise=denoise)
                rows.append(VPFPStructuredEDM.sample(view, conditions[i:i+1], (seed,), steps=self.base.steps, churn=self.base.churn))
            return torch.cat(rows, 0)
        finally:
            self.last_report = stats
            self.assert_frozen()

    @torch.no_grad()
    def draw(self, seeds, conditions=None):
        draw = _draw_inputs(self.base, seeds, conditions)
        future = self.sample_future(draw["initial_context"], draw["equation_conditions"], draw["future_noise_seeds"])
        draw.update(future_white=future, full=self.base.normalizer.decode((draw["initial_context"], future)),
            guidance_report=deepcopy(self.last_report))
        self.assert_frozen()
        return draw

    def batch_from_draw(self, draw):
        return native_batch_from_draw(draw, guidance_report=draw["guidance_report"])

    @torch.no_grad()
    def sample_conditions(self, conditions, seeds):
        return self.draw(seeds, conditions)["full"]

    @torch.no_grad()
    def generate(self, seeds):
        return self.batch_from_draw(self.draw(seeds))


def guidance_payload(model, *, label_manifest_sha256, train_root_ids, validation_root_ids, training):
    """Tensor-only snapshot; no mutable base weights are embedded or rewritten."""
    _sha(label_manifest_sha256, "actual solver label manifest")
    roots = validate_root_partition(train_root_ids, validation_root_ids)
    metadata = json.loads(json.dumps(training, allow_nan=False))
    if metadata.get("label_source") not in ("actual_public_solver_macro_error", "synthetic_fixture"):
        raise ValueError("explicit actual-solver or synthetic-fixture label source required")
    return {"schema": SCHEMA, "config": model.config.to_dict(), "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "label_manifest_sha256": label_manifest_sha256, "root_partition": roots, "training": metadata,
        "architecture": model.architecture_report()}


def regressor_from_payload(payload, *, device="cpu"):
    fields = {"schema", "config", "model", "label_manifest_sha256", "root_partition", "training", "architecture"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != SCHEMA:
        raise ValueError("unexpected solver-error guidance checkpoint fields")
    _sha(payload["label_manifest_sha256"], "actual solver label manifest")
    roots = payload["root_partition"]
    if roots != validate_root_partition(roots["train_root_ids"], roots["validation_root_ids"]):
        raise ValueError("guidance root partition metadata differs")
    if payload["training"].get("label_source") not in ("actual_public_solver_macro_error", "synthetic_fixture"):
        raise ValueError("guidance label source missing")
    state = payload["model"]
    parameter_dtype = _validated_guide_state_dtype(state)
    with torch.random.fork_rng(devices=[]):
        model = make_solver_error_regressor(SolverErrorGuidanceConfig.from_dict(payload["config"]),
            parameter_dtype=parameter_dtype)
    if json.dumps(payload["architecture"], sort_keys=True) != json.dumps(model.architecture_report(), sort_keys=True):
        raise ValueError("guidance architecture metadata differs")
    _validate_guide_state(model, state)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_solver_error_regressor(path, expected_sha256, *, device="cpu"):
    _sha(expected_sha256, "guidance checkpoint")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("guidance checkpoint must be a regular file")
    with path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    from . import calibrated_solver_error_guidance as calibrated
    from . import exponential_solver_error_guidance as exponential
    from . import standardized_solver_error_guidance as standardized
    if isinstance(payload, dict) and payload.get("schema") == standardized.SCHEMA:
        return standardized.regressor_from_payload(payload, device=device)
    if isinstance(payload, dict) and payload.get("schema") == exponential.SCHEMA:
        return exponential.regressor_from_payload(payload, device=device)
    if isinstance(payload, dict) and payload.get("schema") == calibrated.SCHEMA:
        return calibrated.regressor_from_payload(payload, device=device)
    return regressor_from_payload(payload, device=device)
