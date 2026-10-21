"""Native hierarchical weak-VPFP reward heads; no base/decoder modification.

Each head regresses E[I_valid exp(beta * standardized bounded error) / Z_fit]
under native white-coordinate noising. Its raw log prediction is the guidance
potential. The relative-drift sampling policy is not an exact tilted-law claim.
Only fit data calibrate scales; neither development labels nor a minibatch
renormalize targets. Micro context is the original clean macro condition code.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from types import SimpleNamespace

import numpy as np  # Preserve the repository's NumPy-before-Torch convention.
import torch
from torch import nn
from torch.nn import functional as F

from .structured_diffusion import VPFPStructuredEDM, _coefficients

SCHEMA = "vpfp_weak_hierarchical_solver_guidance_v1"
BETA_STD = .5
REWARD_STD_FLOOR = 1e-8
FIELD_STD_FLOOR = 1e-6
POLICY = dict(target_ratio=.1, cap_ratio=.25, sigma_min=.002, sigma_max=2.,
              gradient_rms_floor=1e-12, drift_rms_floor=1e-8, gain_cap=1e6)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tensor_sha(value):
    tensor = value.detach().cpu().contiguous()
    return sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode()
                  + tensor.numpy().tobytes()).hexdigest()


def _white_field_variance(transform, stage):
    """Exact channel-average field variance for native iid unit white noise."""
    if stage == "micro":
        rho = transform.rho_basis.square().sum(0).mean()
        kinetic = transform.kinetic_basis.square().sum(0).mean()
        return torch.stack([rho] + [kinetic] * transform.config.hermite_channels).float()
    base = transform.base if hasattr(transform, "packing") else transform
    global_variance = (transform.packing.square().sum(1).mean() if hasattr(transform, "packing")
                       else base.basis.new_tensor(1.))
    local = base.basis[1:].square().sum(0).mean()
    stress = base.basis.square().sum(0).mean()
    return torch.stack([global_variance] * 2 + [local] * 2 + [stress] * 3).float()


class _SpatialFourierBlock(nn.Module):
    """Spatial Fourier mixing + nonperiodic time convolution, no time FFT."""
    def __init__(self, width, grid):
        super().__init__()
        self.modes = min(3, grid // 2)
        # Real/imag storage keeps all trainable parameters native real FP32.
        self.weight = nn.Parameter(torch.randn(2, width, width, self.modes,
                                               self.modes, 2, dtype=torch.float32) / width)
        self.local = nn.Conv3d(width, width, 1, dtype=torch.float32)
        self.time = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0),
                              groups=width, dtype=torch.float32)
        self.film = nn.Linear(width, 2 * width, dtype=torch.float32)

    def forward(self, value, embedding):
        scale, shift = self.film(embedding).chunk(2, dim=1)
        hidden = F.silu(value * (1 + scale[:, :, None, None, None])
                        + shift[:, :, None, None, None])
        spectrum = torch.fft.rfft2(hidden, norm="ortho")
        result = torch.zeros_like(spectrum)
        k = self.modes
        weights = torch.view_as_complex(self.weight.contiguous())
        result[..., :k, :k] = torch.einsum("bitxy,ioxy->botxy", spectrum[..., :k, :k], weights[0])
        result[..., -k:, :k] = torch.einsum("bitxy,ioxy->botxy", spectrum[..., -k:, :k], weights[1])
        spatial = torch.fft.irfft2(result, s=value.shape[-2:], norm="ortho")
        return value + spatial + self.local(hidden) + self.time(hidden)


class WeakSolverGuidanceHead(nn.Module):
    """One scalar log-desirability head, without registering the base model."""
    def __init__(self, base_sampler, stage, width=32, depth=2):
        super().__init__()
        if stage not in ("macro", "micro") or type(width) is not int or width < 1 or type(depth) is not int or depth < 1:
            raise ValueError("explicit macro/micro stage and positive width/depth required")
        base = getattr(base_sampler, stage)
        self.stage, self.width, self.depth = stage, width, depth
        self.config = base.config
        self.strength = 1.
        self.calibration = None
        self.base_identity = json.loads(_json(base_sampler.identity))
        self.base_class = type(base).__module__ + "." + type(base).__qualname__
        self.transform = deepcopy(base.transform).requires_grad_(False)
        self.macro_transform = (deepcopy(base.macro_transform).requires_grad_(False)
                                if stage == "micro" else None)
        self.macro_config = base.macro_config if stage == "micro" else None
        channels = self.config.field_channels
        self.input = nn.Conv3d(channels, width, 1, dtype=torch.float32)
        self.macro_input = nn.Conv3d(7, width, 1, dtype=torch.float32) if stage == "micro" else None
        self.embedding = nn.Sequential(nn.Linear(18, width, dtype=torch.float32), nn.SiLU(),
                                       nn.Linear(width, width, dtype=torch.float32))
        self.time_position = nn.Linear(3, width, dtype=torch.float32)
        self.blocks = nn.ModuleList(_SpatialFourierBlock(width, self.config.grid_size) for _ in range(depth))
        self.head = nn.Linear(2 * width, 1, dtype=torch.float32)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("field_mean", torch.zeros(channels, dtype=torch.float32))
        self.register_buffer("field_scale", torch.ones(channels, dtype=torch.float32))
        self.register_buffer("white_field_variance", _white_field_variance(self.transform, stage))
        if stage == "micro":
            self.register_buffer("macro_mean", torch.zeros(7, dtype=torch.float32))
            self.register_buffer("macro_scale", torch.ones(7, dtype=torch.float32))

    def _inputs(self, values, sigma, conditions, macro_context):
        parameter = next(self.parameters())
        values = _coefficients(values, self.config, device=parameter.device, dtype=parameter.dtype)
        if not len(values):
            raise ValueError("nonempty guidance batch required")
        sigma = torch.as_tensor(sigma, device=values.device, dtype=values.dtype)
        if sigma.shape != (len(values),) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("sigma must be finite positive with shape [B]")
        condition = VPFPStructuredEDM._conditions(SimpleNamespace(config=self.config), conditions, len(values), values)
        if self.stage == "micro":
            if macro_context is None:
                raise ValueError("micro guidance needs the clean encoded macro context")
            context = _coefficients(macro_context, self.macro_config, device=values.device, dtype=values.dtype)
            if len(context) != len(values):
                raise ValueError("macro context batch mismatch")
        else:
            if macro_context is not None:
                raise ValueError("macro prior guidance must not receive micro macro context")
            context = None
        return values, sigma, condition, context

    @staticmethod
    def _labels(errors, valid, prototype):
        mask = torch.as_tensor(valid, dtype=torch.bool, device=prototype.device)
        if mask.ndim != 1 or len(mask) != len(prototype):
            raise ValueError("one validity flag per row required")
        if isinstance(errors, torch.Tensor):
            values = errors.detach().to(device=prototype.device, dtype=torch.float64)
        else:
            values = torch.tensor([float("nan") if x is None else x for x in errors],
                                  device=prototype.device, dtype=torch.float64)
        if values.shape != mask.shape or not torch.isfinite(values[mask]).all() or (values[mask] < 0).any():
            raise ValueError("valid labels require finite nonnegative solver errors")
        reward = torch.zeros_like(values)
        log_error = torch.log1p(values[mask])
        reward[mask] = log_error / (1 + log_error)
        return reward, mask

    @torch.no_grad()
    def fit_calibration(self, clean_white, conditions, errors, valid, *, macro_context=None):
        if self.calibration is not None:
            raise ValueError("fit calibration is immutable; no silent refitting")
        values, _, _, context = self._inputs(clean_white, torch.ones(len(clean_white)), conditions, macro_context)
        rewards, mask = self._labels(errors, valid, values)
        if mask.sum() < 2:
            raise ValueError("at least two valid fit rewards required")
        mean, raw_std = rewards[mask].mean(), rewards[mask].std(unbiased=False)
        std = raw_std.clamp_min(REWARD_STD_FLOOR)
        log_z = torch.logsumexp(BETA_STD * (rewards[mask] - mean) / std, 0) - math.log(len(values))
        scale_records = {}
        for prefix, transform, inputs in (("field", self.transform, values),
                                         ("macro", self.macro_transform, context)):
            if inputs is None:
                continue
            sums = squares = None
            count = 0
            for chunk in inputs.split(4):
                field = transform.to_field(chunk).double()
                row_sum, row_square = field.sum((0, 2, 3, 4)), field.square().sum((0, 2, 3, 4))
                sums = row_sum if sums is None else sums + row_sum
                squares = row_square if squares is None else squares + row_square
                count += field.shape[0] * math.prod(field.shape[2:])
            center = sums / count
            raw_scale = (squares / count - center.square()).clamp_min(0).sqrt()
            scale = raw_scale.clamp_min(FIELD_STD_FLOOR)
            getattr(self, prefix + "_mean").copy_(center)
            getattr(self, prefix + "_scale").copy_(scale)
            scale_records[prefix] = dict(mean_sha256=_tensor_sha(getattr(self, prefix + "_mean")),
                scale_sha256=_tensor_sha(getattr(self, prefix + "_scale")),
                floored_channels=int((raw_scale < FIELD_STD_FLOOR).sum()))
        self.calibration = dict(fit_count=len(values), valid_fit_count=int(mask.sum()),
            invalid_fit_count=int((~mask).sum()), fit_reward_mean=float(mean),
            fit_reward_std=float(std), raw_fit_reward_std=float(raw_std),
            std_floor_applied=bool(raw_std < REWARD_STD_FLOOR), log_fit_mean_exp=float(log_z),
            beta_std=BETA_STD, fit_std_ddof=0, reward_std_floor=REWARD_STD_FLOOR,
            field_std_floor=FIELD_STD_FLOOR, input_calibration="fit_only_per_field_channel_mean_population_std",
            noisy_input_denominator="sqrt(fit_field_std_squared+sigma_squared*native_white_field_variance)",
            reward_transform="log1p(error)/(1+log1p(error))", error_scale=1.,
            normalization="all_fit_rows_including_invalid_zeros", scales=scale_records,
            fit_white_sha256=_tensor_sha(values), fit_conditions_sha256=_tensor_sha(torch.as_tensor(conditions)),
            fit_rewards_sha256=_tensor_sha(rewards), fit_valid_sha256=_tensor_sha(mask),
            fit_macro_context_sha256=None if context is None else _tensor_sha(context))
        return deepcopy(self.calibration)

    def targets(self, errors, valid, prototype):
        if self.calibration is None:
            raise ValueError("fit-only calibration required")
        reward, mask = self._labels(errors, valid, prototype)
        cal = self.calibration
        target = torch.zeros_like(reward)
        target[mask] = (BETA_STD * (reward[mask] - cal["fit_reward_mean"]) / cal["fit_reward_std"]
                        - cal["log_fit_mean_exp"]).exp()
        target = target.to(prototype)
        if not torch.isfinite(target).all() or (target[mask] <= 0).any():
            raise FloatingPointError("nonfinite/underflow desirability target; no clipping")
        return target.detach(), mask

    def forward(self, noisy, sigma, conditions, *, macro_context=None):
        if self.calibration is None:
            raise ValueError("fit-only calibration required")
        values, sigma, condition, context = self._inputs(noisy, sigma, conditions, macro_context)
        field = self.transform.to_field(values)
        denominator = (self.field_scale[None].square()
                       + sigma[:, None].square() * self.white_field_variance[None]).sqrt()
        field = ((field - self.field_mean[None, :, None, None, None])
                 / denominator[:, :, None, None, None])
        frequency = torch.exp(torch.linspace(0., math.log(100.), 8, device=values.device, dtype=values.dtype))
        phase = sigma.log()[:, None] * frequency[None]
        embedding = self.embedding(torch.cat((phase.sin(), phase.cos(), condition), 1))
        tau = torch.linspace(0., 1., self.config.control_count, device=values.device, dtype=values.dtype)
        position = self.time_position(torch.stack((tau, (math.pi * tau).sin(), (math.pi * tau).cos()), 1))
        hidden = self.input(field) + position.T[None, :, :, None, None]
        if context is not None:
            macro = (self.macro_transform.to_field(context) - self.macro_mean[None, :, None, None, None])
            hidden = hidden + self.macro_input(macro / self.macro_scale[None, :, None, None, None])
        for block in self.blocks:
            hidden = block(hidden, embedding)
        hidden = F.silu(hidden)
        pooled = torch.cat((hidden.mean((2, 3, 4)), hidden.square().mean((2, 3, 4))), 1)
        potential = self.head(pooled).squeeze(1)
        if not torch.isfinite(potential).all():
            raise FloatingPointError("nonfinite log desirability")
        return potential

    def loss(self, noisy, sigma, conditions, errors, valid, *, macro_context=None):
        # Labels/teacher coordinates are fixed data, never a trainable base graph.
        values = torch.as_tensor(noisy).detach()
        context = None if macro_context is None else torch.as_tensor(macro_context).detach()
        raw = self(values, sigma, conditions, macro_context=context)
        target, mask = self.targets(errors, valid, raw)
        prediction = raw.exp()
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
            raise FloatingPointError("nonfinite exponential regression; no clipping")
        return loss, dict(loss=float(loss.detach()), regression="MSE(exp(F),fixed_fit_target)",
            target_mean=float(target.mean()), target_max=float(target.max()),
            prediction_mean=float(prediction.detach().mean()), valid_count=int(mask.sum()),
            invalid_count=int((~mask).sum()), scientific_admission=False)

    def correct(self, denoised, noisy, sigma, conditions, *, macro_context=None,
                strength=None, random_seed=None):
        strength = self.strength if strength is None else strength
        if type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
            raise ValueError("finite nonnegative guidance strength required")
        if strength == 0:
            return denoised, dict(stage=self.stage, bypass="zero_strength", evaluated=False)
        values, sigma, _, context = self._inputs(noisy, sigma, conditions, macro_context)
        clean = torch.as_tensor(denoised, device=values.device, dtype=values.dtype).detach()
        if clean.shape != values.shape or not torch.isfinite(clean).all():
            raise ValueError("finite native denoised state of identical shape required")
        active = (sigma >= POLICY["sigma_min"]) & (sigma <= POLICY["sigma_max"])
        if not active.any():
            return denoised, dict(stage=self.stage, bypass="sigma_window", evaluated=False)
        with torch.enable_grad():
            query = values.detach().requires_grad_(True)
            potential = self(query, sigma, conditions,
                             macro_context=None if context is None else context.detach())
            gradient, = torch.autograd.grad(potential.sum(), query, create_graph=False)
        gradient = gradient.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite guidance gradient")
        grad_rms = gradient.square().mean(1).sqrt()
        drift_rms = ((values.detach() - clean) / sigma[:, None]).square().mean(1).sqrt()
        active &= (grad_rms >= POLICY["gradient_rms_floor"]) & (drift_rms >= POLICY["drift_rms_floor"])
        if random_seed is not None:
            if type(random_seed) is not int or random_seed < 0:
                raise ValueError("random direction requires a nonnegative integer stream seed")
            generator = torch.Generator(device=values.device).manual_seed(random_seed)
            direction = torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=generator)
            direction_rms = direction.square().mean(1).sqrt().clamp_min(POLICY["gradient_rms_floor"])
            gradient = direction * (grad_rms / direction_rms)[:, None]
        desired = min(strength * POLICY["target_ratio"], POLICY["cap_ratio"]) * drift_rms
        gain = (desired / (sigma * grad_rms).clamp_min(torch.finfo(values.dtype).tiny)).clamp_max(POLICY["gain_cap"])
        gain = torch.where(active, gain, torch.zeros_like(gain))
        shift = sigma[:, None].square() * gain[:, None] * gradient
        corrected = clean + shift
        actual_ratio = (shift / sigma[:, None]).square().mean(1).sqrt() / drift_rms.clamp_min(POLICY["drift_rms_floor"])
        if not torch.isfinite(corrected).all() or not torch.isfinite(actual_ratio).all():
            raise FloatingPointError("nonfinite guided denoised state")
        return corrected, dict(stage=self.stage, evaluated=True, strength=float(strength),
            random_seed=random_seed, active_count=int(active.sum()), sigma=sigma.detach().cpu().tolist(),
            gradient_rms=grad_rms.cpu().tolist(), base_drift_rms=drift_rms.cpu().tolist(),
            gain=gain.cpu().tolist(), relative_drift_rms=actual_ratio.cpu().tolist(),
            policy=deepcopy(POLICY), exact_tilted_sampling_claimed=False)

    def identity(self):
        return dict(schema=SCHEMA, stage=self.stage, width=self.width, depth=self.depth,
            base_class=self.base_class, base_identity=deepcopy(self.base_identity),
            base_config=self.config.to_dict(), macro_config=None if self.macro_config is None else self.macro_config.to_dict(),
            architecture="spatial_Fourier_3modes_nonperiodic_time_conv_global_mean_second_moment",
            beta_std=BETA_STD, policy=deepcopy(POLICY), decoder_changed=False,
            scientific_admission=False, exact_tilted_sampling_claimed=False)


def guidance_payload(head):
    if not isinstance(head, WeakSolverGuidanceHead) or head.calibration is None:
        raise ValueError("calibrated weak guidance head required")
    return dict(schema=SCHEMA, identity=head.identity(), calibration=deepcopy(head.calibration),
        state_dict={key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
        strength=float(head.strength))


def load_guidance_payload(payload, base_sampler):
    if not isinstance(payload, dict) or set(payload) != {"schema", "identity", "calibration", "state_dict", "strength"} or payload["schema"] != SCHEMA:
        raise ValueError("exact weak guidance payload schema required")
    identity = payload["identity"]
    # Loading must not advance the native CPU RNG stream.
    with torch.random.fork_rng(devices=[]):
        head = WeakSolverGuidanceHead(base_sampler, identity["stage"], identity["width"], identity["depth"])
    if _json(head.identity()) != _json(identity):
        raise ValueError("guidance identity differs from frozen native hierarchy")
    state = payload["state_dict"]
    expected = head.state_dict()
    if set(state) != set(expected):
        raise ValueError("guidance parameter/buffer inventory mismatch")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError("invalid guidance state tensor: " + name)
        if (name.startswith(("transform.", "macro_transform.")) or name == "white_field_variance") and not torch.equal(value.cpu(), expected[name].cpu()):
            raise ValueError("fixed native transform changed: " + name)
    calibration = deepcopy(payload["calibration"])
    _json(calibration)
    if (calibration.get("beta_std") != BETA_STD or calibration.get("fit_std_ddof") != 0
            or calibration.get("fit_reward_std", 0) < REWARD_STD_FLOOR
            or calibration.get("fit_count", 0) != calibration.get("valid_fit_count", 0) + calibration.get("invalid_fit_count", 0)
            or calibration.get("valid_fit_count", 0) < 2):
        raise ValueError("invalid fixed fit calibration")
    for prefix, record in calibration["scales"].items():
        for suffix in ("mean", "scale"):
            if record[suffix + "_sha256"] != _tensor_sha(state[prefix + "_" + suffix]):
                raise ValueError("input fit scale binding changed")
        if (state[prefix + "_scale"] < FIELD_STD_FLOOR).any():
            raise ValueError("invalid input fit scale")
    head.load_state_dict(state, strict=True)
    head.calibration = calibration
    strength = payload["strength"]
    if type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
        raise ValueError("invalid saved guidance strength")
    head.strength = float(strength)
    return head.eval()
