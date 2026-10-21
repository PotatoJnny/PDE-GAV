"""Versioned fit-only calibration of the unchanged continuous-error objective.

No per-sample normalization, new labels, base training, or physical admission.
The 00 control uses the original forward and loss with identical initialization.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from . import solver_error_guidance as legacy

SCHEMA = "vpfp_calibrated_solver_error_guidance_v1"
SCALE_NAMES = ("future_field_scale", "initial_field_scale", "pooled_scale")
FIT_ROOTS = frozenset(f"fit-{index:04d}" for index in range(64))
HEAD_WEIGHT_MULTIPLIER = .01
PROBABILITY_FLOOR = 1.e-6


def _recipe(scale_enabled, init_enabled):
    if type(scale_enabled) is not bool or type(init_enabled) is not bool:
        raise ValueError("explicit boolean calibration controls required")
    return {"scale_enabled": scale_enabled, "init_enabled": init_enabled,
        "scale_floor": 1., "head_weight_multiplier": HEAD_WEIGHT_MULTIPLIER,
        "reward_probability_floor": PROBABILITY_FLOOR,
        "validity_prior": "Jeffreys_(valid_count+0.5)/(fit_count+1)",
        "fit_root_count": 64, "scale_reduction": "fixed_fit_RMS_no_centering",
        "reward_objective": "unchanged_sigmoid_MSE", "validity_objective": "unchanged_weighted_BCE"}


def _state_sha(state):
    digest = sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class CalibratedSolverErrorRegressor(legacy.SolverErrorRegressor):
    def __init__(self, config, *, scale_enabled, init_enabled):
        _recipe(scale_enabled, init_enabled)
        super().__init__(config)
        self.scale_enabled, self.init_enabled = scale_enabled, init_enabled
        self.calibration_metadata = None
        # Constant buffers neither draw RNG nor alter the original parameters.
        self.register_buffer("future_field_scale", torch.ones(config.base_config.field_channels))
        self.register_buffer("initial_field_scale", torch.ones(config.base_config.field_channels))
        self.register_buffer("pooled_scale", torch.ones(2 * config.width))

    def forward(self, noisy_future_white, sigma, conditions, *, initial_context):
        if not self.scale_enabled:
            return super().forward(noisy_future_white, sigma, conditions, initial_context=initial_context)
        cfg, parameter = self.config.base_config, next(self.parameters())
        noisy = legacy._coefficients(noisy_future_white, cfg, device=parameter.device, dtype=parameter.dtype)
        if not len(noisy):
            raise ValueError("guidance needs a nonempty batch")
        context = legacy._context(initial_context, cfg, prototype=noisy)
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        if (sigma.shape != (len(noisy),) or len(context) != len(noisy)
                or not torch.isfinite(sigma).all() or torch.any(sigma <= 0)):
            raise ValueError("positive sigma and complete clean IC for every future state required")
        condition = legacy.VPFPStructuredEDM._conditions(SimpleNamespace(config=cfg), conditions, len(noisy), noisy)
        field = self.transform.to_field(noisy / (sigma.square()+cfg.sigma_data**2).sqrt()[:, None])
        frequency = torch.exp(torch.linspace(0., math.log(100.), 8, dtype=noisy.dtype, device=noisy.device))
        phase = sigma.log()[:, None]*frequency[None]
        embedding = self.embedding(torch.cat((phase.sin(), phase.cos(), condition), 1))
        tau = torch.linspace(0., 1., cfg.full_config.control_count, dtype=noisy.dtype, device=noisy.device)[1:]
        position = self.time_position(torch.stack((tau, (math.pi*tau).sin(), (math.pi*tau).cos()), 1))
        field = field / self.future_field_scale[None, :, None, None, None]
        initial_field = self._initial_field(context) / self.initial_field_scale[None, :, None, None]
        hidden = self.input(field)+self.initial_input(initial_field)[:, :, None]
        hidden = hidden+position.T[None, :, :, None, None]
        for block in self.blocks:
            hidden = block(hidden, embedding)
        hidden = F.silu(hidden)
        pooled = torch.cat((hidden.mean((2, 3, 4)), hidden.square().mean((2, 3, 4))), 1)
        logits = self.head(pooled / self.pooled_scale[None])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite solver-error prediction")
        return {"reward": logits[:, 0].sigmoid(), "validity_logit": logits[:, 1],
            "valid_probability": logits[:, 1].sigmoid()}

    def runtime_settings(self):
        return {"schema": SCHEMA, **_recipe(self.scale_enabled, self.init_enabled),
            "calibration_metadata_hash": (None if self.calibration_metadata is None
                                           else legacy._json_sha(self.calibration_metadata))}

    def architecture_report(self):
        return {**super().architecture_report(), "schema": SCHEMA,
            "recipe": _recipe(self.scale_enabled, self.init_enabled),
            "calibration": "64_fit_roots_only_fixed_channel_and_pooled_feature_scales",
            "amplitude_information": "preserved_by_fixed_positive_divisors_not_per_sample_normalization",
            "scientific_admission": False}


def make_calibrated_regressor(config, *, parameter_dtype, device="cpu", scale_enabled, init_enabled):
    if parameter_dtype not in (torch.float32, torch.float64):
        raise ValueError("guidance parameters require explicit float32 or float64")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter_dtype)
        model = CalibratedSolverErrorRegressor(config, scale_enabled=scale_enabled, init_enabled=init_enabled)
    finally:
        torch.set_default_dtype(previous)
    return model.to(device=device)


def _fit_rows(model, rows):
    if (not isinstance(rows, dict) or rows.get("split", "fit") != "fit"
            or rows.get("excluded_rows") or "validation" in rows):
        raise ValueError("calibration accepts complete fit rows only")
    roots = list(rows["root_ids"])
    if len(roots) != 64 or set(roots) != FIT_ROOTS:
        raise ValueError("calibration requires exactly the 64 unique registered fit roots")
    parameter, cfg = next(model.parameters()), model.config.base_config
    future = legacy._coefficients(rows["future_white"], cfg, device=parameter.device, dtype=parameter.dtype).detach()
    context = legacy._context(rows["initial_context"], cfg, prototype=future).detach()
    conditions = torch.as_tensor(rows["conditions"], device=parameter.device, dtype=parameter.dtype).detach()
    if future.shape[0] != 64 or context.shape[0] != 64 or conditions.shape != (64, 2):
        raise ValueError("one complete fit input and condition per physical root required")
    legacy.VPFPStructuredEDM._conditions(SimpleNamespace(config=cfg), conditions, 64, future)
    target, valid = legacy.bounded_error_targets(rows["errors"], rows["valid"],
        prototype=future, error_scale=model.config.error_scale)
    if target.shape != (64,) or not bool(valid.any()):
        raise ValueError("calibration requires fit reward labels and at least one valid fit root")
    return future, context, conditions, target, valid, roots


def _batches(future, context, conditions, cfg, *, seed, batch_size):
    # Each traversal deliberately restarts this local stream. Neither ambient
    # CPU/CUDA state nor the later optimizer/noising generator is consumed.
    generator = torch.Generator(device=future.device).manual_seed(seed)
    for start in range(0, len(future), batch_size):
        clean = future[start:start+batch_size]
        sigma = (torch.randn(len(clean), dtype=clean.dtype, device=clean.device, generator=generator)
            * cfg.log_sigma_std + cfg.log_sigma_mean).exp().clamp(cfg.sigma_min, cfg.sigma_max)
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        yield clean + sigma[:, None] * noise, sigma, context[start:start+batch_size], conditions[start:start+batch_size]


def _accumulate(total, count, value):
    values = value.detach().double()
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite fit calibration activation")
    axes = tuple(axis for axis in range(values.ndim) if axis != 1)
    return total + values.square().sum(axes), count + values.numel() // values.shape[1]


def _set_scale(buffer, total, count):
    scale = (total / count).sqrt().clamp_min(1.).to(buffer)
    if not torch.isfinite(scale).all():
        raise ValueError("nonfinite fitted calibration scale")
    buffer.copy_(scale)


@torch.no_grad()
def fit_calibration(model, rows, *, seed, batch_size=4):
    """Fit once on the fixed 64 training roots; return detached JSON metadata."""
    if not isinstance(model, CalibratedSolverErrorRegressor) or model.calibration_metadata is not None:
        raise ValueError("calibration requires a fresh versioned model; no silent refitting")
    if type(seed) is not int or not 0 <= seed < 2**63 or type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("explicit independent calibration seed and bounded batch size required")
    future, context, conditions, target, valid, roots = _fit_rows(model, rows)
    scales_before = {name: getattr(model, name).clone() for name in SCALE_NAMES}
    if any(not torch.equal(value, torch.ones_like(value)) for value in scales_before.values()):
        raise ValueError("fresh calibration buffers must all be one")
    cfg = model.config.base_config
    try:
        if model.scale_enabled:
            fs = torch.zeros_like(model.future_field_scale, dtype=torch.float64)
            initial_sum = torch.zeros_like(model.initial_field_scale, dtype=torch.float64)
            fc = initial_count = 0
            for noisy, sigma, initial, _ in _batches(future, context, conditions, cfg, seed=seed, batch_size=batch_size):
                field = model.transform.to_field(noisy / (sigma.square()+cfg.sigma_data**2).sqrt()[:, None])
                fs, fc = _accumulate(fs, fc, field)
                initial_sum, initial_count = _accumulate(initial_sum, initial_count, model._initial_field(initial))
            _set_scale(model.future_field_scale, fs, fc)
            _set_scale(model.initial_field_scale, initial_sum, initial_count)
            pooled_sum = torch.zeros_like(model.pooled_scale, dtype=torch.float64)
            pooled_count = 0
            def observe(_module, inputs):
                nonlocal pooled_sum, pooled_count
                pooled_sum, pooled_count = _accumulate(pooled_sum, pooled_count, inputs[0])
            handle = model.head.register_forward_pre_hook(observe)
            try:
                for noisy, sigma, initial, theta in _batches(future, context, conditions, cfg, seed=seed, batch_size=batch_size):
                    model(noisy, sigma, theta, initial_context=initial)
            finally:
                handle.remove()
            _set_scale(model.pooled_scale, pooled_sum, pooled_count)
        reward_mean = float(target[valid].double().mean())
        reward_probability = min(1. - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, reward_mean))
        validity_probability = (int(valid.sum()) + .5) / 65.
        if model.init_enabled:
            model.head[-1].weight.mul_(HEAD_WEIGHT_MULTIPLIER)
            model.head[-1].bias.copy_(model.head[-1].bias.new_tensor([
                math.log(reward_probability / (1. - reward_probability)),
                math.log(validity_probability / (1. - validity_probability))]))
        model.calibration_metadata = {
            "schema": SCHEMA + "_calibration", "split": "fit", "fit_root_ids": roots,
            "fit_root_count": 64, "seed": seed, "batch_size": batch_size,
            "noising_law": "same_clipped_log_normal_sigma_and_standard_normal_as_EDM_loss",
            "noise_stream_replayed_for_pooled_fit": bool(model.scale_enabled),
            "noise_replicates_per_root": 1 if model.scale_enabled else 0,
            "ambient_rng_consumed": False, "per_sample_normalization": False,
            "valid_fit_count": int(valid.sum()), "fit_reward_mean": reward_mean,
            "reward_bias_probability": reward_probability, "validity_bias_probability": validity_probability,
            "scale_state_sha256": _state_sha({name: getattr(model, name) for name in SCALE_NAMES}),
            "recipe": _recipe(model.scale_enabled, model.init_enabled)}
    except BaseException:
        for name, value in scales_before.items():
            getattr(model, name).copy_(value)
        raise
    return deepcopy(model.calibration_metadata)


def _validate_calibration(model, metadata, roots, label_sha):
    required = {"schema", "split", "fit_root_ids", "fit_root_count", "seed", "batch_size", "noising_law",
        "noise_stream_replayed_for_pooled_fit", "noise_replicates_per_root", "ambient_rng_consumed",
        "per_sample_normalization", "valid_fit_count", "fit_reward_mean", "reward_bias_probability",
        "validity_bias_probability", "scale_state_sha256", "recipe", "label_manifest_sha256"}
    if (not isinstance(metadata, dict) or set(metadata) != required
            or metadata["schema"] != SCHEMA + "_calibration" or metadata["split"] != "fit"
            or metadata["label_manifest_sha256"] != label_sha
            or metadata["fit_root_count"] != 64 or len(metadata["fit_root_ids"]) != 64
            or set(metadata["fit_root_ids"]) != FIT_ROOTS
            or set(metadata["fit_root_ids"]) != set(roots["train_root_ids"])
            or len(roots["train_root_ids"]) != 64
            or metadata["recipe"] != _recipe(model.scale_enabled, model.init_enabled)
            or metadata["ambient_rng_consumed"] is not False or metadata["per_sample_normalization"] is not False
            or metadata["noising_law"] != "same_clipped_log_normal_sigma_and_standard_normal_as_EDM_loss"
            or metadata["noise_stream_replayed_for_pooled_fit"] is not model.scale_enabled
            or metadata["noise_replicates_per_root"] != int(model.scale_enabled)
            or type(metadata["seed"]) is not int or not 0 <= metadata["seed"] < 2**63
            or type(metadata["batch_size"]) is not int or not 1 <= metadata["batch_size"] <= 64
            or type(metadata["valid_fit_count"]) is not int or not 1 <= metadata["valid_fit_count"] <= 64):
        raise ValueError("calibration fit provenance or controls differ")
    reward_mean = metadata["fit_reward_mean"]
    if type(reward_mean) not in (float, int) or not math.isfinite(reward_mean) or not 0 <= reward_mean <= 1:
        raise ValueError("invalid fit reward mean")
    if (metadata["reward_bias_probability"] != min(1.-PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, reward_mean))
            or metadata["validity_bias_probability"] != (metadata["valid_fit_count"] + .5) / 65.):
        raise ValueError("calibration label-derived initialization differs")
    scales = {name: getattr(model, name) for name in SCALE_NAMES}
    if any(not torch.isfinite(value).all() or torch.any(value < 1.) for value in scales.values()):
        raise ValueError("fixed calibration scales must be finite and at least one")
    if not model.scale_enabled and any(not torch.equal(value, torch.ones_like(value)) for value in scales.values()):
        raise ValueError("disabled calibration scales must remain exactly one")


def guidance_payload(model, *, label_manifest_sha256, train_root_ids, validation_root_ids, training):
    if not isinstance(model, CalibratedSolverErrorRegressor) or model.calibration_metadata is None:
        raise ValueError("versioned payload requires completed fit-only calibration")
    legacy._sha(label_manifest_sha256, "actual solver label manifest")
    roots = legacy.validate_root_partition(train_root_ids, validation_root_ids)
    metadata = json.loads(json.dumps(training, allow_nan=False))
    if (metadata.get("label_source") not in ("actual_public_solver_macro_error", "synthetic_fixture")
            or type(metadata.get("step")) is not int or metadata["step"] < 0):
        raise ValueError("explicit label source and nonnegative training step required")
    calibration = {**deepcopy(model.calibration_metadata), "label_manifest_sha256": label_manifest_sha256}
    _validate_calibration(model, calibration, roots, label_manifest_sha256)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {"schema": SCHEMA, "config": model.config.to_dict(), "model": state,
        "label_manifest_sha256": label_manifest_sha256, "root_partition": roots,
        "training": metadata, "architecture": model.architecture_report(),
        "recipe": _recipe(model.scale_enabled, model.init_enabled), "calibration": calibration,
        "state_sha256": _state_sha(state)}
    payload["metadata_sha256"] = legacy._json_sha({key: value for key, value in payload.items() if key != "model"})
    return payload


def regressor_from_payload(payload, *, device="cpu"):
    fields = {"schema", "config", "model", "label_manifest_sha256", "root_partition", "training",
              "architecture", "recipe", "calibration", "state_sha256", "metadata_sha256"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != SCHEMA:
        raise ValueError("unexpected calibrated guidance checkpoint fields")
    legacy._sha(payload["label_manifest_sha256"], "actual solver label manifest")
    recipe = payload["recipe"]
    if not isinstance(recipe, dict) or recipe != _recipe(recipe.get("scale_enabled"), recipe.get("init_enabled")):
        raise ValueError("unknown calibrated guidance recipe")
    roots = payload["root_partition"]
    if roots != legacy.validate_root_partition(roots["train_root_ids"], roots["validation_root_ids"]):
        raise ValueError("calibrated guidance root partition differs")
    training = payload["training"]
    if (training.get("label_source") not in ("actual_public_solver_macro_error", "synthetic_fixture")
            or type(training.get("step")) is not int or training["step"] < 0):
        raise ValueError("invalid calibrated guidance training provenance")
    state = payload["model"]
    parameter_dtype = legacy._validated_guide_state_dtype(state)
    with torch.random.fork_rng(devices=[]):
        model = make_calibrated_regressor(legacy.SolverErrorGuidanceConfig.from_dict(payload["config"]),
            parameter_dtype=parameter_dtype, scale_enabled=recipe["scale_enabled"], init_enabled=recipe["init_enabled"])
    if json.dumps(payload["architecture"], sort_keys=True) != json.dumps(model.architecture_report(), sort_keys=True):
        raise ValueError("calibrated guidance architecture differs")
    legacy._validate_guide_state(model, state)
    model.load_state_dict(state, strict=True)
    _validate_calibration(model, payload["calibration"], roots, payload["label_manifest_sha256"])
    model.calibration_metadata = deepcopy(payload["calibration"])
    del model.calibration_metadata["label_manifest_sha256"]
    return model.to(device).eval().requires_grad_(False)
