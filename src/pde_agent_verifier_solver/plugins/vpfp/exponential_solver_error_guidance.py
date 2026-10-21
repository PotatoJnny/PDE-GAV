"""Fit-only, hard-valid joint exponential desirability for frozen VPFP EDM.

The regression estimand is E[I_valid exp(beta*(r-b)) | noisy, IC, nu, T].
The raw first output is its log, not a sigmoid reward or a log-target MSE.
Clipped, windowed finite-step Heun sampling is not claimed to be an exact tilt.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from . import calibrated_solver_error_guidance as calibrated
from . import solver_error_guidance as legacy

SCHEMA = "vpfp_exponential_solver_error_guidance_v1"
OBJECTIVE = "exponential_reward_log_h_v1"
BETA = 2.0


def _objective(beta):
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta != BETA:
        raise ValueError("this registered exponential objective fixes beta=2.0")
    return {"objective": OBJECTIVE, "beta": float(beta), "scale_enabled": True, "init_enabled": True,
        "estimand": "E[I_valid*exp(beta*(bounded_reward-fit_valid_reward_mean))|noisy,IC,nu,T]",
        "regression": "mean((exp(raw_log_h)-hard_valid_exponential_target)^2)",
        "auxiliary_validity": "unchanged_weighted_BCE_diagnostic_only_not_multiplied_into_score",
        "score_potential": "raw_log_desirability", "expected_sampling_strength": 1.0,
        "beta_applied_in_target_only": True, "per_sample_normalization": False}


def exponential_targets(macro_errors, numerically_valid, *, prototype, reward_center, beta=BETA, error_scale=1.):
    """Return joint positive/zero targets; invalid NaN errors never enter exp."""
    _objective(beta)
    if (type(reward_center) not in (int, float) or not math.isfinite(reward_center)
            or not 0. <= reward_center <= 1.):
        raise ValueError("fixed fit-only bounded reward center required")
    rewards, valid = legacy.bounded_error_targets(macro_errors, numerically_valid,
        prototype=prototype, error_scale=error_scale)
    weights = torch.zeros_like(rewards)
    # Select before nonlinear arithmetic: an unavailable error is not a reward.
    weights[valid] = (float(beta) * (rewards[valid].double() - float(reward_center))).exp().to(weights)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("nonfinite exponential solver-error target")
    return weights.detach(), valid


def desirability_mse(log_desirability, weights):
    """Positive regression: optimum exp(F)=E[w], not F=E[log(w)]."""
    if (log_desirability.ndim != 1 or weights.shape != log_desirability.shape
            or not log_desirability.numel() or not torch.isfinite(log_desirability).all()
            or not torch.isfinite(weights).all() or torch.any(weights < 0)):
        raise ValueError("finite one-dimensional log desirability and nonnegative targets required")
    value = (log_desirability.exp() - weights.detach()).square().mean()
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite exponential desirability regression; no silent clipping")
    return value


class ExponentialSolverErrorRegressor(calibrated.CalibratedSolverErrorRegressor):
    expected_sampling_strength = 1.0

    def __init__(self, config, *, beta=BETA):
        _objective(beta)
        super().__init__(config, scale_enabled=True, init_enabled=True)
        self.beta = float(beta)
        self.reward_center = None
        self.exponential_metadata = None

    def forward(self, noisy_future_white, sigma, conditions, *, initial_context):
        if self.scale_enabled is not True or self.init_enabled is not True:
            raise ValueError("exponential guidance requires the fixed combined calibration")
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
            raise FloatingPointError("nonfinite exponential guidance output")
        desirability = logits[:, 0].exp()
        if not torch.isfinite(desirability).all():
            raise FloatingPointError("exponential desirability overflow; no silent clipping")
        return {"log_desirability": logits[:, 0], "desirability": desirability,
            "validity_logit": logits[:, 1], "valid_probability": logits[:, 1].sigmoid()}

    def score_potential(self, prediction):
        if self.expected_sampling_strength != 1.0:
            raise ValueError("beta is already in the target; external sampling strength must be one")
        value = prediction["log_desirability"]
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError("score potential requires a finite raw log-desirability vector")
        return value  # No sigmoid, log(exp(.)), auxiliary validity factor, or second beta.

    def loss(self, clean_future_white, conditions, initial_context, macro_errors, numerically_valid,
             *, generator=None, sigma=None, noise=None):
        if self.exponential_metadata is None or self.reward_center is None:
            raise ValueError("fit-only exponential calibration must precede training")
        cfg, parameter = self.config.base_config, next(self.parameters())
        clean = legacy._coefficients(clean_future_white, cfg, device=parameter.device, dtype=parameter.dtype).detach()
        context = legacy._context(initial_context, cfg, prototype=clean).detach()
        if sigma is None:
            sigma = (torch.randn(len(clean), dtype=clean.dtype, device=clean.device, generator=generator)
                * cfg.log_sigma_std+cfg.log_sigma_mean).exp().clamp(cfg.sigma_min, cfg.sigma_max)
        sigma = torch.as_tensor(sigma, dtype=clean.dtype, device=clean.device)
        if (sigma.shape != (len(clean),) or not torch.isfinite(sigma).all()
                or torch.any(sigma < cfg.sigma_min) or torch.any(sigma > cfg.sigma_max)):
            raise ValueError("guide training sigma leaves the frozen EDM interval")
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator) if noise is None else legacy._coefficients(
            noise, cfg, device=clean.device, dtype=clean.dtype)
        if noise.shape != clean.shape:
            raise ValueError("guidance noise batch differs")
        targets, valid = exponential_targets(macro_errors, numerically_valid, prototype=clean,
            reward_center=self.reward_center, beta=self.beta, error_scale=self.config.error_scale)
        if len(targets) != len(clean):
            raise ValueError("solver label and state counts differ")
        prediction = self(clean+sigma[:, None]*noise.detach(), sigma, conditions, initial_context=context)
        regression = desirability_mse(prediction["log_desirability"], targets)
        validity_loss = F.binary_cross_entropy_with_logits(prediction["validity_logit"], valid.to(clean))
        loss = regression + self.config.validity_weight * validity_loss
        return loss, {"desirability_loss": regression.detach(), "validity_loss": validity_loss.detach(),
            "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()), "sigma": sigma.detach(),
            "desirability_target": targets.detach(), "predicted_desirability": prediction["desirability"].detach(),
            "log_desirability": prediction["log_desirability"].detach()}

    def runtime_settings(self):
        return {"schema": SCHEMA, **_objective(self.beta), "reward_center": self.reward_center,
            "scale_enabled": self.scale_enabled, "init_enabled": self.init_enabled,
            "expected_sampling_strength": self.expected_sampling_strength,
            "calibration_metadata_hash": (None if self.calibration_metadata is None
                                           else legacy._json_sha(self.calibration_metadata)),
            "exponential_metadata_hash": (None if self.exponential_metadata is None
                                           else legacy._json_sha(self.exponential_metadata))}

    def architecture_report(self):
        return {**super().architecture_report(), "schema": SCHEMA, "recipe": _objective(self.beta),
            "reward_target": "hard_valid_joint_exponential_desirability_not_conditional_mean_reward",
            "invalid_labels": "zero_desirability_targets_and_separate_auxiliary_validity_BCE",
            "output": "raw_log_h_and_auxiliary_validity_logit", "scientific_admission": False,
            "exact_tilted_sampling_claimed": False}


def make_exponential_regressor(config, *, parameter_dtype, device="cpu", beta=BETA):
    if parameter_dtype not in (torch.float32, torch.float64):
        raise ValueError("guidance parameters require explicit float32 or float64")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter_dtype)
        model = ExponentialSolverErrorRegressor(config, beta=beta)
    finally:
        torch.set_default_dtype(previous)
    return model.to(device=device)


@torch.no_grad()
def fit_exponential_calibration(model, rows, *, seed, batch_size=4):
    if (not isinstance(model, ExponentialSolverErrorRegressor) or model.calibration_metadata is not None
            or model.exponential_metadata is not None or model.reward_center is not None):
        raise ValueError("fresh exponential model required; no silent refitting")
    _objective(model.beta)
    if model.scale_enabled is not True or model.init_enabled is not True:
        raise ValueError("the registered exponential model fixes combined calibration")
    parameter = next(model.parameters())
    # Fit through the unchanged v2 forward, whose outputs are ignored during
    # pooled-scale collection. This avoids exponentiating an uncalibrated head.
    # The same-state temporary guide has no optimizer and consumes no ambient RNG.
    with torch.random.fork_rng(devices=[]):
        view = calibrated.make_calibrated_regressor(model.config, parameter_dtype=parameter.dtype,
            device=parameter.device, scale_enabled=True, init_enabled=True)
    view.load_state_dict(model.state_dict(), strict=True)
    calibration = calibrated.fit_calibration(view, rows, seed=seed, batch_size=batch_size)
    center = calibration["fit_reward_mean"]  # Mean over defined, valid fit rewards only.
    weights, valid = exponential_targets(rows["errors"], rows["valid"], prototype=parameter,
        reward_center=center, beta=model.beta, error_scale=model.config.error_scale)
    target_mean = float(weights.double().mean())  # Includes invalid roots' exact zero targets.
    if not math.isfinite(target_mean) or target_mean <= 0.:
        raise ValueError("fit desirability mean must be finite and positive")
    view.head[-1].bias[0] = math.log(target_mean)
    model.load_state_dict(view.state_dict(), strict=True)
    model.calibration_metadata = deepcopy(calibration)
    model.reward_center = float(center)
    model.exponential_metadata = {"schema": SCHEMA + "_calibration", "beta": model.beta,
        "reward_center": model.reward_center, "centering": "mean_defined_valid_fit_bounded_rewards",
        "fit_desirability_mean": target_mean, "initial_log_desirability_bias": math.log(target_mean),
        "fit_root_ids": list(rows["root_ids"]), "fit_root_count": 64, "valid_fit_count": int(valid.sum()),
        "target_mean_includes_invalid_zeros": True, "external_sampling_strength": 1.0,
        "scale_calibration_metadata_sha256": legacy._json_sha(calibration)}
    return deepcopy(model.exponential_metadata)


def _validate_exponential(model, metadata, calibration, roots, label_sha):
    fields = {"schema", "beta", "reward_center", "centering", "fit_desirability_mean",
        "initial_log_desirability_bias", "fit_root_ids", "fit_root_count", "valid_fit_count",
        "target_mean_includes_invalid_zeros", "external_sampling_strength",
        "scale_calibration_metadata_sha256", "label_manifest_sha256"}
    if (not isinstance(metadata, dict) or set(metadata) != fields or metadata["schema"] != SCHEMA + "_calibration"
            or metadata["label_manifest_sha256"] != label_sha or metadata["beta"] != model.beta
            or metadata["reward_center"] != calibration["fit_reward_mean"]
            or metadata["centering"] != "mean_defined_valid_fit_bounded_rewards"
            or metadata["fit_root_ids"] != calibration["fit_root_ids"]
            or set(metadata["fit_root_ids"]) != set(roots["train_root_ids"])
            or metadata["fit_root_count"] != 64 or metadata["valid_fit_count"] != calibration["valid_fit_count"]
            or metadata["target_mean_includes_invalid_zeros"] is not True
            or type(metadata["external_sampling_strength"]) not in (int, float)
            or metadata["external_sampling_strength"] != 1.0):
        raise ValueError("exponential fit provenance, center, or beta differs")
    without_label = {key: value for key, value in calibration.items() if key != "label_manifest_sha256"}
    mean = metadata["fit_desirability_mean"]
    center = metadata["reward_center"]
    if (type(mean) not in (int, float) or not math.isfinite(mean) or mean <= 0.
            or mean > math.exp(model.beta * (1. - center)) * metadata["valid_fit_count"] / 64. + 1.e-6
            or mean < math.exp(-model.beta * center) * metadata["valid_fit_count"] / 64. - 1.e-6
            or metadata["initial_log_desirability_bias"] != math.log(mean)):
        raise ValueError("invalid fit joint desirability mean or initialization")


def exponential_payload(model, *, label_manifest_sha256, train_root_ids, validation_root_ids, training):
    if (not isinstance(model, ExponentialSolverErrorRegressor) or model.exponential_metadata is None
            or model.reward_center is None or model.scale_enabled is not True or model.init_enabled is not True):
        raise ValueError("payload requires completed combined exponential calibration")
    _objective(model.beta)
    payload = calibrated.guidance_payload(model, label_manifest_sha256=label_manifest_sha256,
        train_root_ids=train_root_ids, validation_root_ids=validation_root_ids, training=training)
    exponential = {**deepcopy(model.exponential_metadata), "label_manifest_sha256": label_manifest_sha256}
    _validate_exponential(model, exponential, payload["calibration"], payload["root_partition"], label_manifest_sha256)
    if model.reward_center != exponential["reward_center"] or model.expected_sampling_strength != 1.0:
        raise ValueError("exponential runtime settings changed")
    payload.update(schema=SCHEMA, recipe=_objective(model.beta), exponential=exponential)
    payload["metadata_sha256"] = legacy._json_sha({key: value for key, value in payload.items()
                                                  if key not in ("model", "metadata_sha256")})
    return payload


def regressor_from_payload(payload, *, device="cpu"):
    fields = {"schema", "config", "model", "label_manifest_sha256", "root_partition", "training",
        "architecture", "recipe", "calibration", "exponential", "state_sha256", "metadata_sha256"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != SCHEMA:
        raise ValueError("unexpected exponential guidance checkpoint fields")
    legacy._sha(payload["label_manifest_sha256"], "actual solver label manifest")
    recipe = payload["recipe"]
    if not isinstance(recipe, dict) or recipe != _objective(recipe.get("beta")):
        raise ValueError("unknown exponential guidance objective or recipe")
    roots = payload["root_partition"]
    if roots != legacy.validate_root_partition(roots["train_root_ids"], roots["validation_root_ids"]):
        raise ValueError("exponential guidance root partition differs")
    training = payload["training"]
    if (training.get("label_source") not in ("actual_public_solver_macro_error", "synthetic_fixture")
            or type(training.get("step")) is not int or training["step"] < 0):
        raise ValueError("invalid exponential guidance training provenance")
    state = payload["model"]
    dtype = legacy._validated_guide_state_dtype(state)
    with torch.random.fork_rng(devices=[]):
        model = make_exponential_regressor(legacy.SolverErrorGuidanceConfig.from_dict(payload["config"]),
            parameter_dtype=dtype, beta=recipe["beta"])
    if json.dumps(payload["architecture"], sort_keys=True) != json.dumps(model.architecture_report(), sort_keys=True):
        raise ValueError("exponential guidance architecture differs")
    legacy._validate_guide_state(model, state)
    model.load_state_dict(state, strict=True)
    calibrated._validate_calibration(model, payload["calibration"], roots, payload["label_manifest_sha256"])
    _validate_exponential(model, payload["exponential"], payload["calibration"], roots, payload["label_manifest_sha256"])
    model.calibration_metadata = deepcopy(payload["calibration"])
    del model.calibration_metadata["label_manifest_sha256"]
    model.exponential_metadata = deepcopy(payload["exponential"])
    del model.exponential_metadata["label_manifest_sha256"]
    model.reward_center = float(model.exponential_metadata["reward_center"])
    return model.to(device).eval().requires_grad_(False)
