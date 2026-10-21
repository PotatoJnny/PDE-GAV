"""Fit-only standardized VPFP reward guidance, with separate sampling policy.

This version preserves the historical log1p-bounded solver error. Exponential
regression learns the conditional mean of fixed-normalized positive weights;
the mean-reward control learns the standardized valid-conditional reward.
Neither the auxiliary validity head nor an output sigmoid enters the score.
"""
from __future__ import annotations
import json

from copy import deepcopy
import math
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from . import calibrated_solver_error_guidance as calibrated
from . import exponential_solver_error_guidance as exponential
from . import solver_error_guidance as legacy

SCHEMA = "vpfp_standardized_solver_error_guidance_v1"
OBJECTIVE = "standardized_exponential_reward_log_h_v1"
MEAN_OBJECTIVE = "standardized_mean_reward"
REWARD_TRANSFORM = "log1p(error/error_scale)/(1+log1p(error/error_scale))"


def _recipe(beta_std, objective):
    if (type(beta_std) not in (int, float) or not math.isfinite(beta_std)
            or beta_std not in (.5, 1.)):
        raise ValueError("registered standardized beta_std must be 0.5 or 1.0")
    if objective not in (OBJECTIVE, MEAN_OBJECTIVE):
        raise ValueError("unknown standardized guidance objective")
    return {"objective": objective, "beta_std": float(beta_std),
        "reward_transform": REWARD_TRANSFORM, "fit_root_count": 64,
        "fit_std_ddof": 0, "scale_enabled": True, "init_enabled": True,
        "first_head_initial_bias": 0., "auxiliary_validity_weight": .1,
        "regression": ("MSE(exp(raw_F),I_valid*exp(beta_std*standardized_reward-log_fit_mean_exp))"
                       if objective == OBJECTIVE else "valid_only_MSE(raw_F,standardized_reward)"),
        "score_potential": "raw_F" if objective == OBJECTIVE else "beta_std*raw_F",
        "invalid_labels": "joint_zero_desirability" if objective == OBJECTIVE else "auxiliary_validity_only",
        "auxiliary_validity_enters_score": False, "per_sample_normalization": False,
        "normalization": "fixed_all_64_fit_mean_exp_including_invalid_zeros_not_batch_normalization",
        "expected_sampling_strength": None,
        "sampling_policy": "external_lambda_separate_from_training_beta_not_an_exact_tilt_claim"}


def _normalization(fit_reward_mean, fit_reward_std, log_fit_mean_exp):
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (fit_reward_mean, fit_reward_std, log_fit_mean_exp))
            or not 0. <= fit_reward_mean <= 1. or not 0. < fit_reward_std <= 1.):
        raise ValueError("finite nondegenerate fixed fit reward mean/std/log normalization required")


def standardized_targets(macro_errors, numerically_valid, *, prototype,
                         fit_reward_mean, fit_reward_std, log_fit_mean_exp,
                         beta_std, objective=OBJECTIVE, error_scale=1.):
    """No fit/refit or batch normalization; unavailable labels never enter exp."""
    _recipe(beta_std, objective)
    _normalization(fit_reward_mean, fit_reward_std, log_fit_mean_exp)
    rewards, valid = legacy.bounded_error_targets(macro_errors, numerically_valid,
        prototype=prototype.double(), error_scale=error_scale)
    targets = torch.zeros_like(rewards)
    standardized = (rewards[valid] - fit_reward_mean) / fit_reward_std
    if not torch.isfinite(standardized).all():
        raise FloatingPointError("nonfinite standardized reward; no silent clipping")
    targets[valid] = ((beta_std * standardized - log_fit_mean_exp).exp()
                      if objective == OBJECTIVE else standardized)
    targets = targets.to(prototype)
    if not torch.isfinite(targets).all():
        raise FloatingPointError("nonfinite standardized guidance target; no silent clipping")
    if objective == OBJECTIVE and torch.any(targets[valid] <= 0.):
        raise FloatingPointError("valid desirability target underflow; no silent clipping")
    return targets.detach(), valid


def _statistics(records, beta_std):
    """Recomputable float64 scalar metadata; invalid roots remain in denominator."""
    if (not isinstance(records, list) or len(records) != 64
            or any(not isinstance(row, dict) or set(row) != {"root_id", "valid", "reward"} for row in records)
            or any(type(row["valid"]) is not bool or not isinstance(row["root_id"], str) for row in records)
            or {row["root_id"] for row in records} != calibrated.FIT_ROOTS):
        raise ValueError("exactly 64 unique registered fit label records required")
    values = []
    for row in records:
        if not row["valid"]:
            if row["reward"] is not None:
                raise ValueError("invalid fit roots must have unavailable rewards")
        elif (type(row["reward"]) not in (int, float) or not math.isfinite(row["reward"])
                or not 0. <= row["reward"] <= 1.):
            raise ValueError("finite bounded valid fit rewards required")
        else:
            values.append(float(row["reward"]))
    if len(values) < 2 or min(values) == max(values):
        raise ValueError("nondegenerate fit standardization requires at least two valid roots")
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean)**2 for value in values) / len(values))
    if not math.isfinite(std) or std <= 0.:
        raise ValueError("degenerate fit reward standard deviation; no silent floor")
    standardized = [(value - mean) / std for value in values]
    logits = [beta_std * value for value in standardized]
    largest = max(logits)
    log_z = largest + math.log(math.fsum(math.exp(value - largest) for value in logits)) - math.log(64.)
    effective_beta = beta_std / std
    _normalization(mean, std, log_z)
    if not math.isfinite(effective_beta):
        raise ValueError("nonfinite effective reward beta")
    weights = [math.exp(value - log_z) for value in logits]
    all_weights = weights + [0.] * (64 - len(values))
    total = math.fsum(all_weights)
    return {"fit_reward_mean": mean, "fit_reward_std": std, "log_fit_mean_exp": log_z,
        "effective_beta": effective_beta, "valid_fit_count": len(values),
        "fit_desirability_mean": total / 64.,
        "fit_desirability_ess": total**2 / math.fsum(value**2 for value in all_weights),
        "fit_desirability_range": [min(all_weights), max(all_weights)],
        "fit_valid_desirability_range": [min(weights), max(weights)],
        "fit_standardized_reward_range": [min(standardized), max(standardized)]}


def _metadata(model, records, calibration):
    return {"schema": SCHEMA + "_calibration", **_recipe(model.beta_std, model.objective),
        **_statistics(records, model.beta_std), "fit_root_ids": [row["root_id"] for row in records],
        "fit_label_records": deepcopy(records), "fit_label_records_sha256": legacy._json_sha(records),
        "config_sha256": legacy._json_sha(model.config.to_dict()),
        "scale_calibration_metadata_sha256": legacy._json_sha(calibration)}


class StandardizedSolverErrorRegressor(calibrated.CalibratedSolverErrorRegressor):
    expected_sampling_strength = None

    def __init__(self, config, *, beta_std, objective=OBJECTIVE):
        _recipe(beta_std, objective)
        if config.validity_weight != .1:
            raise ValueError("registered standardized guidance fixes auxiliary validity weight=0.1")
        super().__init__(config, scale_enabled=True, init_enabled=True)
        self.beta_std, self.objective = float(beta_std), objective
        self.standardized_metadata = None
        self.fit_reward_mean = self.fit_reward_std = self.log_fit_mean_exp = None
        # A completed fit replaces this with beta_std / fit_reward_std.
        self.beta = None

    def forward(self, noisy_future_white, sigma, conditions, *, initial_context):
        if self.scale_enabled is not True or self.init_enabled is not True:
            raise ValueError("standardized guidance requires combined fixed input calibration")
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
            raise FloatingPointError("nonfinite standardized guidance output")
        result = {"validity_logit": logits[:, 1], "valid_probability": logits[:, 1].sigmoid()}
        if self.objective == MEAN_OBJECTIVE:
            return {**result, "standardized_reward": logits[:, 0]}
        desirability = logits[:, 0].exp()
        if not torch.isfinite(desirability).all():
            raise FloatingPointError("exponential desirability overflow; no silent clipping")
        return {**result, "log_desirability": logits[:, 0], "desirability": desirability}

    def score_potential(self, prediction):
        value = prediction["log_desirability" if self.objective == OBJECTIVE else "standardized_reward"]
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError("finite raw standardized guidance potential required")
        # Lambda is a separate sampler policy, not another target beta.
        return value if self.objective == OBJECTIVE else self.beta_std * value

    def loss(self, clean_future_white, conditions, initial_context, macro_errors, numerically_valid,
             *, generator=None, sigma=None, noise=None):
        if self.standardized_metadata is None:
            raise ValueError("completed fit-only standardization required before training")
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
        targets, valid = standardized_targets(macro_errors, numerically_valid, prototype=clean,
            fit_reward_mean=self.fit_reward_mean, fit_reward_std=self.fit_reward_std,
            log_fit_mean_exp=self.log_fit_mean_exp, beta_std=self.beta_std,
            objective=self.objective, error_scale=self.config.error_scale)
        if len(targets) != len(clean):
            raise ValueError("solver label and state counts differ")
        prediction = self(clean+sigma[:, None]*noise.detach(), sigma, conditions, initial_context=context)
        if self.objective == OBJECTIVE:
            regression = exponential.desirability_mse(prediction["log_desirability"], targets)
            metrics = {"desirability_loss": regression.detach(), "desirability_target": targets.detach(),
                "predicted_desirability": prediction["desirability"].detach(),
                "log_desirability": prediction["log_desirability"].detach()}
        else:
            raw = prediction["standardized_reward"]
            regression = ((raw[valid] - targets[valid]).square().mean()
                          if bool(valid.any()) else raw.sum() * 0.)
            metrics = {"standardized_reward_loss": regression.detach(),
                "standardized_reward_target": targets.detach(), "predicted_standardized_reward": raw.detach()}
        validity_loss = F.binary_cross_entropy_with_logits(prediction["validity_logit"], valid.to(clean))
        loss = regression + self.config.validity_weight * validity_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite standardized guidance loss; no silent clipping")
        return loss, {**metrics, "regression_loss": regression.detach(), "validity_loss": validity_loss.detach(),
            "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()), "sigma": sigma.detach()}

    def runtime_settings(self):
        return {"schema": SCHEMA, **_recipe(self.beta_std, self.objective), "effective_beta": self.beta,
            "fit_reward_mean": self.fit_reward_mean, "fit_reward_std": self.fit_reward_std,
            "log_fit_mean_exp": self.log_fit_mean_exp, "expected_sampling_strength": self.expected_sampling_strength,
            "scale_enabled": self.scale_enabled, "init_enabled": self.init_enabled,
            "calibration_metadata_hash": (None if self.calibration_metadata is None
                                           else legacy._json_sha(self.calibration_metadata)),
            "standardized_metadata_hash": (None if self.standardized_metadata is None
                                           else legacy._json_sha(self.standardized_metadata))}

    def architecture_report(self):
        return {**super().architecture_report(), "schema": SCHEMA,
            "recipe": _recipe(self.beta_std, self.objective),
            "reward_target": ("fit_normalized_joint_valid_exponential_of_standardized_bounded_reward"
                              if self.objective == OBJECTIVE else "valid_fit_standardized_bounded_reward"),
            "invalid_labels": ("zero_desirability_targets_and_separate_auxiliary_validity_BCE"
                               if self.objective == OBJECTIVE else "excluded_from_mean_regression_auxiliary_validity_BCE_only"),
            "output": "raw_log_h" if self.objective == OBJECTIVE else "raw_standardized_reward",
            "scientific_admission": False, "exact_tilted_sampling_claimed": False}


def make_standardized_regressor(config, *, parameter_dtype, beta_std, objective=OBJECTIVE, device="cpu"):
    if parameter_dtype not in (torch.float32, torch.float64):
        raise ValueError("guidance parameters require explicit float32 or float64")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter_dtype)
        model = StandardizedSolverErrorRegressor(config, beta_std=beta_std, objective=objective)
    finally:
        torch.set_default_dtype(previous)
    return model.to(device=device)


def _install_metadata(model, metadata):
    model.standardized_metadata = deepcopy(metadata)
    model.fit_reward_mean, model.fit_reward_std = metadata["fit_reward_mean"], metadata["fit_reward_std"]
    model.log_fit_mean_exp, model.beta = metadata["log_fit_mean_exp"], metadata["effective_beta"]


@torch.no_grad()
def fit_standardized_calibration(model, rows, *, seed, batch_size=4):
    if (not isinstance(model, StandardizedSolverErrorRegressor) or model.calibration_metadata is not None
            or model.standardized_metadata is not None or any(value is not None for value in
                (model.fit_reward_mean, model.fit_reward_std, model.log_fit_mean_exp, model.beta))):
        raise ValueError("fresh standardized model required; no silent refitting")
    if model.scale_enabled is not True or model.init_enabled is not True:
        raise ValueError("standardized guidance fixes combined calibration")
    parameter = next(model.parameters())
    # Validate roots/inputs before fitting anything and compute labels in float64.
    calibrated._fit_rows(model, rows)
    rewards, valid = legacy.bounded_error_targets(rows["errors"], rows["valid"],
        prototype=parameter.double(), error_scale=model.config.error_scale)
    # Inspect original labels, not just transformed rewards: vector/scalar
    # transcendental kernels may add lane-dependent rounding to equal inputs.
    # Reject the degenerate problem; do not normalize that numerical residue.
    valid_errors = torch.as_tensor(rows["errors"], dtype=torch.float64, device=valid.device)[valid]
    if len(valid_errors) < 2 or bool(valid_errors.min() == valid_errors.max()):
        raise ValueError("nondegenerate fit standardization requires distinct valid solver errors")
    records = [{"root_id": root, "valid": bool(valid[index]),
                "reward": float(rewards[index]) if bool(valid[index]) else None}
               for index, root in enumerate(rows["root_ids"])]
    _statistics(records, model.beta_std)  # Fail degenerate labels before touching model state.
    with torch.random.fork_rng(devices=[]):
        view = calibrated.make_calibrated_regressor(model.config, parameter_dtype=parameter.dtype,
            device=parameter.device, scale_enabled=True, init_enabled=True)
    view.load_state_dict(model.state_dict(), strict=True)
    calibration = calibrated.fit_calibration(view, rows, seed=seed, batch_size=batch_size)
    metadata = _metadata(model, records, calibration)
    view.head[-1].bias[0] = 0.
    model.load_state_dict(view.state_dict(), strict=True)
    model.calibration_metadata = deepcopy(calibration)
    _install_metadata(model, metadata)
    return deepcopy(metadata)


def _validate_standardized(model, metadata, calibration, roots, label_sha):
    if not isinstance(metadata, dict) or "fit_label_records" not in metadata:
        raise ValueError("missing standardized fit provenance")
    without_label = {key: value for key, value in calibration.items() if key != "label_manifest_sha256"}
    expected = {**_metadata(model, metadata["fit_label_records"], without_label), "label_manifest_sha256": label_sha}
    if (legacy._json_sha(metadata) != legacy._json_sha(expected)
            or metadata["fit_root_ids"] != calibration["fit_root_ids"]
            or set(metadata["fit_root_ids"]) != set(roots["train_root_ids"])
            or metadata["valid_fit_count"] != calibration["valid_fit_count"]):
        raise ValueError("standardized objective, fit statistics, roots or label provenance differs")
    # v2's historical reward-mean diagnostic is stored at parameter precision.
    dtype = next(model.parameters()).dtype
    fit_valid = [row["reward"] for row in metadata["fit_label_records"] if row["valid"]]
    legacy_mean = float(torch.tensor(fit_valid, dtype=dtype).double().mean())
    if not math.isclose(calibration["fit_reward_mean"], legacy_mean, rel_tol=0., abs_tol=1.e-15):
        raise ValueError("standardized fit labels differ from v2 input-calibration labels")


def standardized_payload(model, *, label_manifest_sha256, train_root_ids, validation_root_ids, training):
    if (not isinstance(model, StandardizedSolverErrorRegressor) or model.standardized_metadata is None
            or model.scale_enabled is not True or model.init_enabled is not True
            or model.expected_sampling_strength is not None):
        raise ValueError("payload requires completed standardized calibration and separate sampling policy")
    payload = calibrated.guidance_payload(model, label_manifest_sha256=label_manifest_sha256,
        train_root_ids=train_root_ids, validation_root_ids=validation_root_ids, training=training)
    metadata = {**deepcopy(model.standardized_metadata), "label_manifest_sha256": label_manifest_sha256}
    _validate_standardized(model, metadata, payload["calibration"], payload["root_partition"], label_manifest_sha256)
    if any(getattr(model, name) != metadata[key] for name, key in
           (("fit_reward_mean", "fit_reward_mean"), ("fit_reward_std", "fit_reward_std"),
            ("log_fit_mean_exp", "log_fit_mean_exp"), ("beta", "effective_beta"))):
        raise ValueError("standardized runtime statistics changed after fitting")
    payload.update(schema=SCHEMA, recipe=_recipe(model.beta_std, model.objective), standardized=metadata)
    payload["metadata_sha256"] = legacy._json_sha({key: value for key, value in payload.items()
                                                  if key not in ("model", "metadata_sha256")})
    return payload


def regressor_from_payload(payload, *, device="cpu"):
    fields = {"schema", "config", "model", "label_manifest_sha256", "root_partition", "training",
        "architecture", "recipe", "calibration", "standardized", "state_sha256", "metadata_sha256"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != SCHEMA:
        raise ValueError("unexpected standardized guidance checkpoint fields")
    legacy._sha(payload["label_manifest_sha256"], "actual solver label manifest")
    recipe = payload["recipe"]
    if not isinstance(recipe, dict) or recipe != _recipe(recipe.get("beta_std"), recipe.get("objective")):
        raise ValueError("unknown standardized guidance recipe")
    roots = payload["root_partition"]
    if roots != legacy.validate_root_partition(roots["train_root_ids"], roots["validation_root_ids"]):
        raise ValueError("standardized guidance root partition differs")
    training = payload["training"]
    if (not isinstance(training, dict)
            or training.get("label_source") not in ("actual_public_solver_macro_error", "synthetic_fixture")
            or type(training.get("step")) is not int or training["step"] < 0):
        raise ValueError("invalid standardized guidance training provenance")
    state = payload["model"]
    dtype = legacy._validated_guide_state_dtype(state)
    with torch.random.fork_rng(devices=[]):
        model = make_standardized_regressor(legacy.SolverErrorGuidanceConfig.from_dict(payload["config"]),
            parameter_dtype=dtype, beta_std=recipe["beta_std"], objective=recipe["objective"])
    if json.dumps(payload["architecture"], sort_keys=True) != json.dumps(model.architecture_report(), sort_keys=True):
        raise ValueError("standardized guidance architecture differs")
    legacy._validate_guide_state(model, state)
    model.load_state_dict(state, strict=True)
    calibrated._validate_calibration(model, payload["calibration"], roots, payload["label_manifest_sha256"])
    _validate_standardized(model, payload["standardized"], payload["calibration"], roots, payload["label_manifest_sha256"])
    model.calibration_metadata = deepcopy(payload["calibration"])
    del model.calibration_metadata["label_manifest_sha256"]
    metadata = deepcopy(payload["standardized"])
    del metadata["label_manifest_sha256"]
    _install_metadata(model, metadata)
    return model.to(device).eval().requires_grad_(False)
