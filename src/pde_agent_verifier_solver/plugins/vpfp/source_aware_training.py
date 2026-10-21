"""Thin opt-in baseline / jet / source micro-EDM training ablation.

The old loss performs the ONLY noise draw/denoiser call and supplies its full
metrics. Source steps capture that same attached clean prediction. Source is
an intentionally biased auxiliary objective, not equivalent score matching.
No numerical audit evaluator, independent macro sample, or noisy-field PDE
constraint is used. A factory/driver must separately verify frozen warmstart,
calibration SHA, geometry, and save source_reports even on a failed run.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

import numpy as np
import torch
from torch import nn

from .diffusion_data import _stratum
from .innovation_diffusion import VPFPInnovationConditionalEDM
from .macro_jet_conditioning import MacroConditionCoordinateJets, VPFPMacroJetInnovationEDM
from .source_aware_loss import (SourceBudgetObjective, innovation_prediction_to_values,
                                source_budget_loss)


ARMS = ("baseline", "jet", "source")
SOURCE_STEP_STATE_KEY = "_source_aux_training_step"
SOURCE_CADENCE = 4
SOURCE_SIGMA_CUTOFF = .5
TIME_FRACTIONS = (0., .015625, .0625, .125, .25, .5, .75, 1.)


def _sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class SourceAwareTrainingContext:
    """Non-Module, externally frozen, shared read-only with an EMA deepcopy.

    ``calibration`` is the complete TRAIN-only calibration report, not a
    candidate-derived threshold. Its SHA is checked by the caller against the
    file bytes; this class validates its role, strata and training root subset.
    Modules must already be on the compute device with frozen parameters.
    ``model.to(...)`` deliberately does not move or register this context.
    """
    def __init__(self, *, decoder, normalizer, transform, macro_normalizer,
                 calibration, budget_sha256, source_weight, probe_seed, budget_multiplier):
        if (not isinstance(budget_sha256, str) or len(budget_sha256) != 64
                or any(c not in "0123456789abcdef" for c in budget_sha256)):
            raise ValueError("frozen calibration file SHA256 required")
        if type(probe_seed) is not int or not 0 <= probe_seed < 2**63:
            raise ValueError("an explicit nonnegative int64 probe seed is required")
        if (isinstance(source_weight, bool) or not math.isfinite(source_weight) or source_weight < 0
                or isinstance(budget_multiplier, bool) or not math.isfinite(budget_multiplier)
                or budget_multiplier <= 0):
            raise ValueError("explicit finite source weight and positive budget multiplier required")
        if (calibration.get("training_budget_usable") is not True or calibration.get("fit_split") != "train"
                or calibration.get("stratum_order") != "nu_bin * temperature_bin_count + temperature_bin"):
            raise ValueError("usable TRAIN-only calibration with original stratum order required")
        ids = calibration.get("budget_sample_ids", [])
        train_ids = normalizer.fit_metadata.get("sample_ids", [])
        if (len(ids) < 2 or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)
                or not set(ids).issubset(set(train_ids))):
            raise ValueError("calibration roots must be distinct members of the normalizer TRAIN roots")
        if any(module.training or any(p.requires_grad for p in module.parameters())
               for module in (decoder, normalizer, transform, macro_normalizer)):
            raise ValueError("freeze/eval all physical context modules before attachment")
        edges = tuple(tuple(float(x) for x in axis) for axis in calibration["bin_edges"])
        if (len(edges) != 2 or tuple(map(len, edges)) != (5, 3)
                or any(not all(math.isfinite(x) for x in axis)
                       or any(a >= b for a, b in zip(axis, axis[1:])) for axis in edges)):
            raise ValueError("original finite increasing 4x2 calibration bin edges required")
        strata = calibration.get("strata", {})
        if set(strata) != {str(i) for i in range(8)}:
            raise ValueError("all eight TRAIN budget strata are required")
        budgets = tuple((float(strata[str(i)]["rms_budget"])*budget_multiplier,
                         float(strata[str(i)]["point_budget"])*budget_multiplier) for i in range(8))
        if any(not math.isfinite(v) or v <= 0 for row in budgets for v in row):
            raise ValueError("all original-unit source budgets must be positive and finite")
        self.decoder, self.normalizer, self.transform = decoder, normalizer, transform
        self.macro_normalizer = macro_normalizer
        self.bin_edges, self.budgets = edges, budgets
        self.source_weight, self.probe_seed = float(source_weight), probe_seed
        self._identity = {"schema": "vpfp_source_training_context_v1", "budget_sha256": budget_sha256,
            "budget_sample_ids": list(ids), "fit_split": "train", "bin_edges": edges,
            "effective_budgets": budgets, "budget_multiplier": float(budget_multiplier),
            "source_weight": self.source_weight, "probe_seed": probe_seed,
            "normalizer_sha256": _sha(normalizer.to_payload()),
            "macro_normalizer_sha256": _sha(macro_normalizer.to_payload()),
            "transform_sha256": _sha(transform.to_payload()),
            "probe_law": "4_points_independent_step_streams_time_uniform_8fractions_xy_uniform_v_two_thermal_two_3thermal_Gaussians",
            "time_fractions": TIME_FRACTIONS, "source_sigma_cutoff": SOURCE_SIGMA_CUTOFF,
            "source_cadence": SOURCE_CADENCE, "selected_batch_index": 0,
            "source_divisor": "entire_actual_batch_size_no_cadence_or_selection_compensation",
            "physical_admission": False}

    def __deepcopy__(self, memo):
        # Shared trainer deepcopy(model) must not duplicate physical assets or
        # inadvertently introduce them into EMA's parameter inventory.
        memo[id(self)] = self
        return self

    def identity(self):
        return deepcopy(self._identity)

    def validate_model(self, model):
        if self.normalizer.config != model.config or self.macro_normalizer.config != model.macro_config:
            raise ValueError("source context must use the micro model's existing condition codecs")
        if self.transform.config.value_config != model.config:
            raise ValueError("source innovation inverse differs from the trained coordinates")
        physical = self.decoder.value_config
        for name in ("control_count", "hermite_channels", "rho_max_mode", "kinetic_max_mode",
                     "time_domain", "collision_nu_range", "temperature_range"):
            if getattr(physical, name) != getattr(model.config, name):
                raise ValueError(f"source decoder/model physical layout differs: {name}")
        if (self.decoder.macro_config.max_mode != model.macro_config.maximum_mode
                or tuple(self.decoder.macro_config.spatial_domain_lengths) != model.spatial_domain_lengths):
            raise ValueError("source decoder/macro spatial geometry differs")
        if any((axis[0], axis[-1]) != bounds for axis, bounds in zip(self.bin_edges,
                (model.config.collision_nu_range, model.config.temperature_range))):
            raise ValueError("calibration condition regime differs from the model")

    def objective(self, condition):
        index = _stratum(condition.detach().cpu().double().tolist(), self.bin_edges)
        rms, point = self.budgets[index]
        return index, SourceBudgetObjective(self.source_weight, SOURCE_SIGMA_CUTOFF, rms, point)

    def probe_points(self, step, condition):
        """Same declared measure, new locations; neither sampler RNG is touched."""
        if type(step) is not int or step < 1:
            raise ValueError("positive training step required for probes")
        def generator(stream):
            seed = int.from_bytes(sha256(f"vpfp-source-training:{self.probe_seed}:{step}:{stream}".encode()).digest()[:8], "little") % 2**63
            return torch.Generator(device="cpu").manual_seed(seed)
        slots = torch.randint(0, len(TIME_FRACTIONS), (4,), generator=generator("time"), device="cpu")
        t0, t1 = self.decoder.macro_config.time_domain
        time = t0+(t1-t0)*torch.tensor(TIME_FRACTIONS, dtype=torch.float64)[slots]
        xy = torch.rand((4, 2), generator=generator("space"), dtype=torch.float64, device="cpu")
        xy *= torch.tensor(self.decoder.macro_config.spatial_domain_lengths, dtype=torch.float64)
        thermal = math.sqrt(float(condition[1].detach()))
        velocity = torch.randn((4, 2), generator=generator("velocity"), dtype=torch.float64, device="cpu")
        velocity *= torch.tensor([1., 3., 1., 3.], dtype=torch.float64)[:, None]*thermal
        return torch.cat((time[:, None], xy, velocity), 1).to(next(self.decoder.parameters()).device)


class VPFPSourceAwareTrainingEDM(VPFPInnovationConditionalEDM):
    """baseline=old; jet=jet only; source=old + source auxiliary (NO jet).

    The counter is an explicitly serialized non-parameter/non-buffer tensor:
    shared.train treats registered buffers as immutable and EMA updates only
    parameters. Raw checkpoints restore this counter; EMA's counter stays 0
    and has no role in validation/sampling. On continuation the driver MUST
    verify raw counter equals the saved completed training step.
    """
    def __init__(self, config, macro_config, *, arm, spatial_domain_lengths, physics_context=None):
        if arm not in ARMS:
            raise ValueError("only isolated baseline, jet and source arms are registered")
        if tuple(config.time_domain) != tuple(macro_config.time_domain):
            raise ValueError("micro/macro time domains differ")
        super().__init__(config, macro_config)
        self.arm = arm
        self.checkpoint_schema = f"vpfp_source_aware_{arm}_innovation_checkpoint_v1"
        self.spatial_domain_lengths = tuple(float(x) for x in spatial_domain_lengths)
        if len(self.spatial_domain_lengths) != 2 or not all(math.isfinite(x) and x > 0 for x in self.spatial_domain_lengths):
            raise ValueError("two explicit positive spatial periods required")
        if arm == "jet":
            self.macro_jets = MacroConditionCoordinateJets(macro_config, spatial_domain_lengths=self.spatial_domain_lengths)
            p = next(self.parameters())
            with torch.random.fork_rng(devices=[] if p.device.type == "cpu" else [p.device.index]):
                self.jet_input = nn.Conv3d(21, config.width, 1, bias=False, device=p.device, dtype=p.dtype)
                nn.init.zeros_(self.jet_input.weight)
        if physics_context is not None and not isinstance(physics_context, SourceAwareTrainingContext):
            raise TypeError("physical assets must be attached as a non-Module SourceAwareTrainingContext")
        self._physics_context = physics_context
        if physics_context is not None:
            physics_context.validate_model(self)
        self._source_step = 0
        self._capture_prediction = False
        self._captured_prediction = None
        self._inside_loss = False
        self.source_reports = []

    @property
    def source_step(self):
        return self._source_step

    def set_source_step(self, step):
        if type(step) is not int or not 0 <= step < 2**63:
            raise ValueError("nonnegative int64 source training step required")
        self._source_step = step

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix+SOURCE_STEP_STATE_KEY] = torch.tensor(self.source_step, dtype=torch.int64, device="cpu")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix+SOURCE_STEP_STATE_KEY
        value = state_dict.pop(key, None)
        if value is None:
            if strict:
                missing_keys.append(key)
        elif not isinstance(value, torch.Tensor) or value.shape != () or value.dtype != torch.int64 or int(value) < 0:
            error_msgs.append(f"invalid source training counter: {key}")
        else:
            self.set_source_step(int(value))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def identity(self):
        return {"schema": self.checkpoint_schema, "arm": self.arm,
            "config": self.config.to_dict(), "macro_config": self.macro_config.to_dict(),
            "spatial_domain_lengths": self.spatial_domain_lengths,
            "physics_context": None if self._physics_context is None else self._physics_context.identity(),
            "counter": "raw_training_loss_calls_custom_state_tensor_not_EMA_parameter_or_buffer",
            "metrics": "sigma_bins_include_aux_when_active_base_and_aux_reported_separately",
            "validation": "original_EDM_only", "scientific_qualification": False}

    def denoise(self, noisy_normalized_coefficients, sigma, conditions):
        if self.arm == "jet":
            result = VPFPMacroJetInnovationEDM.denoise(self, noisy_normalized_coefficients, sigma, conditions)
        else:
            result = super().denoise(noisy_normalized_coefficients, sigma, conditions)
        if self._capture_prediction:
            if self._captured_prediction is not None:
                raise RuntimeError("source auxiliary requires exactly one denoiser call")
            self._captured_prediction = result
        return result

    def loss(self, clean_normalized_coefficients, conditions, *, generator=None, sigma=None, noise=None, residual_metric=None):
        if self._inside_loss:
            raise RuntimeError("source training loss is not reentrant")
        step = self.source_step+1 if self.training else self.source_step
        scheduled = self.training and self.arm == "source" and step % SOURCE_CADENCE == 0
        if self.training and self.arm == "source" and self._physics_context is None:
            raise ValueError("source training requires its frozen physical context; eval is base-only")
        capture = scheduled and self._physics_context.source_weight > 0 if scheduled else False
        self._inside_loss, self._capture_prediction = True, capture
        try:
            base, metrics = super().loss(clean_normalized_coefficients, conditions,
                generator=generator, sigma=sigma, noise=noise, residual_metric=residual_metric)
            prediction = self._captured_prediction
        finally:
            self._inside_loss = self._capture_prediction = False
            self._captured_prediction = None
        loss = base
        if scheduled:
            context = self._physics_context
            level = float(metrics["sigma"][0])
            event = {"step": step, "batch_size": len(metrics["sigma"]), "batch_index": 0,
                     "sigma": level, "evaluated": False, "status": "skipped"}
            self.source_reports.append(event)  # Preserve attempted failure before raising.
            if context.source_weight == 0:
                event["reason"] = "zero_weight"
            elif level >= SOURCE_SIGMA_CUTOFF:
                event["reason"] = "outside_low_sigma_gate"
            else:
                try:
                    if not torch.is_grad_enabled():
                        raise RuntimeError("source training auxiliary requires autograd; use eval for validation")
                    p = next(self.parameters())
                    if next(context.decoder.parameters()).device != p.device:
                        raise ValueError("place the unregistered physical context on the training device explicitly")
                    condition = torch.as_tensor(conditions, device=p.device, dtype=p.dtype)[:1]
                    theta = condition[0, :2]
                    stratum, objective = context.objective(theta)
                    m = context.macro_normalizer.decode(condition[:, 2:])
                    z = innovation_prediction_to_values(prediction[:1], context.normalizer, context.transform, m)
                    points = context.probe_points(step, theta)
                    aux, report = source_budget_loss(context.decoder, points, {"m": m, "z": z}, theta,
                                                    sigma=level, objective=objective)
                    loss = base+aux/len(metrics["sigma"])
                    per_case = metrics["loss_per_case"].to(aux).clone()
                    per_case[0] += aux.detach()
                    bins = torch.bucketize(metrics["sigma"], metrics["sigma"].new_tensor(self.config.sigma_bin_edges[1:-1]))
                    metrics["loss_per_case"] = per_case
                    metrics["sigma_bin_loss_sums"] = torch.stack([per_case[bins == i].sum() for i in range(4)])
                    metrics["source_aux_loss"] = (aux/len(per_case)).detach()
                    event.update({key: float(value) if isinstance(value, torch.Tensor) else value
                                  for key, value in report.items()})
                    event.update(status="evaluated", stratum=stratum, aux_case_loss=float(aux.detach()),
                                 aux_batch_loss=float((aux/len(per_case)).detach()))
                except Exception as error:
                    event.update(status="failed", error_type=type(error).__name__, error=str(error))
                    raise
        if self.training:
            self.set_source_step(step)
        return loss, metrics
