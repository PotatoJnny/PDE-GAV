"""EDM in independent VPFP value controls, with a fixed final slope lift.

Noise, score regression and every Heun state live in the reduced value space.
This is not projection of the full-state score. The output is lifted to the
existing decoder ABI only after sampling. The source encoder's fixed finite
difference constraint is checked before discarding its redundant slopes.
Neither this representation nor the linear Sobolev loss qualifies physical
sample distributions or matches nonlinear f/macro derivative distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import torch
from torch import nn

from .structured_diffusion import (
    VPFPStructuredEDMConfig, VPFPStructuredEDM, VPFPDiffusionBatchSampler,
    VPFPCoefficientNormalizer,
    _SpatiotemporalBlock, _coefficients, _fourier_basis, _join, _seeds, _sha,
    _split,
)
from .sobolev_matching import _fourier_weights, _time_factors, VPFPSobolevMetric


CHECKPOINT_SCHEMA = "vpfp_value_control_diffusion_checkpoint_v1"
LIFT_SCHEMA = "uniform_tau_endpoint_one_sided_interior_centered_difference_v1"


@dataclass(frozen=True)
class VPFPValueEDMConfig:
    full_config: VPFPStructuredEDMConfig
    schema: str = "vpfp_independent_value_control_edm_v1"

    def __post_init__(self):
        if not isinstance(self.full_config, VPFPStructuredEDMConfig):
            raise TypeError("value-state config needs the fixed full decoder config")
        if self.schema != "vpfp_independent_value_control_edm_v1":
            raise ValueError("unknown VPFP value-state schema")

    def __getattr__(self, name):
        # Explicit getattr avoids recursive lookup during deepcopy/pickle.
        full = object.__getattribute__(self, "full_config")
        return getattr(full, name)

    @property
    def full_dimension(self):
        return self.full_config.latent_dimension

    @property
    def rho_dimension(self):
        return self.control_count * self.rho_modes

    @property
    def kinetic_dimension(self):
        return self.hermite_channels * self.control_count * self.kinetic_modes

    @property
    def latent_dimension(self):
        return self.rho_dimension + self.kinetic_dimension

    @property
    def inactive_dim(self):
        return 0

    @property
    def field_channels(self):
        return self.hermite_channels + 1

    @classmethod
    def from_full_config(cls, config):
        return cls(config)

    def to_dict(self):
        return {"schema": self.schema, "full_config": self.full_config.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema", "full_config"}:
            raise ValueError("unexpected value-state config fields")
        return cls(VPFPStructuredEDMConfig.from_dict(payload["full_config"]), payload["schema"])


def _value_split(values, config):
    batch = len(values)
    return (values[:, :config.rho_dimension].reshape(batch, config.control_count, config.rho_modes),
            values[:, config.rho_dimension:].reshape(batch, config.hermite_channels,
                                                     config.control_count, config.kinetic_modes))


def value_control_slopes(values, config):
    """Fixed d/dtau finite differences, with time on the penultimate axis."""
    h = config.slope_interval
    return torch.cat(((values[..., 1:2, :] - values[..., :1, :]) / h,
                      (values[..., 2:, :] - values[..., :-2, :]) / (2*h),
                      (values[..., -1:, :] - values[..., -2:-1, :]) / h), dim=-2)


def lift_value_controls(values, config):
    values = _coefficients(values, config)
    rho, kinetic = _value_split(values, config)
    return _join(torch.stack((rho, value_control_slopes(rho, config)), dim=1),
                 torch.stack((kinetic, value_control_slopes(kinetic, config)), dim=2),
                 config.full_config)


def extract_value_controls(full_values, config, *, validate_slopes=True):
    """Reject answers off the encoder subspace; allow only roundoff-sized error.

    Quantization of source values before differencing is accounted for using
    the magnitude of the actual neighboring values, not just the resulting
    slope (which can cancel). No fitted or generated statistic sets tolerance.
    """
    full_values = _coefficients(full_values, config.full_config)
    rho, kinetic = _split(full_values, config.full_config)
    if validate_slopes:
        epsilon = torch.finfo(full_values.dtype).eps
        for group, controls in (("rho", rho[:, None]), ("kinetic", kinetic)):
            # Chunking bounds temporary FP64 memory on production corpora.
            for chunk in controls.split(8):
                values, slopes = chunk[:, :, 0].double(), chunk[:, :, 1].double()
                expected = value_control_slopes(values, config)
                magnitude = values.abs()
                neighboring = torch.cat((magnitude[..., 1:2, :] + magnitude[..., :1, :],
                    (magnitude[..., 2:, :] + magnitude[..., :-2, :]) / 2,
                    magnitude[..., -1:, :] + magnitude[..., -2:-1, :]), dim=-2) / config.slope_interval
                tolerance = 8*epsilon*(neighboring + slopes.abs()) + 8*torch.finfo(full_values.dtype).tiny
                if not bool(torch.all((slopes - expected).abs() <= tolerance).item()):
                    raise ValueError(f"{group} slopes leave the fixed value-control lift subspace")
    return torch.cat((rho[:, 0].flatten(1), kinetic[:, :, 0].flatten(1)), dim=1)


class VPFPValueControlNormalizer(nn.Module):
    """Exact value scales inherited from the paired full-state normalizer.

    The full-state block floor includes interval-scaled slopes; recomputing a
    value-only floor would change high-Hermite noise scales and confound the
    representation comparison. We retain exactly its value column instead.
    """

    def __init__(self, config, group_scales, *, fit_metadata):
        super().__init__()
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("normalizer needs value-state config")
        scales = torch.as_tensor(group_scales, dtype=torch.float64).detach().clone()
        if (scales.shape != (1 + config.hermite_channels,) or
                not torch.isfinite(scales).all() or torch.any(scales <= 0)):
            raise ValueError("invalid value-state normalization scales")
        if fit_metadata.get("split") != "train" or fit_metadata.get("generated_samples_used_for_fit") is not False:
            raise ValueError("value normalization must be fitted on training answers only")
        self.config, self.fit_metadata = config, dict(fit_metadata)
        self.register_buffer("group_scales", scales)

    @classmethod
    def fit(cls, config, full_training_latents, *, sample_ids, split="train",
            relative_floor=.05, absolute_floor=1.e-6):
        if split != "train":
            raise ValueError("normalization only accepts the train split")
        if (not math.isfinite(relative_floor) or not 0 < relative_floor <= 1
                or not math.isfinite(absolute_floor) or absolute_floor <= 0):
            raise ValueError("normalization floors must be positive and bounded")
        # Validate source slopes before fitting or discarding them. The return
        # is deliberately not retained beside the full production corpus.
        extract_value_controls(full_training_latents, config)
        ids = _seeds(sample_ids)
        if len(full_training_latents) < 2 or len(full_training_latents) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("normalization needs >=2 distinct training sample identities")
        parent = VPFPCoefficientNormalizer.fit(config.full_config, full_training_latents,
            sample_ids=ids, split=split, relative_floor=relative_floor, absolute_floor=absolute_floor)
        parent_payload = parent.to_payload()
        parent_sha = sha256(json.dumps(parent_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        scales = parent.group_scales[:, 0]
        return cls(config, scales, fit_metadata={
            "schema": "vpfp_train_value_group_rms_normalization_v1", "split": "train",
            "sample_ids": list(ids), "generated_samples_used_for_fit": False,
            "centering": "none", "scale": "value_group_rms_with_inherited_full_block_relative_floor",
            "floor_policy": "inherited_full_representation_for_paired_control",
            "parent_normalizer_payload": parent_payload, "parent_normalizer_sha256": parent_sha,
            "relative_floor": float(relative_floor), "absolute_floor": float(absolute_floor),
            "rho_floor": parent.fit_metadata["rho_floor"], "kinetic_floor": parent.fit_metadata["kinetic_floor"],
            "slope_lift": LIFT_SCHEMA, "delta_tau": config.slope_interval,
            "slope_subspace_verified": True, "roundoff_tolerance_ulps": 8,
            "inactive_policy": "exact_zero_after_lift", "physical_answer_fidelity_certified": False})

    def _transform(self, values, inverse):
        values = _coefficients(values, self.config)
        rho, kinetic = _value_split(values, self.config)
        scales = self.group_scales.to(values)
        if not torch.isfinite(scales).all() or torch.any(scales <= 0):
            raise ValueError("value normalization scales unrepresentable in input dtype")
        factors = scales if inverse else scales.reciprocal()
        output = torch.cat(((rho * factors[0]).flatten(1),
            (kinetic * factors[1:][None, :, None, None]).flatten(1)), dim=1)
        return _coefficients(output, self.config)

    def encode(self, full_raw_latents):
        return self._transform(extract_value_controls(full_raw_latents, self.config), False)

    def decode(self, normalized_values):
        return lift_value_controls(self._transform(normalized_values, True), self.config)

    def to_payload(self):
        return {"config": self.config.to_dict(), "group_scales": self.group_scales.detach().cpu().tolist(),
                "fit_metadata": dict(self.fit_metadata)}

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"config", "group_scales", "fit_metadata"}:
            raise ValueError("unexpected value normalizer fields")
        return cls(VPFPValueEDMConfig.from_dict(payload["config"]), payload["group_scales"],
                   fit_metadata=payload["fit_metadata"])


class VPFPValueFourierTransform(nn.Module):
    """Exact Fourier computational view with one channel per value field."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("rho_basis", _fourier_basis(config.rho_max_mode, config.grid_size, constant=False, normalized=False))
        self.register_buffer("kinetic_basis", _fourier_basis(config.kinetic_max_mode, config.grid_size, constant=True, normalized=True))

    def to_field(self, values):
        values = _coefficients(values, self.config)
        rho, kinetic = _value_split(values, self.config)
        return torch.cat((torch.einsum("bnm,mxy->bnxy", rho, self.rho_basis.to(values))[:, None],
                          torch.einsum("bcnm,mxy->bcnxy", kinetic, self.kinetic_basis.to(values))), dim=1)

    def to_coefficients(self, field):
        cfg = self.config
        if field.ndim != 5 or field.shape[1:] != (cfg.field_channels, cfg.control_count, cfg.grid_size, cfg.grid_size):
            raise ValueError("unexpected VPFP value field layout")
        if not field.is_floating_point() or not torch.isfinite(field).all():
            raise ValueError("VPFP value field must be finite floating values")
        rho = torch.einsum("bnxy,mxy->bnm", field[:, 0], self.rho_basis.to(field)) * (2/cfg.grid_size**2)
        kinetic = torch.einsum("bcnxy,mxy->bcnm", field[:, 1:], self.kinetic_basis.to(field)) / cfg.grid_size**2
        return torch.cat((rho.flatten(1), kinetic.flatten(1)), dim=1)


class VPFPValueControlEDM(VPFPStructuredEDM):
    """Inherited EDM formulas operate exclusively on reduced value states."""

    checkpoint_schema = CHECKPOINT_SCHEMA

    def __init__(self, config):
        nn.Module.__init__(self)
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("value-state model needs its explicit config")
        self.config = config
        self.transform = VPFPValueFourierTransform(config)
        self.input = nn.Conv3d(config.field_channels, config.width, 1)
        self.context = nn.Sequential(nn.Linear(18, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.time_position = nn.Linear(3, config.width)
        self.blocks = nn.ModuleList(_SpatiotemporalBlock(config.width,
            config.temporal_dilation_cycle[index % len(config.temporal_dilation_cycle)]) for index in range(config.depth))
        self.output = nn.Conv3d(config.width, config.field_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class VPFPValueSobolevMetric(nn.Module):
    """Fixed SPD residual metric, trace-normalized in REDUCED noise space.

    The time operator includes the fixed slope lift. Its trace is not the
    old independent-value-and-slope operator trace. All physical operators,
    normalizer factors and weights are frozen, preserving the population
    conditional mean for EDM denoising or a flow-matching velocity residual.
    """

    _checked_cast = VPFPSobolevMetric._checked_cast
    _normalized = staticmethod(VPFPSobolevMetric._normalized)

    def __init__(self, config, normalizer, *, spatial_weight=0., temporal_weight=.25,
                 spatial_lengths=(2*math.pi, 2*math.pi), latent_input_scale=2., structured_local_scale=1.):
        super().__init__()
        if config != normalizer.config:
            raise ValueError("Sobolev config differs from value normalizer")
        for name, value in (("spatial_weight", spatial_weight), ("temporal_weight", temporal_weight)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite nonnegative")
        lengths = tuple(float(value) for value in spatial_lengths)
        if len(lengths) != 2 or not all(math.isfinite(value) and value > 0 for value in lengths):
            raise ValueError("spatial lengths must contain two positive finite values")
        if not all(math.isfinite(value) and value > 0 for value in (latent_input_scale, structured_local_scale)):
            raise ValueError("physical decoder scales must be positive finite")
        self.config = config
        self.spatial_weight, self.temporal_weight = float(spatial_weight), float(temporal_weight)
        self.total_scale = 1 + self.spatial_weight + self.temporal_weight
        if not math.isfinite(self.total_scale):
            raise ValueError("Sobolev total scale overflow")
        self.spatial_lengths = lengths
        self.latent_input_scale, self.structured_local_scale = float(latent_input_scale), float(structured_local_scale)
        self.normalizer_sha256 = sha256(json.dumps(normalizer.to_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        factors = normalizer.group_scales.detach().cpu().double().clone() * latent_input_scale
        factors[1:] *= structured_local_scale / math.sqrt(config.kinetic_modes)
        self.register_buffer("decode_factors", factors)
        mass_factor, time_factor = _time_factors(config)
        self.register_buffer("mass_factor", mass_factor)
        self.register_buffer("time_factor", time_factor)
        # Each basis row is one independent value-coordinate perturbation.
        basis = torch.eye(config.control_count, dtype=torch.float64)[:, :, None]
        slopes = value_control_slopes(basis, config)
        local_basis = torch.stack((basis[:, :-1], basis[:, 1:], slopes[:, :-1], slopes[:, 1:]), dim=-1)
        temporal_traces = [(local_basis @ operator.T).square().sum() for operator in (mass_factor, time_factor)]
        for group, maximum, constant, normalized in (("rho", config.rho_max_mode, False, False),
                ("kinetic", config.kinetic_max_mode, True, True)):
            weights = _fourier_weights(maximum, lengths, constant=constant, normalized=normalized)
            for name, values in zip(("mass", "x", "y"), weights):
                self.register_buffer(f"{group}_{name}_weights", values)
            channel_scale = factors[:1] if group == "rho" else factors[1:]
            traces = channel_scale.square().mean() * torch.stack((
                temporal_traces[0] * (weights[1]+weights[2]).sum(), temporal_traces[1] * weights[0].sum()))
            if not torch.isfinite(traces).all() or traces[1] <= 0 or ((group == "rho" or maximum > 0) and traces[0] <= 0):
                raise ValueError("value Sobolev trace overflow or underflow")
            self.register_buffer(f"{group}_traces", traces)
        self._buffer_nonzero_counts = {name: int(torch.count_nonzero(buffer).item()) for name, buffer in self.named_buffers()}

    @property
    def enabled(self):
        return self.spatial_weight != 0 or self.temporal_weight != 0

    def components(self, residual):
        values = _coefficients(residual, self.config)
        if values.dtype not in (torch.float32, torch.float64):
            raise ValueError("value Sobolev needs float32 or float64")
        rho, kinetic = _value_split(values, self.config)
        output = {"rho_base": rho.square().flatten(1).mean(1), "kinetic_base": kinetic.square().flatten(1).mean(1)}
        decode = self._checked_cast("decode_factors", values)
        mass_factor, time_factor = self._checked_cast("mass_factor", values), self._checked_cast("time_factor", values)
        for group, coefficients, factors in (("rho", rho[:, None], decode[:1]), ("kinetic", kinetic, decode[1:])):
            raw = coefficients * factors[None, :, None, None]
            slopes = value_control_slopes(raw, self.config)
            local = torch.stack((raw[:, :, :-1], raw[:, :, 1:], slopes[:, :, :-1], slopes[:, :, 1:]), dim=-1)
            mass_by_mode = (local @ mass_factor.T).square().sum(dim=(-1, 2)).mean(1)
            time_by_mode = (local @ time_factor.T).square().sum(dim=(-1, 2)).mean(1)
            for axis in ("x", "y"):
                output[f"{group}_raw_{axis}"] = (mass_by_mode * self._checked_cast(f"{group}_{axis}_weights", values)).sum(1)
            output[f"{group}_raw_temporal"] = (time_by_mode * self._checked_cast(f"{group}_mass_weights", values)).sum(1)
            traces = self._checked_cast(f"{group}_traces", values)
            output[f"{group}_spatial"] = self._normalized(output[f"{group}_raw_x"]+output[f"{group}_raw_y"], traces[0])
            output[f"{group}_temporal"] = self._normalized(output[f"{group}_raw_temporal"], traces[1])
            output[f"{group}_total"] = output[f"{group}_base"]
            if self.spatial_weight:
                output[f"{group}_total"] = output[f"{group}_total"] + self.spatial_weight * output[f"{group}_spatial"]
            if self.temporal_weight:
                output[f"{group}_total"] = output[f"{group}_total"] + self.temporal_weight * output[f"{group}_temporal"]
            if self.enabled:
                output[f"{group}_total"] = output[f"{group}_total"] / self.total_scale
        fraction = self.config.macro_loss_fraction
        for component in ("base", "spatial", "temporal", "total"):
            output[component] = fraction*output[f"rho_{component}"] + (1-fraction)*output[f"kinetic_{component}"]
        if not torch.isfinite(torch.stack(tuple(output.values()), dim=-1)).all():
            raise ValueError("value Sobolev energies overflow")
        return output

    def forward(self, residual):
        return self.components(residual)["total"]

    def report(self):
        return {"schema": "vpfp_fixed_value_lift_sobolev_residual_v1", "normalizer_sha256": self.normalizer_sha256,
            "spatial_weight": self.spatial_weight, "temporal_weight": self.temporal_weight,
            "total_scale_divisor": self.total_scale, "spatial_lengths": list(self.spatial_lengths),
            "time_domain": list(self.config.time_domain), "delta_tau": self.config.slope_interval,
            "latent_input_scale": self.latent_input_scale, "structured_local_scale": self.structured_local_scale,
            "rho_spatial_temporal_traces": self.rho_traces.detach().cpu().tolist(),
            "kinetic_spatial_temporal_traces": self.kinetic_traces.detach().cpu().tolist(),
            "normalization": "fixed_operator_trace_under_iid_unit_independent_value_noise",
            "slope_lift": LIFT_SCHEMA, "population_optimum": "unchanged_conditional_mean_in_reduced_state",
            "duration_scaling": "raw_temporal_energy_and_trace_both_scale_as_duration_inverse_squared",
            "physical_derivative_distribution_matching": False, "prediction_or_batch_dependent_weights": False}


class VPFPValueDiffusionBatchSampler(VPFPDiffusionBatchSampler):
    def __init__(self, model, normalizer, **kwargs):
        super().__init__(model, normalizer, **kwargs)
        self.identity.update(kind="independent_value_control_conditional_edm",
            latent_dimension=model.config.full_dimension, random_dimension=model.config.latent_dimension,
            slope_lift=LIFT_SCHEMA)
        self.identity["sampling"]["noise_space"] = "independent_value_coefficients_only"
        self.identity["sampling"]["decoder_lift_timing"] = "after_complete_reduced_Heun_trajectory"


def _sampler_from_value_payload(payload, expected_sha256, device, *, steps, churn):
    """Called only after the canonical loader verified bytes and safe loading."""
    expected = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != expected or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected value diffusion checkpoint schema/fields")
    base_sha = _sha(payload["base_decoder_sha256"], "base decoder identity")
    corpus_sha = _sha(payload["corpus_manifest_sha256"], "corpus manifest identity")
    training = payload["training"]
    if not isinstance(training, dict) or not training:
        raise ValueError("value checkpoint needs nonempty training provenance")
    try:
        json.dumps(training, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("training provenance must be finite JSON") from error
    config = VPFPValueEDMConfig.from_dict(payload["config"])
    normalizer = VPFPValueControlNormalizer.from_payload(payload["normalizer"])
    if config != normalizer.config:
        raise ValueError("value checkpoint model/normalizer configs differ")
    metadata = normalizer.fit_metadata
    ids = _seeds(metadata.get("sample_ids", ()))
    if (metadata.get("schema") != "vpfp_train_value_group_rms_normalization_v1" or len(ids) < 2 or len(set(ids)) != len(ids)
            or metadata.get("centering") != "none" or metadata.get("inactive_policy") != "exact_zero_after_lift"
            or metadata.get("slope_lift") != LIFT_SCHEMA or metadata.get("delta_tau") != config.slope_interval
            or metadata.get("slope_subspace_verified") is not True or metadata.get("roundoff_tolerance_ulps") != 8):
        raise ValueError("value normalization provenance/layout invalid")
    try:
        json.dumps(metadata, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("value normalization metadata must be finite JSON") from error
    parent_payload = metadata.get("parent_normalizer_payload")
    if not isinstance(parent_payload, dict) or set(parent_payload) != {"config", "group_scales", "fit_metadata"}:
        raise ValueError("value normalization needs its bound parent payload")
    parent_sha = _sha(metadata.get("parent_normalizer_sha256"), "parent normalizer identity")
    parent = VPFPCoefficientNormalizer.from_payload(parent_payload)
    parent_metadata = parent.fit_metadata
    if (parent.config != config.full_config or not torch.equal(parent.group_scales[:, 0], normalizer.group_scales)
            or metadata.get("floor_policy") != "inherited_full_representation_for_paired_control"
            or metadata.get("scale") != "value_group_rms_with_inherited_full_block_relative_floor"
            or parent_metadata.get("schema") != "vpfp_train_group_rms_normalization_v1"
            or parent_metadata.get("centering") != "none"
            or parent_metadata.get("scale") != "group_rms_with_block_relative_floor"
            or parent_metadata.get("inactive_policy") != "exact_zero"
            or parent_metadata.get("slope_transform") != "d_d_tau_times_delta_tau"
            or parent_metadata.get("delta_tau") != config.slope_interval
            or parent_metadata.get("sample_ids") != list(ids)
            or any(metadata.get(key) != parent_metadata.get(key) for key in
                ("relative_floor", "absolute_floor", "rho_floor", "kinetic_floor"))):
        raise ValueError("value normalization differs from its paired full-state parent")
    model = VPFPValueControlEDM(config)
    state, expected_state = payload["model"], model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected_state):
        raise ValueError("value checkpoint model tensor keys differ")
    for name, expected_tensor in expected_state.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.shape != expected_tensor.shape
                or tensor.dtype != expected_tensor.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"invalid value checkpoint tensor: {name}")
        if name.startswith("transform.") and not torch.equal(tensor, expected_tensor):
            raise ValueError(f"checkpoint modified fixed value Fourier transform: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    normalizer.to(device=device)
    sampler = VPFPValueDiffusionBatchSampler(model, normalizer, base_decoder_sha256=base_sha,
        checkpoint_sha256=expected_sha256, steps=steps, churn=churn)
    sampler.identity.update(corpus_manifest_sha256=corpus_sha, training=dict(training))
    return sampler
