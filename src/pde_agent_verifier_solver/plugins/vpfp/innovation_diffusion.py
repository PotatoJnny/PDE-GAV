"""EDM learning coordinates for complete, conditional VPFP innovations.

The shared trainer sees a declared derived VALUE-coordinate view, not fake
physical answers with invented slopes.  Normalization is fitted on that train
view only.  A sampled coordinate must be unnormalized, inverse-transformed
with the SAME physical macro condition, and finally lifted through the old
native slope map.  No reference PDE or physical projection is used here.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

import torch
from torch import nn

from .coherent_diffusion import VPFPMacroConditionalEDM
from .value_control_diffusion import VPFPValueEDMConfig, _value_split, lift_value_controls


NORMALIZER_SCHEMA = "vpfp_complete_innovation_train_normalizer_v2"
CHECKPOINT_SCHEMA = "vpfp_complete_innovation_conditional_edm_checkpoint_v2"
PRECISION_POLICY = {
    "raw_coordinate_storage": "float64",
    "scale_statistics": "float64",
    "normalization_arithmetic": "float64",
    "white_model_coordinates": "float32",
    "inverse_coordinate_arithmetic": "float64",
    "native_values_before_lift": "float32",
    "native_slope_lift": "original_float32",
    "white_quantization_is_mathematically_lossless": False,
}


def _json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _values(values, config):
    if (not isinstance(values, torch.Tensor) or values.layout != torch.strided
            or values.dtype not in (torch.float32, torch.float64)
            or values.ndim != 2 or values.shape[0] < 1
            or values.shape[1] != config.latent_dimension
            or not bool(torch.isfinite(values).all())):
        raise ValueError("expected finite FP32/FP64 raw VALUE coordinates [batch, D]")
    return values


def _ids(values):
    values = list(values)
    if (len(values) < 2 or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("normalization requires distinct nonnegative training root IDs")
    return values


class VPFPInnovationNormalizer(nn.Module):
    """Train-only channel RMS with distinct first-frame/increment scales.

    There is no centering, covariance eigensystem, time compression, or fit on
    generated/validation answers.  Each channel has two scales, shared over
    spatial Fourier modes (including each cos/sin pair).  A fixed positive
    floor is shared across the two time roles of a channel.  High-Hermite
    channels are retained even when a training residual happens to be zero.
    Identity and innovation arms use exactly this same fitting recipe, NOT
    identical physical noise covariance or comparable raw validation losses.
    """

    def __init__(self, config, scales, *, fit_metadata):
        super().__init__()
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("innovation normalization requires the original value layout")
        scales = torch.as_tensor(scales, dtype=torch.float64).detach().clone()
        metadata = deepcopy(fit_metadata)
        if (scales.shape != (1 + config.hermite_channels, 2)
                or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any())):
            raise ValueError("invalid first-frame/increment normalization scales")
        if (not isinstance(metadata, dict)
                or metadata.get("schema") != "vpfp_train_complete_innovation_rms_v2"
                or metadata.get("precision_policy") != PRECISION_POLICY
                or metadata.get("split") != "train"
                or metadata.get("generated_samples_used_for_fit") is not False
                or metadata.get("validation_samples_used_for_fit") is not False
                or metadata.get("centering") != "none"
                or metadata.get("roles") != ["initial_frame", "remaining_frames"]
                or not isinstance(metadata.get("coordinate_transform"), dict)):
            raise ValueError("invalid train-only innovation fit metadata")
        _ids(metadata.get("sample_ids", ()))
        if (metadata.get("coordinate_transform_sha256")
                != _json_sha(metadata["coordinate_transform"])):
            raise ValueError("innovation transform identity differs from its fit")
        for key in ("relative_floor", "absolute_floor"):
            value = metadata.get(key)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0
                    or (key == "relative_floor" and value > 1)):
                raise ValueError("invalid declared innovation scale floor")
        self.config, self.fit_metadata = config, metadata
        self.register_buffer("scales", scales)

    @classmethod
    def fit(cls, config, raw_coordinates, *, sample_ids, coordinate_transform,
            split="train", relative_floor=.05, absolute_floor=1.e-6):
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("innovation fit needs the full retained VALUE layout")
        if split != "train" or not isinstance(coordinate_transform, dict):
            raise ValueError("only a declared train-coordinate view can fit normalization")
        for value in (relative_floor, absolute_floor):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("innovation floors must be finite positive numbers")
        if relative_floor > 1:
            raise ValueError("relative floor must not exceed one")
        ids = _ids(sample_ids)
        raw = _values(raw_coordinates, config)
        if raw.dtype != torch.float64:
            raise ValueError("v2 fit requires FP64 raw coordinates before white-model quantization")
        if len(ids) != len(raw):
            raise ValueError("training roots and complete coordinate trajectories differ")
        # Accumulate whole-trajectory sufficient statistics in bounded FP64
        # chunks; production fit must run in its scheduled compute allocation.
        sums = torch.zeros((1 + config.hermite_channels, 2), dtype=torch.float64,
                           device=raw.device)
        for chunk in raw.split(8):
            rho, kinetic = _value_split(chunk.double(), config)
            sums[0, 0] += rho[:, :1].square().sum()
            sums[0, 1] += rho[:, 1:].square().sum()
            sums[1:, 0] += kinetic[:, :, :1].square().sum(dim=(0, 2, 3))
            sums[1:, 1] += kinetic[:, :, 1:].square().sum(dim=(0, 2, 3))
        counts = sums.new_empty(sums.shape)
        counts[0] = sums.new_tensor([len(raw)*config.rho_modes,
            len(raw)*(config.control_count-1)*config.rho_modes])
        counts[1:, 0] = len(raw)*config.kinetic_modes
        counts[1:, 1] = len(raw)*(config.control_count-1)*config.kinetic_modes
        measured = (sums/counts).sqrt()
        if not bool(torch.isfinite(measured).all()):
            raise FloatingPointError("innovation RMS is numerically unrepresentable")
        floor = (measured.max(dim=1, keepdim=True).values*relative_floor).clamp_min(absolute_floor)
        scales = torch.maximum(measured, floor)
        return cls(config, scales, fit_metadata={
            "schema": "vpfp_train_complete_innovation_rms_v2", "split": "train",
            "precision_policy": deepcopy(PRECISION_POLICY),
            "sample_ids": ids, "generated_samples_used_for_fit": False,
            "validation_samples_used_for_fit": False, "centering": "none",
            "roles": ["initial_frame", "remaining_frames"],
            "scale": "whole_train_trajectory_group_RMS_shared_cos_sin",
            "relative_floor": float(relative_floor), "absolute_floor": float(absolute_floor),
            "measured_rms": measured.detach().cpu().tolist(),
            "coordinate_transform": deepcopy(coordinate_transform),
            "coordinate_transform_sha256": _json_sha(coordinate_transform),
            "no_physical_answer_admission": True})

    def _transform(self, values, inverse):
        values = _values(values, self.config).double()
        rho, kinetic = _value_split(values, self.config)
        if self.scales.dtype != torch.float64:
            raise ValueError("v2 normalization scales must remain FP64; dtype conversion is not permitted")
        scales = self.scales.to(device=values.device)
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise FloatingPointError("normalization scale is not representable in input dtype")
        factors = scales if inverse else scales.reciprocal()
        rho_factors = torch.cat((factors[0, :1], factors[0, 1:].expand(self.config.control_count-1)))
        kinetic_factors = torch.cat((factors[1:, :1],
            factors[1:, 1:].expand(-1, self.config.control_count-1)), dim=1)
        result = torch.cat(((rho*rho_factors[None, :, None]).flatten(1),
            (kinetic*kinetic_factors[None, :, :, None]).flatten(1)), dim=1)
        return _values(result, self.config)

    def encode(self, raw_coordinates):
        """FP64 raw coordinates -> FP32 white EDM input, with one explicit cast.

        The model/score objective stays FP32. This quantization is not a
        mathematical coordinate inverse and must be measured separately.
        """
        raw_coordinates = _values(raw_coordinates, self.config)
        if raw_coordinates.dtype != torch.float64:
            raise ValueError("v2 encode requires FP64 raw coordinates before white-model quantization")
        return _values(self._transform(raw_coordinates, False).float(), self.config)

    def decode(self, white_coordinates):
        """White -> FP64 raw coordinates; cannot undo FP32 white quantization."""
        return self._transform(white_coordinates, True)

    def to_payload(self):
        return {"schema": NORMALIZER_SCHEMA, "config": self.config.to_dict(),
            "scales": self.scales.detach().cpu().tolist(),
            "fit_metadata": deepcopy(self.fit_metadata)}

    @classmethod
    def from_payload(cls, payload):
        if (not isinstance(payload, dict)
                or set(payload) != {"schema", "config", "scales", "fit_metadata"}
                or payload["schema"] != NORMALIZER_SCHEMA):
            raise ValueError("unknown or incomplete innovation normalizer payload")
        return cls(VPFPValueEDMConfig.from_dict(payload["config"]), payload["scales"],
                   fit_metadata=payload["fit_metadata"])


class VPFPInnovationConditionalEDM(VPFPMacroConditionalEDM):
    """Unchanged conditional network/EDM loss, explicit new coordinate identity.

    The old native dispatcher deliberately cannot mistake this checkpoint for
    a physical value-state model.  No old z-score checkpoint is reused.
    """
    checkpoint_schema = CHECKPOINT_SCHEMA

    def __init__(self, config, macro_config):
        original_dtype = torch.get_default_dtype()
        try:
            # Legacy physical imports may change the ambient dtype. Learned
            # weights remain FP32 while explicit immutable Fourier buffers
            # retain their source-defined precision.
            torch.set_default_dtype(torch.float32)
            super().__init__(config, macro_config)
        finally:
            torch.set_default_dtype(original_dtype)


def decode_complete_innovation_answer(white, normalizer, transform, *, physical_macro=None):
    """Undo the declared full-dimensional coordinate and apply native lift.

    This returns z's complete native coefficients, not the complete A answer
    by itself: the SAME physical macro and original moving-frame decoder are
    still required. No floor or velocity-space shape is modified here.
    """
    from .temporal_innovation import VPFPTemporalInnovationTransform
    from .bulk_transport_innovation import VPFPBulkTransportInnovationTransform

    if (not isinstance(normalizer, VPFPInnovationNormalizer)
            or not isinstance(transform, (VPFPTemporalInnovationTransform,
                                          VPFPBulkTransportInnovationTransform))
            or transform.config.value_config != normalizer.config
            or _json_sha(transform.to_payload())
                != normalizer.fit_metadata["coordinate_transform_sha256"]):
        raise ValueError("sample decoder differs from the trained innovation coordinate")
    raw = normalizer.decode(white)
    if isinstance(transform, VPFPBulkTransportInnovationTransform):
        values = transform.decode(raw, physical_macro)
    else:
        values = transform.decode(raw)
    # Retain the exact old physical decoder/slope ABI. Promoting the slope
    # lift or decoder would be a separate representation change, not this fix.
    return lift_value_controls(_values(values.float(), normalizer.config), normalizer.config)
