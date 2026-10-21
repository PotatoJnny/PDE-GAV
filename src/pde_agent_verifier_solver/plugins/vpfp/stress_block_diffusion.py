"""One-variable weak-macro EDM representation experiment, research only.

Compare the old pooled stress coordinates with train-only DC/coherent/rest
stress scales. Keep every log-Cholesky coefficient and its original center;
U0, polynomial drive, local velocity, network, noise law and EDM loss are the
old ones. Different physical noise covariances are intentional, not a claim
that validation losses across the two coordinates are comparable.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .coherent_velocity_diffusion import (
    VPFPVelocityMacroConfig, VPFPVelocityMacroEDM, VPFPVelocityMacroNormalizer, json_sha)
from .structured_diffusion import _coefficients

SCHEMA = "vpfp_stress_block_velocity_macro_diffusion_checkpoint_v1"
NORMALIZER_SCHEMA = "vpfp_stress_block_velocity_macro_normalizer_v1"
BLOCK_NAMES = ("DC", "coherent_nonDC_k_le_2", "retained_k_gt_2")


def stress_block_masks(config):
    """Three disjoint NumPy bool masks in the original real Fourier order.

    Cos/sin partners always share a block. max(|kx|,|ky|) agrees with the
    existing coherent mask; no mode is removed, fitted or reordered.
    """
    if not isinstance(config, VPFPVelocityMacroConfig) or config.maximum_mode < 3:
        raise ValueError("original velocity macro config with all three nonempty stress blocks required")
    modes = [0]
    maximum = config.maximum_mode
    for kx in range(-maximum, maximum+1):
        for ky in range(-maximum, maximum+1):
            if kx > 0 or (kx == 0 and ky > 0):
                modes.extend([max(abs(kx), abs(ky))]*2)
    modes = np.asarray(modes, dtype=np.int64)
    return dict(zip(BLOCK_NAMES, (modes == 0, (modes > 0) & (modes <= 2), modes > 2)))


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA256 required")
    return value


def _array_digest(values):
    array = values.detach().cpu().contiguous().numpy()
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return sha256(header+b"\0"+array.tobytes()).hexdigest()


class VPFPStressBlockMacroNormalizer(nn.Module):
    """Wrap the exact pooled normalizer; override only stress block scales.

    The absolute floor is taken from the original parent normalizer's frozen
    fit metadata (currently 1e-5), not retuned from validation or generation.
    Zero-variance blocks remain full-dimensional because every scale is >0.
    """
    schema = NORMALIZER_SCHEMA

    def __init__(self, config, pooled, stress_rms, fit_metadata):
        super().__init__()
        masks = stress_block_masks(config)
        if not isinstance(pooled, VPFPVelocityMacroNormalizer) or pooled.config != config:
            raise ValueError("bound original pooled velocity normalizer required")
        # No caller-owned mutable module/buffer is retained by the new arm.
        self.pooled = VPFPVelocityMacroNormalizer.from_payload(pooled.to_payload())
        self.config = config
        metadata = deepcopy(fit_metadata)
        if set(metadata) != set(self.pooled.fit_metadata) | {"stress_block_fit"}:
            raise ValueError("exact original training metadata plus stress block fit required")
        if any(metadata[k] != value for k, value in self.pooled.fit_metadata.items()):
            raise ValueError("stress normalization must retain original train root IDs and centering metadata")
        detail = metadata["stress_block_fit"]
        expected = {"schema", "block_names", "mode_counts", "absolute_scale_floor", "training_values_sha256",
            "original_pooled_payload_sha256", "fit_row_count", "center_rule", "cos_sin_scales_shared",
            "all_coordinates_retained", "normalization_changes_physical_noise_covariance"}
        floor = self.pooled.parent.fit_metadata.get("absolute_scale_floor")
        if (not isinstance(detail, dict) or set(detail) != expected
                or detail["schema"] != "vpfp_train_only_stress_spectral_RMS_fit_v1"
                or detail["block_names"] != list(BLOCK_NAMES)
                or detail["mode_counts"] != [int(mask.sum()) for mask in masks.values()]
                or isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0
                or detail["absolute_scale_floor"] != floor
                or detail["original_pooled_payload_sha256"] != json_sha(self.pooled.to_payload())
                or type(detail["fit_row_count"]) is not int or detail["fit_row_count"] < 2
                or detail["fit_row_count"] != len(self.pooled.fit_metadata["sample_ids"])
                or detail["center_rule"] != "exact_original_time_dependent_DC_center; no_nonDC_centering"
                or any(detail[k] is not True for k in ("cos_sin_scales_shared", "all_coordinates_retained", "normalization_changes_physical_noise_covariance"))):
            raise ValueError("invalid frozen train-only stress block geometry/floor/provenance")
        _digest(detail["training_values_sha256"])
        rms = torch.as_tensor(stress_rms, dtype=torch.float64)
        if rms.shape != (3, 3) or not torch.isfinite(rms).all() or (rms < 0).any():
            raise ValueError("finite nonnegative per-channel/per-block training RMS required")
        scales = rms.clamp_min(float(floor))
        block_indices = np.zeros(config.modes, dtype=np.int64)
        for index, mask in enumerate(masks.values()):
            block_indices[mask] = index
        self.register_buffer("stress_rms", rms.clone())
        self.register_buffer("stress_scales", scales.clone())
        self.register_buffer("stress_mode_blocks", torch.as_tensor(block_indices, dtype=torch.long))
        self.fit_metadata = metadata

    @classmethod
    def fit(cls, config, values, *, sample_ids, parent_normalizer, split="train"):
        masks = stress_block_masks(config)
        values = _coefficients(values, config)
        ids = list(sample_ids)
        if (split != "train" or len(values) < 2 or len(ids) != len(values)
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in ids)
                or len(set(map(int, ids))) != len(ids)):
            raise ValueError("distinct integer IDs of at least two training roots required; no validation fit")
        ids = list(map(int, ids))
        pooled = VPFPVelocityMacroNormalizer.fit(config, values, sample_ids=ids,
            parent_normalizer=parent_normalizer, split=split)
        centered = values[:, config.rho_dimension:].double() - pooled.parent.center[
            config.base_macro_config.rho_dimension:].to(device=values.device, dtype=torch.float64)
        stress = centered.reshape(len(values), 3, config.control_count, config.modes)
        rms = torch.stack([stress[..., torch.as_tensor(mask, device=values.device)].square().mean(dim=(0, 2, 3)).sqrt()
            for mask in masks.values()], dim=1)
        metadata = {**deepcopy(pooled.fit_metadata), "stress_block_fit": {
            "schema": "vpfp_train_only_stress_spectral_RMS_fit_v1", "block_names": list(BLOCK_NAMES),
            "mode_counts": [int(mask.sum()) for mask in masks.values()],
            "absolute_scale_floor": pooled.parent.fit_metadata["absolute_scale_floor"],
            "training_values_sha256": _array_digest(values),
            "original_pooled_payload_sha256": json_sha(pooled.to_payload()), "fit_row_count": len(values),
            "center_rule": "exact_original_time_dependent_DC_center; no_nonDC_centering",
            "cos_sin_scales_shared": True, "all_coordinates_retained": True,
            "normalization_changes_physical_noise_covariance": True}}
        return cls(config, pooled, rms, metadata)

    def _transform(self, values, inverse):
        values = _coefficients(values, self.config)
        # Calling the old normalizer for the prefix retains its exact operation
        # order and bits for U0, drive and local velocity in both directions.
        original = self.pooled.decode(values) if inverse else self.pooled.encode(values)
        start = self.config.rho_dimension
        center = self.pooled.parent.center[self.config.base_macro_config.rho_dimension:].to(values)
        scales = self.stress_scales.to(values).index_select(-1, self.stress_mode_blocks.to(values.device))
        scales = scales[:, None, :].expand(3, self.config.control_count, self.config.modes).reshape(1, -1)
        stress = values[:, start:]*scales+center if inverse else (values[:, start:]-center)/scales
        return torch.cat((original[:, :start], stress), dim=1)

    def encode(self, values): return self._transform(values, False)
    def decode(self, values): return self._transform(values, True)

    def to_payload(self):
        data = {"schema": self.schema, "config": self.config.to_dict(), "pooled": self.pooled.to_payload(),
            "stress_rms": self.stress_rms.detach().cpu().tolist(),
            "stress_scales": self.stress_scales.detach().cpu().tolist(), "fit_metadata": deepcopy(self.fit_metadata)}
        return {**data, "sha256": json_sha(data)}

    @classmethod
    def from_payload(cls, payload):
        fields = {"schema", "config", "pooled", "stress_rms", "stress_scales", "fit_metadata", "sha256"}
        if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != cls.schema:
            raise ValueError("exact stress block normalizer schema and fields required")
        data = {k: value for k, value in payload.items() if k != "sha256"}
        if json_sha(data) != _digest(payload["sha256"]):
            raise ValueError("stress block normalizer payload SHA differs")
        result = cls(VPFPVelocityMacroConfig.from_dict(payload["config"]),
            VPFPVelocityMacroNormalizer.from_payload(payload["pooled"]), payload["stress_rms"], payload["fit_metadata"])
        if result.to_payload() != payload:
            raise ValueError("stress block scales differ from the frozen RMS/floor or noncanonical payload")
        return result


class VPFPStressBlockMacroEDM(VPFPVelocityMacroEDM):
    """Identical parameters, initialization, denoiser and objective; new ABI."""
    checkpoint_schema = SCHEMA


def load_stress_block_macro_checkpoint(path, expected_sha256, *, device="cpu"):
    """Strict seven-field research loader; no old-score reinterpretation.

    Fixed-buffer portability checks are deliberately not blanket allclose.
    Until an individually justified portability validator is supplied, all
    fixed buffers must match their analytical construction exactly.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != _digest(expected_sha256):
        raise ValueError("stress block checkpoint file SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != SCHEMA:
        raise ValueError("exact seven-field stress block checkpoint required")
    config = VPFPVelocityMacroConfig.from_dict(payload["config"])
    normalizer = VPFPStressBlockMacroNormalizer.from_payload(payload["normalizer"])
    if normalizer.config != config:
        raise ValueError("stress block model and normalizer configurations differ")
    for name in ("base_decoder_sha256", "corpus_manifest_sha256"):
        _digest(payload[name])
    training = payload["training"]
    if (not isinstance(training, dict) or training.get("normalizer_sha256") != json_sha(payload["normalizer"])
            or not isinstance(training.get("train_case_ids"), list)
            or len(training["train_case_ids"]) != len(normalizer.fit_metadata["sample_ids"])
            or any(type(v) is not str or not v for v in training["train_case_ids"])
            or len(set(training["train_case_ids"])) != len(training["train_case_ids"])):
        raise ValueError("stress block checkpoint normalizer and training root binding differs")
    model = VPFPStressBlockMacroEDM(config)
    expected, buffers = model.state_dict(), dict(model.named_buffers())
    if not isinstance(payload["model"], dict) or set(payload["model"]) != set(expected):
        raise ValueError("stress block model tensor inventory differs")
    for name, value in payload["model"].items():
        if (not torch.is_tensor(value) or value.layout != torch.strided
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError("invalid stress block weight or fixed buffer: " + name)
        if name in buffers and not torch.equal(value, expected[name]):
            raise ValueError("immutable stress block model buffer differs: " + name)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), normalizer.to(device), payload
