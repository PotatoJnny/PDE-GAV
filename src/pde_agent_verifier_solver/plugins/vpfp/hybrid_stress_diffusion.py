"""Full-rank persistent/dynamic DC stress coordinates for a weak-VPFP pilot.

Only global spatial-DC log-Cholesky coordinates change. A fixed orthogonal
1+32 time split separates centered trajectory mean from zero-mean variation;
train-only amplitude RMS and regularized residual covariance define an
invertible affine map. White coordinates are repacked onto all 33 time slots,
so the original joint EDM, isotropic white noise and coordinate metric apply.
No derivative penalty, temporal truncation, output averaging or physics claim.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path

# Preserve the frozen macro assets' NumPy-before-Torch QR initialization path.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .coherent_velocity_diffusion import VPFPVelocityMacroConfig, json_sha
from .stress_block_diffusion import (
    VPFPStressBlockMacroEDM, VPFPStressBlockMacroNormalizer, _array_digest, _digest)
from .structured_diffusion import _coefficients

H1_SCHEMA = "vpfp_hybrid_stress_macro_diffusion_checkpoint_v1"
H2_SCHEMA = "vpfp_hybrid_stress_joint_macro_diffusion_checkpoint_v1"
NORMALIZER_SCHEMA = "vpfp_hybrid_stress_macro_normalizer_v1"
RESIDUAL_SHRINKAGE = 0.01


def hybrid_time_basis(count=33):
    """Deterministic Helmert columns: positive constant, then its complement.

    Column order is an algebraic basis, not a frequency ordering. No QR,
    fitting or dimensionality reduction is involved. Return NumPy float64.
    """
    if type(count) is not int or count != 33:
        raise ValueError("the frozen hybrid pilot retains exactly 33 time coordinates")
    basis = np.zeros((count, count), dtype=np.float64)
    basis[:, 0] = 1 / math.sqrt(count)
    for column in range(1, count):
        scale = math.sqrt(column * (column + 1))
        basis[:column, column] = 1 / scale
        basis[column, column] = -column / scale
    return basis


class VPFPHybridStressMacroNormalizer(nn.Module):
    """Old non-DC block scaling, new full-rank global stress time geometry.

    ``a`` is sqrt(33) times the *centered log-coordinate* temporal mean, not
    mean physical stress. The original time-dependent TRAIN center is retained.
    Residual covariance uses the empirical second moment (population divisor
    N), 1% trace-isotropic shrinkage and the original absolute floor squared.
    There is no mean/residual statistical independence assumption.
    """
    schema = NORMALIZER_SCHEMA

    def __init__(self, config, block, amplitude_rms, residual_covariance,
                 residual_cholesky, fit_metadata):
        super().__init__()
        if not isinstance(config, VPFPVelocityMacroConfig) or not isinstance(block, VPFPStressBlockMacroNormalizer) or block.config != config:
            raise ValueError("bound original config and train-only stress block normalizer required")
        self.config = config
        self.block = VPFPStressBlockMacroNormalizer.from_payload(block.to_payload())
        metadata = deepcopy(fit_metadata)
        if (set(metadata) != set(block.fit_metadata) | {"hybrid_stress_fit"}
                or any(metadata[key] != value for key, value in block.fit_metadata.items())):
            raise ValueError("hybrid fit must preserve the original training roots and block geometry")
        floor = float(block.pooled.parent.fit_metadata["absolute_scale_floor"])
        expected_detail = {
            "schema": "vpfp_train_only_global_stress_mean_residual_covariance_v1",
            "basis": "deterministic_Helmert_positive_constant_then_32_zero_mean_columns",
            "control_count": 33, "persistent_dimension_per_channel": 1,
            "dynamic_dimension_per_channel": 32, "stress_channels": 3,
            "center_rule": "unchanged_original_TRAIN_time_dependent_DC_center",
            "persistent_scale_rule": "RMS_of_sqrt33_times_centered_temporal_mean",
            "persistent_scale_floor": math.sqrt(33) * floor,
            "residual_covariance_rule": "population_second_moment_divisor_N",
            "residual_regularization": "(1-alpha)*C+alpha*trace(C)/32*I+absolute_floor_squared*I",
            "residual_shrinkage": RESIDUAL_SHRINKAGE,
            "absolute_residual_scale_floor": floor,
            "block_normalizer_sha256": json_sha(block.to_payload()),
            "training_values_sha256": block.fit_metadata["stress_block_fit"]["training_values_sha256"],
            "fit_row_count": len(block.fit_metadata["sample_ids"]),
            "white_layout": "orthogonal_repacking_to_all_original_33_time_slots",
            "all_coordinates_retained": True, "nonDC_and_velocity_map_unchanged": True,
            "statistical_mean_residual_independence_assumed": False,
            "normalization_changes_physical_noise_covariance": True,
            "source_reasonableness_validated": False}
        if metadata["hybrid_stress_fit"] != expected_detail:
            raise ValueError("invalid frozen hybrid fit provenance or regularization recipe")
        rms = torch.as_tensor(amplitude_rms, dtype=torch.float64)
        covariance = torch.as_tensor(residual_covariance, dtype=torch.float64)
        factor = torch.as_tensor(residual_cholesky, dtype=torch.float64)
        if (rms.shape != (3,) or covariance.shape != (3, 32, 32) or factor.shape != covariance.shape
                or not all(torch.isfinite(value).all() for value in (rms, covariance, factor))
                or (rms < 0).any() or (factor.diagonal(dim1=-2, dim2=-1) <= 0).any()
                or torch.count_nonzero(torch.triu(factor, diagonal=1))):
            raise ValueError("finite amplitude RMS and positive lower-triangular residual factors required")
        # These are algebraic consistency checks on stored TRAIN-fit factors,
        # not blanket tolerance for immutable model buffers. Stored factors
        # remain authoritative and are never regenerated during load.
        scale = max(float(covariance.abs().max()), floor * floor)
        tolerance = 512 * torch.finfo(torch.float64).eps * scale
        if (not torch.allclose(covariance, covariance.transpose(-1, -2), rtol=0., atol=tolerance)
                or float(torch.linalg.eigvalsh(covariance).min()) < -tolerance):
            raise ValueError("residual training covariance must be symmetric positive semidefinite")
        regularized = self._regularize(covariance, floor)
        if not torch.allclose(factor @ factor.transpose(-1, -2), regularized, rtol=0., atol=tolerance):
            raise ValueError("stored residual factor differs from the frozen full-rank regularization")
        self.register_buffer("time_basis", torch.from_numpy(hybrid_time_basis()).clone())
        self.register_buffer("amplitude_rms", rms.clone())
        self.register_buffer("amplitude_scales", rms.clamp_min(math.sqrt(33) * floor).clone())
        self.register_buffer("residual_covariance", covariance.clone())
        self.register_buffer("residual_cholesky", factor.clone())
        self.fit_metadata = metadata

    @staticmethod
    def _regularize(covariance, floor):
        identity = torch.eye(32, device=covariance.device, dtype=covariance.dtype)
        variance = covariance.diagonal(dim1=-2, dim2=-1).mean(-1)
        return ((1 - RESIDUAL_SHRINKAGE) * covariance
                + (RESIDUAL_SHRINKAGE * variance + floor * floor)[:, None, None] * identity)

    @classmethod
    def fit(cls, config, values, *, sample_ids, parent_normalizer, split="train"):
        values = _coefficients(values, config)
        block = VPFPStressBlockMacroNormalizer.fit(config, values, sample_ids=sample_ids,
            parent_normalizer=parent_normalizer, split=split)
        center = block.pooled.parent.center[config.base_macro_config.rho_dimension:].to(values.device)
        centered = (values[:, config.rho_dimension:].double() - center.double()).reshape(-1, 3, 33, config.modes)
        basis = torch.from_numpy(hybrid_time_basis()).to(values.device)
        coordinates = centered[..., 0] @ basis
        amplitude_rms = coordinates[..., 0].square().mean(0).sqrt()
        residual = coordinates[..., 1:]
        covariance = torch.einsum("bci,bcj->cij", residual, residual) / len(values)
        floor = float(block.pooled.parent.fit_metadata["absolute_scale_floor"])
        factor = torch.linalg.cholesky(cls._regularize(covariance, floor))
        # Keep the small canonical metadata recipe adjacent to the validator;
        # no fit is performed by the payload loader.
        detail = {
            "schema": "vpfp_train_only_global_stress_mean_residual_covariance_v1",
            "basis": "deterministic_Helmert_positive_constant_then_32_zero_mean_columns",
            "control_count": 33, "persistent_dimension_per_channel": 1,
            "dynamic_dimension_per_channel": 32, "stress_channels": 3,
            "center_rule": "unchanged_original_TRAIN_time_dependent_DC_center",
            "persistent_scale_rule": "RMS_of_sqrt33_times_centered_temporal_mean",
            "persistent_scale_floor": math.sqrt(33) * floor,
            "residual_covariance_rule": "population_second_moment_divisor_N",
            "residual_regularization": "(1-alpha)*C+alpha*trace(C)/32*I+absolute_floor_squared*I",
            "residual_shrinkage": RESIDUAL_SHRINKAGE,
            "absolute_residual_scale_floor": floor,
            "block_normalizer_sha256": json_sha(block.to_payload()),
            "training_values_sha256": _array_digest(values), "fit_row_count": len(values),
            "white_layout": "orthogonal_repacking_to_all_original_33_time_slots",
            "all_coordinates_retained": True, "nonDC_and_velocity_map_unchanged": True,
            "statistical_mean_residual_independence_assumed": False,
            "normalization_changes_physical_noise_covariance": True,
            "source_reasonableness_validated": False}
        return cls(config, block, amplitude_rms.cpu(), covariance.cpu(), factor.cpu(),
                   {**deepcopy(block.fit_metadata), "hybrid_stress_fit": detail})

    @property
    def pooled(self): return self.block.pooled

    @property
    def stress_scales(self): return self.block.stress_scales

    @property
    def stress_rms(self): return self.block.stress_rms

    def _transform(self, values, inverse):
        values = _coefficients(values, self.config)
        start, modes = self.config.rho_dimension, self.config.modes
        output = self.block.decode(values) if inverse else self.block.encode(values)
        stress = values[:, start:].reshape(-1, 3, 33, modes)
        center = self.block.pooled.parent.center[self.config.base_macro_config.rho_dimension:].to(values).reshape(3, 33, modes)[..., 0]
        basis, factor = self.time_basis.to(values), self.residual_cholesky.to(values)
        dc = stress[..., 0] if inverse else stress[..., 0] - center
        coordinates = dc @ basis
        if inverse:
            amplitude = coordinates[..., :1] * self.amplitude_scales.to(values)[None, :, None]
            residual = torch.einsum("cij,bcj->bci", factor, coordinates[..., 1:])
        else:
            amplitude = coordinates[..., :1] / self.amplitude_scales.to(values)[None, :, None]
            residual = torch.linalg.solve_triangular(factor[None], coordinates[..., 1:, None], upper=False).squeeze(-1)
        dc = torch.cat((amplitude, residual), dim=-1) @ basis.T
        if inverse:
            dc = dc + center
        # Clone avoids mutating caller coordinates or any old normalizer state.
        result_stress = output[:, start:].reshape(-1, 3, 33, modes).clone()
        result_stress[..., 0] = dc
        return torch.cat((output[:, :start], result_stress.flatten(1)), dim=1)

    def encode(self, values): return self._transform(values, False)
    def decode(self, values): return self._transform(values, True)

    def geometry_report(self):
        """Small TRAIN geometry diagnostics, never a candidate-derived gate."""
        covariance = self.residual_covariance.detach().cpu()
        factor = self.residual_cholesky.detach().cpu()
        empirical = torch.linalg.eigvalsh(covariance)
        regularized = torch.linalg.eigvalsh(factor @ factor.transpose(-1, -2))
        white_left = torch.linalg.solve_triangular(factor, covariance, upper=False)
        white = torch.linalg.solve_triangular(factor, white_left.transpose(-1, -2), upper=False).transpose(-1, -2)
        return {"schema": "vpfp_hybrid_stress_TRAIN_geometry_report_v1",
            "training_values_sha256": self.fit_metadata["hybrid_stress_fit"]["training_values_sha256"],
            "training_sample_ids": list(self.fit_metadata["sample_ids"]),
            "fit_row_count": len(self.fit_metadata["sample_ids"]),
            "residual_shrinkage": RESIDUAL_SHRINKAGE,
            "absolute_residual_scale_floor": self.fit_metadata["hybrid_stress_fit"]["absolute_residual_scale_floor"],
            "persistent_RMS": self.amplitude_rms.detach().cpu().tolist(),
            "persistent_scales": self.amplitude_scales.detach().cpu().tolist(),
            "empirical_residual_covariance_eigenvalues": empirical.tolist(),
            "regularized_residual_covariance_eigenvalues": regularized.tolist(),
            "regularized_covariance_condition_numbers": (regularized[:, -1] / regularized[:, 0]).tolist(),
            "empirical_white_residual_coordinate_variances": white.diagonal(dim1=-2, dim2=-1).tolist(),
            "mean_residual_statistical_independence_claimed": False,
            "scientific_qualification": False}

    def to_payload(self):
        data = {"schema": self.schema, "config": self.config.to_dict(), "block": self.block.to_payload(),
            "amplitude_rms": self.amplitude_rms.detach().cpu().tolist(),
            "amplitude_scales": self.amplitude_scales.detach().cpu().tolist(),
            "residual_covariance": self.residual_covariance.detach().cpu().tolist(),
            "residual_cholesky": self.residual_cholesky.detach().cpu().tolist(),
            "fit_metadata": deepcopy(self.fit_metadata)}
        return {**data, "sha256": json_sha(data)}

    @classmethod
    def from_payload(cls, payload):
        fields = {"schema", "config", "block", "amplitude_rms", "amplitude_scales",
                  "residual_covariance", "residual_cholesky", "fit_metadata", "sha256"}
        if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != cls.schema:
            raise ValueError("exact hybrid normalizer schema and fields required")
        if json_sha({key: value for key, value in payload.items() if key != "sha256"}) != _digest(payload["sha256"]):
            raise ValueError("hybrid normalizer payload SHA differs")
        result = cls(VPFPVelocityMacroConfig.from_dict(payload["config"]),
            VPFPStressBlockMacroNormalizer.from_payload(payload["block"]), payload["amplitude_rms"],
            payload["residual_covariance"], payload["residual_cholesky"], payload["fit_metadata"])
        if result.to_payload() != payload:
            raise ValueError("hybrid payload scales or fitted factors are not canonical")
        return result


class VPFPHybridStressMacroEDM(VPFPStressBlockMacroEDM):
    """H1: exactly the original network and EDM, with the hybrid normalizer."""
    checkpoint_schema = H1_SCHEMA

    def architecture_report(self):
        return {**super().architecture_report(), "schema": "vpfp_hybrid_stress_H1_architecture_v1",
            "hybrid_arm": "H1", "global_stress_time_dimensions": [1, 32],
            "additional_parameters": 0, "same_backbone_and_EDM_as_stress_block": True,
            "all_33_time_coordinates_retained": True, "source_reasonableness_validated": False}


class VPFPHybridStressJointMacroEDM(VPFPHybridStressMacroEDM):
    """H2: shared score trunk plus zero-initialized persistent/dynamic heads.

    Both heads read the same noisy joint state and sigma/nu/T embedding. The
    persistent correction is constant in time; the dynamic correction is
    zero-mean in time. They augment, not replace, the original DC score, making
    zero-head H2 exactly compatible with any transferred H1 backbone. This is
    a full joint sample, not independent amplitude and dynamics generators.
    """
    checkpoint_schema = H2_SCHEMA

    def __init__(self, config):
        super().__init__(config)
        width = config.width
        self.persistent_head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 3))
        self.dynamic_head = nn.Sequential(nn.Conv1d(3 * width, width, 1), nn.SiLU(), nn.Conv1d(width, 3, 1))
        for layer in (self.persistent_head[-1], self.dynamic_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _dc_corrections(self, hidden, embedding):
        local = hidden.mean(dim=(-1, -2))
        persistent = local.mean(dim=-1)
        a = self.persistent_head(torch.cat((persistent, embedding), dim=1))
        inputs = torch.cat((local, persistent[..., None].expand_as(local), embedding[..., None].expand_as(local)), dim=1)
        dynamic = self.dynamic_head(inputs)
        dynamic = dynamic - dynamic.mean(dim=-1, keepdim=True)
        return a, dynamic

    def denoise(self, noisy_normalized_coefficients, sigma, conditions):
        # Same operation order as the old denoiser up to its predicted score.
        parameter = next(self.parameters())
        noisy = _coefficients(noisy_normalized_coefficients, self.config, device=parameter.device, dtype=parameter.dtype)
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype)
        if sigma.shape != (len(noisy),) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("sigma must be finite and positive with shape [B]")
        condition = self._conditions(conditions, len(noisy), noisy)
        denominator = torch.sqrt(sigma.square() + self.config.sigma_data ** 2)
        c_skip = self.config.sigma_data ** 2 / denominator.square()
        c_out = sigma * self.config.sigma_data / denominator
        field = self.transform.to_field(noisy / denominator[:, None])
        frequencies = torch.exp(torch.linspace(0., math.log(100.), 8, device=noisy.device, dtype=noisy.dtype))
        phase = sigma.log()[:, None] * frequencies[None]
        embedding = self.context(torch.cat((phase.sin(), phase.cos(), condition), dim=1))
        tau = torch.linspace(0., 1., self.config.control_count, device=noisy.device, dtype=noisy.dtype)
        position = self.time_position(torch.stack((tau, (math.pi * tau).sin(), (math.pi * tau).cos()), dim=1))
        hidden = self.input(field) + position.T[None, :, :, None, None]
        for block in self.blocks:
            hidden = block(hidden, embedding)
        activated = F.silu(hidden)
        residual = self.transform.to_coefficients(self.output(activated))
        amplitude, dynamic = self._dc_corrections(activated, embedding)
        start = self.config.rho_dimension
        stress = residual[:, start:].reshape(-1, 3, 33, self.config.modes).clone()
        stress[..., 0] = stress[..., 0] + amplitude[..., None] + dynamic
        residual = torch.cat((residual[:, :start], stress.flatten(1)), dim=1)
        return c_skip[:, None] * noisy + c_out[:, None] * residual

    def architecture_report(self):
        additional = sum(p.numel() for head in (self.persistent_head, self.dynamic_head) for p in head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {**super().architecture_report(), "schema": "vpfp_hybrid_stress_H2_architecture_v1",
            "hybrid_arm": "H2", "parameters": total, "backbone_parameters": total - additional,
            "additional_parameters": additional, "same_backbone_and_EDM_as_stress_block": False,
            "same_original_backbone_parameters": True, "additive_zero_initialized_DC_heads": True,
            "head_outputs": ["persistent_constant_time", "dynamic_zero_temporal_mean"],
            "shared_noisy_state_and_noise_condition_embedding": True,
            "independent_persistent_dynamic_sampling": False, "equal_compute_claimed": False}


def load_hybrid_stress_macro_checkpoint(path, expected_sha256, *, device="cpu"):
    """Strict H1/H2 seven-field loader; stored fit factors stay authoritative."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != _digest(expected_sha256):
        raise ValueError("hybrid stress checkpoint file SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    factories = {H1_SCHEMA: VPFPHybridStressMacroEDM, H2_SCHEMA: VPFPHybridStressJointMacroEDM}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] not in factories:
        raise ValueError("exact seven-field H1 or H2 hybrid checkpoint required")
    config = VPFPVelocityMacroConfig.from_dict(payload["config"])
    normalizer = VPFPHybridStressMacroNormalizer.from_payload(payload["normalizer"])
    if normalizer.config != config:
        raise ValueError("hybrid model and normalizer configurations differ")
    for name in ("base_decoder_sha256", "corpus_manifest_sha256"):
        _digest(payload[name])
    training = payload["training"]
    if (not isinstance(training, dict) or training.get("normalizer_sha256") != json_sha(payload["normalizer"])
            or not isinstance(training.get("train_case_ids"), list)
            or len(training["train_case_ids"]) != len(normalizer.fit_metadata["sample_ids"])
            or any(type(v) is not str or not v for v in training["train_case_ids"])
            or len(set(training["train_case_ids"])) != len(training["train_case_ids"])):
        raise ValueError("hybrid checkpoint normalizer and training root binding differs")
    model = factories[payload["schema"]](config)
    expected, buffers = model.state_dict(), dict(model.named_buffers())
    if not isinstance(payload["model"], dict) or set(payload["model"]) != set(expected):
        raise ValueError("hybrid model tensor inventory differs")
    for name, value in payload["model"].items():
        if (not torch.is_tensor(value) or value.layout != torch.strided or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError("invalid hybrid model weight or fixed buffer: " + name)
        if name in buffers and not torch.equal(value, expected[name]):
            raise ValueError("immutable hybrid model buffer differs: " + name)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), normalizer.to(device), payload
