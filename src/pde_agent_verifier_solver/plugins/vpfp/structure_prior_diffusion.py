"""A controlled low-spatial-mode prior for the native VPFP value EDM.

The complete original denoiser is retained. A parallel branch jointly mixes
all Hermite-amplitude channels, restricted to low spatial modes only in the
prior arm. Its parameter-matched control uses identity projections with the
same operations and shapes. Neither arm imposes a source-free PDE, truncates
the sampled state, or treats square-root amplitudes as distribution moments.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from .structured_diffusion import _SpatiotemporalBlock, _coefficients
from .value_control_diffusion import VPFPValueEDMConfig
from .temporal_covariance_diffusion import (
    CHECKPOINT_SCHEMA as TEMPORAL_CHECKPOINT_SCHEMA,
    VPFPTemporalCovarianceEDM, VPFPTemporalCovarianceBatchSampler,
    _sampler_from_temporal_payload,
)


BRANCH_KINDS = ("lowmode_coherent", "branch_identity_control")
CHECKPOINT_SCHEMAS = {
    "lowmode_coherent": "vpfp_lowmode_coherent_diffusion_checkpoint_v1",
    "branch_identity_control": "vpfp_branch_identity_control_diffusion_checkpoint_v1",
}
BRANCH_WIDTH = 64
BRANCH_DILATIONS = (1, 2, 4, 9)
LOW_AXIS_MODE = 2


def _spatial_projector(grid_size, branch_kind):
    """Real orthogonal periodic projector; no learned or data-fitted basis."""
    if branch_kind not in BRANCH_KINDS:
        raise ValueError("unknown VPFP structure branch kind")
    if grid_size < 2 * LOW_AXIS_MODE + 1:
        raise ValueError("the fixed low-mode structure prior requires grid_size >= 5")
    if branch_kind == "branch_identity_control":
        return torch.eye(grid_size, dtype=torch.float32)
    index = torch.arange(grid_size, dtype=torch.float64)
    difference = index[:, None] - index[None, :]
    result = torch.ones(grid_size, grid_size, dtype=torch.float64)
    for mode in range(1, LOW_AXIS_MODE + 1):
        result += 2 * torch.cos(2 * math.pi * mode * difference / grid_size)
    return (result / grid_size).to(torch.float32)


class _JointSpatialBranch(nn.Module):
    def __init__(self, config, branch_kind):
        super().__init__()
        self.register_buffer("spatial_projector", _spatial_projector(config.grid_size, branch_kind))
        self.input = nn.Conv3d(config.field_channels, BRANCH_WIDTH, 1)
        self.context = nn.Sequential(nn.Linear(18, BRANCH_WIDTH), nn.SiLU(),
                                     nn.Linear(BRANCH_WIDTH, BRANCH_WIDTH))
        self.time_position = nn.Linear(3, BRANCH_WIDTH)
        self.blocks = nn.ModuleList(_SpatiotemporalBlock(BRANCH_WIDTH, dilation)
                                    for dilation in BRANCH_DILATIONS)
        self.output = nn.Conv3d(BRANCH_WIDTH, config.field_channels, 1)
        # Only this final affine output is zeroed: no second zero gate can
        # prevent the first optimizer update from activating the branch.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def project(self, field):
        projection = self.spatial_projector.to(field)
        # Keep these same two matrix multiplications in the identity control;
        # its learnable layers and operation shapes are exactly matched.
        return torch.matmul(torch.matmul(projection, field), projection)

    def forward(self, field, sigma, condition):
        frequencies = torch.exp(torch.linspace(0., math.log(100.), 8,
            device=field.device, dtype=field.dtype))
        phase = sigma.log()[:, None] * frequencies[None]
        embedding = self.context(torch.cat((phase.sin(), phase.cos(), condition), dim=1))
        tau = torch.linspace(0., 1., field.shape[2], device=field.device, dtype=field.dtype)
        position = self.time_position(torch.stack((tau, (math.pi*tau).sin(), (math.pi*tau).cos()), dim=1))
        hidden = self.input(self.project(field)) + position.T[None, :, :, None, None]
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.project(self.output(F.silu(hidden)))


class VPFPWeakStructurePriorEDM(VPFPTemporalCovarianceEDM):
    """Unchanged EDM loss/Heun, with an additive learned architecture prior."""

    def __init__(self, config, *, branch_kind="lowmode_coherent"):
        if branch_kind not in BRANCH_KINDS:
            raise ValueError("unknown VPFP structure branch kind")
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("structure prior requires a value-state EDM config")
        # Initialize the original trunk first, in its original RNG order.
        # Resetting the same seed gives the exact old baseline trunk and
        # identical added parameters across the two structure arms.
        super().__init__(config)
        self.branch_kind = branch_kind
        self.checkpoint_schema = CHECKPOINT_SCHEMAS[branch_kind]
        self.structure_branch = _JointSpatialBranch(config, branch_kind)

    def denoise(self, noisy_normalized_coefficients, sigma, conditions):
        original = super().denoise(noisy_normalized_coefficients, sigma, conditions)
        parameter = next(self.parameters())
        noisy = _coefficients(noisy_normalized_coefficients, self.config,
            device=parameter.device, dtype=parameter.dtype)
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        condition = self._conditions(conditions, len(noisy), noisy)
        denominator = torch.sqrt(sigma.square() + self.config.sigma_data**2)
        field = self.transform.to_field(noisy / denominator[:, None])
        residual = self.transform.to_coefficients(self.structure_branch(field, sigma, condition))
        c_out = sigma * self.config.sigma_data / denominator
        return original + c_out[:, None] * residual

    def architecture_report(self):
        branch_parameters = sum(value.numel() for value in self.structure_branch.parameters())
        total_parameters = sum(value.numel() for value in self.parameters())
        return {
            "schema": "vpfp_low_spatial_mode_joint_amplitude_branch_v1",
            "checkpoint_schema": self.checkpoint_schema, "branch_kind": self.branch_kind,
            "branch_width": BRANCH_WIDTH, "branch_depth": len(BRANCH_DILATIONS),
            "branch_temporal_dilations": list(BRANCH_DILATIONS),
            "low_axis_mode": LOW_AXIS_MODE, "grid_size": self.config.grid_size,
            "projector": "cartesian_low_fourier_modes" if self.branch_kind == "lowmode_coherent" else "identity",
            "projection_locations": ["branch_input", "branch_output"],
            "parameter_matched_operation_shapes": True,
            "unchanged_full_support_trunk": True, "all_Hermite_amplitude_channels_retained": True,
            "rho_channel_jointly_mixed": True, "coordinate_system": "model_normalized_value_coordinates",
            "zero_initialization": "branch_output_affine_only_no_additional_zero_gate",
            "base_parameters": total_parameters - branch_parameters,
            "branch_parameters": branch_parameters, "total_parameters": total_parameters,
            "objective": "unchanged_EDM_denoising_score_matching",
            "sampling": "unchanged_full_value_state_Heun_no_state_projection",
            "physical_derivative_distribution_loss": False,
            "exact_Hermite_transport_physics_imposed": False, "source_free_PDE_imposed": False,
            "scientific_admission": False,
        }


def make_structure_prior_model(config, branch_kind="lowmode_coherent"):
    """Factory for the existing shared trainer's model_factory hook."""
    return VPFPWeakStructurePriorEDM(config, branch_kind=branch_kind)


class VPFPWeakStructurePriorBatchSampler(VPFPTemporalCovarianceBatchSampler):
    def __init__(self, model, normalizer, **kwargs):
        if not isinstance(model, VPFPWeakStructurePriorEDM):
            raise TypeError("structure sampler requires its native architecture")
        super().__init__(model, normalizer, **kwargs)
        architecture = model.architecture_report()
        encoded = json.dumps(architecture, allow_nan=False, sort_keys=True, separators=(",", ":"))
        self.identity.update(kind="weak_structure_prior_value_control_conditional_edm",
            checkpoint_schema=model.checkpoint_schema, architecture=architecture,
            architecture_sha256=sha256(encoded.encode()).hexdigest())


def _sampler_from_structure_payload(payload, expected_sha256, device, *, steps, churn):
    """Validate the fixed architecture in addition to the native temporal asset."""
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != fields or payload.get("schema") not in CHECKPOINT_SCHEMAS.values():
        raise ValueError("unexpected structure-prior checkpoint schema/fields")
    kind = next(kind for kind, schema in CHECKPOINT_SCHEMAS.items() if schema == payload["schema"])
    model = make_structure_prior_model(VPFPValueEDMConfig.from_dict(payload["config"]), kind)
    state, expected = payload["model"], model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("structure-prior checkpoint model tensor keys differ")
    for name, prototype in expected.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != prototype.shape or tensor.dtype != prototype.dtype
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"invalid structure-prior checkpoint tensor: {name}")
        if (name.startswith("transform.") or name == "structure_branch.spatial_projector") and not torch.equal(tensor, prototype):
            raise ValueError(f"checkpoint modified a fixed structure/Fourier projector: {name}")
    # This strictly checked in-memory view delegates existing config, source,
    # training-JSON, Fourier-trunk, and train-only normalizer provenance checks.
    # All new branch tensors were checked above; none are silently discarded.
    temporal_view = dict(payload, schema=TEMPORAL_CHECKPOINT_SCHEMA,
        model={name: tensor for name, tensor in state.items() if not name.startswith("structure_branch.")})
    checked = _sampler_from_temporal_payload(temporal_view, expected_sha256, device, steps=steps, churn=churn)
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    sampler = VPFPWeakStructurePriorBatchSampler(model, checked.normalizer,
        base_decoder_sha256=checked.identity["base_decoder_sha256"], checkpoint_sha256=expected_sha256,
        steps=steps, churn=churn)
    sampler.identity.update(corpus_manifest_sha256=checked.identity["corpus_manifest_sha256"],
        training=deepcopy(checked.identity["training"]))
    return sampler


__all__ = ["BRANCH_KINDS", "CHECKPOINT_SCHEMAS", "VPFPWeakStructurePriorEDM",
           "VPFPWeakStructurePriorBatchSampler", "make_structure_prior_model"]
