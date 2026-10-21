"""Research macro EDM with driven-U coordinates, never an admitted asset.

U0 and nonperiodic Legendre acceleration coefficients are learned jointly
with the unchanged local velocity and log-Cholesky fields. The score lives
in these genuine reduced coordinates; no noisy full state is projected.
The fixed network packing is not the physical OU inverse or a PDE penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .coherent_diffusion import (VPFPCoherentMacroConfig, VPFPCoherentMacroEDM,
    VPFPCoherentMacroTransform, VPFPCoherentMacroNormalizer)
from .coherent_velocity_dynamics import (VPFPVelocityDynamicsConfig,
    VPFPVelocityDynamicsNormalizer, evaluate_velocity_dynamics, fit_velocity_dynamics)
from .structured_diffusion import _coefficients


SCHEMA = "vpfp_coherent_velocity_macro_diffusion_checkpoint_v1"


def json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class VPFPVelocityMacroConfig:
    base_macro_config: VPFPCoherentMacroConfig
    drive_degree: int = 8
    schema: str = "vpfp_coherent_velocity_macro_edm_v1"

    def __post_init__(self):
        if (not isinstance(self.base_macro_config, VPFPCoherentMacroConfig)
                or self.control_count != 33 or type(self.drive_degree) is not int
                or self.drive_degree not in (4, 8)
                or self.schema != "vpfp_coherent_velocity_macro_edm_v1"):
            raise ValueError("explicit original33 macro config and degree4/8 required")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base_macro_config"), name)

    @property
    def dynamics(self):
        return VPFPVelocityDynamicsConfig(drive_degree=self.drive_degree,
            t0=self.time_domain[0], t1=self.time_domain[1])

    @property
    def velocity_dimension(self):
        return self.dynamics.latent_dimension

    @property
    def latent_dimension(self):
        return self.base_macro_config.latent_dimension - 66 + self.velocity_dimension

    @property
    def rho_dimension(self):
        # Compatibility name: this is the velocity block, NOT density.
        return self.base_macro_config.rho_dimension - 66 + self.velocity_dimension

    def to_dict(self):
        return {"schema": self.schema, "base_macro_config": self.base_macro_config.to_dict(),
                "drive_degree": self.drive_degree}

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != {"schema", "base_macro_config", "drive_degree"}:
            raise ValueError("exact velocity macro config fields required")
        return cls(VPFPCoherentMacroConfig.from_dict(payload["base_macro_config"]),
                   payload["drive_degree"], payload["schema"])


def fit_macro_coordinates(macro, conditions, config):
    """Per-row representation fit; validation fits do not fit normalization."""
    macro = _coefficients(macro, config.base_macro_config)
    condition = torch.as_tensor(conditions, device=macro.device)
    if condition.shape != (len(macro), 2) or not torch.isfinite(condition).all():
        raise ValueError("one finite nu/T condition per macro row required")
    fitted = fit_velocity_dynamics(macro[:, :66].reshape(-1, 2, 33), condition[:, 0], config.dynamics)
    return torch.cat((fitted.coordinates.to(macro), macro[:, 66:]), dim=1), fitted.diagnostics


def decode_macro_coordinates(values, conditions, config):
    """Physical33-node m for the existing conditional micro interface.

This is compression, not a reversible transform of all66 old U values.
To condition an old micro model, encode this m with THAT asset's macro
normalizer; the new white coefficient vector is never its condition tensor.
"""
    values = _coefficients(values, config)
    condition = torch.as_tensor(conditions, device=values.device)
    if condition.shape != (len(values), 2) or not torch.isfinite(condition).all():
        raise ValueError("one finite nu/T condition per decoded macro required")
    points = torch.linspace(*config.time_domain, 33, dtype=torch.float64, device=values.device)
    velocity = evaluate_velocity_dynamics(values[:, :config.velocity_dimension], condition[:, 0],
                                          points, config.dynamics)["U"].to(values)
    return torch.cat((velocity.flatten(1), values[:, config.velocity_dimension:]), dim=1)


class VPFPVelocityMacroTransform(nn.Module):
    """Fixed injective coefficient packing and left inverse, not physical U(t).

    U0/drive entries are packed in a discrete orthogonal polynomial basis on33
network slots. q=d+2 columns correspond to coefficient indices, not the
    physical Legendre acceleration law; OU is applied only at clean decoding.
    Scaling sqrt(33/q) preserves unit average field variance for independent
    unit white coefficients, matching the old normalized33-node U channel.
"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = VPFPCoherentMacroTransform(config.base_macro_config)
        matrix = np.polynomial.legendre.legvander(np.linspace(-1., 1., 33), config.drive_degree+1)
        basis, triangular = np.linalg.qr(matrix)
        basis *= np.where(np.diag(triangular) < 0., -1., 1.)[None]
        self.register_buffer("packing", torch.from_numpy(basis*np.sqrt(33./(config.drive_degree+2))).float())

    def to_field(self, values):
        values = _coefficients(values, self.config)
        # Layout groups the two U0 entries first, then x/y drive coefficients.
        q = self.config.drive_degree+1
        c = torch.cat((values[:, :2, None], values[:, 2:self.config.velocity_dimension].reshape(-1,2,q)), dim=2)
        u = torch.einsum("bcq,tq->bct", c, self.packing.to(values))
        return self.base.to_field(torch.cat((u.flatten(1), values[:, self.config.velocity_dimension:]), dim=1))

    def to_coefficients(self, field):
        macro = self.base.to_coefficients(field)
        c = torch.einsum("bct,tq->bcq", macro[:, :66].reshape(-1,2,33), self.packing.to(field))*(self.config.drive_degree+2)/33
        return torch.cat((c[:, :, 0], c[:, :, 1:].flatten(1), macro[:, 66:]), dim=1)


class VPFPVelocityMacroNormalizer(nn.Module):
    """Train-only U-coefficient scales plus exactly the old non-U scales."""
    schema = "vpfp_velocity_macro_normalizer_v1"

    def __init__(self, config, parent, velocity_payload):
        super().__init__()
        velocity = VPFPVelocityDynamicsNormalizer.from_payload(velocity_payload)
        if (parent.config != config.base_macro_config or velocity.config != config.dynamics
                or list(map(str, parent.fit_metadata["sample_ids"])) != velocity.fit_metadata["sample_ids"]):
            raise ValueError("parent/dynamics config or train sample IDs differ")
        self.config, self.parent = config, parent
        self.fit_metadata = dict(parent.fit_metadata)
        self.velocity_fit_metadata = velocity.fit_metadata
        self.register_buffer("velocity_center", velocity.center.clone())
        self.register_buffer("velocity_scales", velocity.scales.clone())

    @classmethod
    def fit(cls, config, values, *, sample_ids, parent_normalizer, split="train"):
        values = _coefficients(values, config)
        if split != "train" or list(map(int, sample_ids)) != parent_normalizer.fit_metadata["sample_ids"]:
            raise ValueError("train-only coordinates must match the original macro normalizer IDs")
        velocity = VPFPVelocityDynamicsNormalizer.fit(config.dynamics,
            values[:, :config.velocity_dimension], sample_ids=sample_ids)
        # Do not share mutable nn.Module buffers with the caller or another arm.
        parent = VPFPCoherentMacroNormalizer.from_payload(parent_normalizer.to_payload())
        return cls(config, parent, velocity.to_payload())

    def _transform(self, values, inverse):
        values = _coefficients(values, self.config)
        n = self.config.velocity_dimension
        u = values[:, :n]
        center, scales = self.velocity_center.to(values), self.velocity_scales.to(values)
        u = u*scales+center if inverse else (u-center)/scales
        macro = torch.cat((values.new_zeros((len(values),66)), values[:, n:]), dim=1)
        remainder = (self.parent.decode(macro) if inverse else self.parent.encode(macro))[:,66:]
        return torch.cat((u, remainder), dim=1)

    def encode(self, values): return self._transform(values, False)
    def decode(self, values): return self._transform(values, True)

    def to_payload(self):
        velocity = VPFPVelocityDynamicsNormalizer(self.config.dynamics, self.velocity_center,
            self.velocity_scales, self.velocity_fit_metadata).to_payload()
        data = {"schema": self.schema, "config": self.config.to_dict(),
                "parent": self.parent.to_payload(), "velocity": velocity}
        return {**data, "sha256": json_sha(data)}

    @classmethod
    def from_payload(cls, payload):
        if set(payload) != {"schema", "config", "parent", "velocity", "sha256"}:
            raise ValueError("exact velocity macro normalizer payload required")
        data = {k:v for k,v in payload.items() if k != "sha256"}
        if data['schema'] != cls.schema:
            raise ValueError("velocity macro normalizer SHA/schema differs")
        return cls(VPFPVelocityMacroConfig.from_dict(payload["config"]),
            VPFPCoherentMacroNormalizer.from_payload(payload["parent"]), payload["velocity"])


class VPFPVelocityMacroEDM(VPFPCoherentMacroEDM):
    checkpoint_schema = SCHEMA

    def __init__(self, config):
        # Exactly the old named learnable tensors and initialization RNG order.
        super().__init__(config.base_macro_config)
        self.config = config
        self.transform = VPFPVelocityMacroTransform(config)

    def architecture_report(self):
        return {"schema": "vpfp_velocity_macro_architecture_v1", "drive_degree": self.config.drive_degree,
            "parameters": sum(p.numel() for p in self.parameters()),
            "random_dimension": self.config.latent_dimension,
            "decoded_macro_dimension": self.config.base_macro_config.latent_dimension,
            "joint_global_local_macro_score": True, "old_33node_U_full_support": False,
            "packing": "fixed_orthogonal_polynomial_unit_mean_white_field_variance_not_physical_velocity",
            "nonlinear_noisy_physics_penalty": False, "equal_compute_claimed": False}


class VelocityCoordinateMetric:
    """Fixed SPD denoising metric preserving old total U/local/stress weights.

Its minimizer is still the conditional mean at every sigma. It neither uses
the physical OU inverse nor penalizes derivatives. Dividing by the original
velocity dimension avoids reducing U's total weight merely by compression.
"""
    enabled = True

    def __init__(self, config, normalizer=None): self.config = config

    def report(self):
        return {"schema": "vpfp_velocity_coordinate_fixed_metric_v1",
            "U_coordinate_multiplier": 66/self.config.velocity_dimension,
            "velocity_denominator": self.config.base_macro_config.rho_dimension,
            "positive_fixed_quadratic": True, "physical_derivative_penalty": False}

    def components(self, residual):
        cfg = self.config
        error = residual.square()
        velocity = (error[:, :cfg.velocity_dimension].sum(1)*(66/cfg.velocity_dimension)
            +error[:, cfg.velocity_dimension:cfg.rho_dimension].sum(1))/cfg.base_macro_config.rho_dimension
        stress = error[:, cfg.rho_dimension:].mean(1)
        zero = torch.zeros_like(stress)
        return {"rho_total": velocity, "kinetic_total": stress, "spatial": zero, "temporal": zero}


def load_velocity_macro_checkpoint(path, expected_sha256, *, device="cpu"):
    """Strict research seven-field loader; not the public full-answer ABI."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("velocity macro checkpoint must be a regular file")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if set(payload) != fields or payload["schema"] != SCHEMA:
        raise ValueError("exact seven-field velocity macro checkpoint required")
    cfg = VPFPVelocityMacroConfig.from_dict(payload["config"])
    normalizer = VPFPVelocityMacroNormalizer.from_payload(payload["normalizer"])
    if normalizer.config != cfg:
        raise ValueError("model/normalizer config differs")
    for key in ("base_decoder_sha256", "corpus_manifest_sha256"):
        if len(payload[key]) != 64 or any(c not in "0123456789abcdef" for c in payload[key]):
            raise ValueError("explicit source SHA required")
    training = payload["training"]
    if len(training['train_case_ids']) != len(normalizer.fit_metadata['sample_ids']):
        raise ValueError("checkpoint normalizer/train binding differs")
    model = VPFPVelocityMacroEDM(cfg)
    expected, buffers = model.state_dict(), dict(model.named_buffers())
    if set(payload["model"]) != set(expected): raise ValueError("model tensor inventory differs")
    for name,value in payload["model"].items():
        if (not torch.is_tensor(value) or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all() or (name in buffers and not torch.equal(value, expected[name]))):
            raise ValueError("invalid weight or immutable packing buffer: "+name)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), normalizer.to(device), payload
