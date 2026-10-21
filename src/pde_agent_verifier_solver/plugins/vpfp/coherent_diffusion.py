"""Explicit macro-prior / conditional full-state EDM research models.

The coordinate is not a pure shape quotient. The conditional model retains
the old complete value state (including rho). Sampling its joint (m,z) once
permits a decoder-only paired experiment; neither model qualifies a regime.
No physical penalty or nonlinear projection is applied at noisy states.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import torch
from torch import nn
from torch.nn import functional as F

from .structured_diffusion import (
    VPFPStructuredEDM, VPFPStructuredEDMConfig, _SpatiotemporalBlock,
    _coefficients, _fourier_basis, _seeds,
)
from .value_control_diffusion import VPFPValueEDMConfig, VPFPValueControlEDM


@dataclass(frozen=True)
class VPFPCoherentMacroConfig:
    """U[2,T], local_u[2,T,M-1], log-Cholesky[3,T,M], real Fourier."""
    full_config: VPFPStructuredEDMConfig
    maximum_mode: int = 6
    coherent_mode: int = 2
    schema: str = "vpfp_coherent_macro_edm_v1"

    def __post_init__(self):
        if not isinstance(self.full_config, VPFPStructuredEDMConfig):
            raise TypeError("macro prior requires the explicit physical/full config")
        if (self.schema != "vpfp_coherent_macro_edm_v1" or self.maximum_mode < 1
                or not 0 <= self.coherent_mode <= self.maximum_mode
                or self.grid_size < 2*self.maximum_mode+1):
            raise ValueError("invalid coherent macro schema or aliased grid")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "full_config"), name)

    @property
    def modes(self):
        return (2*self.maximum_mode+1)**2

    @property
    def rho_dimension(self):
        # Inherited EDM balances velocity and stress coordinate blocks;
        # this name is a trainer compatibility alias, NOT physical rho.
        return 2*self.control_count*self.modes

    @property
    def latent_dimension(self):
        return 5*self.control_count*self.modes

    @property
    def full_dimension(self):
        """Source corpus ABI; macro latents are a derived view of its answers."""
        return self.full_config.latent_dimension

    @property
    def inactive_dim(self):
        return 0

    @property
    def field_channels(self):
        return 7

    def to_dict(self):
        return dict(schema=self.schema, full_config=self.full_config.to_dict(),
                    maximum_mode=self.maximum_mode, coherent_mode=self.coherent_mode)

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        data["full_config"] = VPFPStructuredEDMConfig.from_dict(data["full_config"])
        return cls(**data)


def split_macro(values, config):
    values = _coefficients(values, config)
    n, t, m = len(values), config.control_count, config.modes
    return (values[:, :2*t].reshape(n, 2, t),
            values[:, 2*t:2*t*m].reshape(n, 2, t, m-1),
            values[:, 2*t*m:].reshape(n, 3, t, m))


def join_macro(global_velocity, local_velocity, log_cholesky):
    return torch.cat((global_velocity.flatten(1), local_velocity.flatten(1),
                      log_cholesky.flatten(1)), dim=1)


class VPFPCoherentMacroTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("basis", _fourier_basis(config.maximum_mode,
            config.grid_size, constant=True, normalized=True))
        mask = [True]
        for kx in range(-config.maximum_mode, config.maximum_mode+1):
            for ky in range(-config.maximum_mode, config.maximum_mode+1):
                if kx > 0 or (kx == 0 and ky > 0):
                    mask.extend([max(abs(kx), abs(ky)) <= config.coherent_mode]*2)
        self.register_buffer("coherent_mask", torch.tensor(mask, dtype=torch.bool))

    def to_field(self, values, *, coherent_only=False):
        u, local, chol = split_macro(values, self.config)
        basis = self.basis.to(values)
        if coherent_only:
            basis = basis * self.coherent_mask.to(values)[:, None, None]
        size = self.config.grid_size
        return torch.cat((u[..., None, None].expand(-1,-1,-1,size,size),
            torch.einsum("bctm,mxy->bctxy", local, basis[1:]),
            torch.einsum("bctm,mxy->bctxy", chol, basis)), dim=1)

    def to_coefficients(self, field):
        cfg = self.config
        if (field.ndim != 5 or tuple(field.shape[1:]) !=
                (7, cfg.control_count, cfg.grid_size, cfg.grid_size)
                or not torch.isfinite(field).all()):
            raise ValueError("invalid coherent macro field")
        basis = self.basis.to(field)
        return join_macro(field[:, :2].mean(dim=(-1,-2)),
            torch.einsum("bctxy,mxy->bctm", field[:, 2:4], basis[1:])/cfg.grid_size**2,
            torch.einsum("bctxy,mxy->bctm", field[:, 4:], basis)/cfg.grid_size**2)


class VPFPCoherentMacroNormalizer(nn.Module):
    """Seven train-only block scales; no coordinate-wise tiny variance inverse.

    Center time-dependent DC coordinates only. Shared cos/sin scales preserve
    the translation action on local fields. Macro data never include held-out
    or generated values during this fit.
    """
    def __init__(self, config, center, scales, fit_metadata):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float64)
        scales = torch.as_tensor(scales, dtype=torch.float64)
        if (center.shape != (config.latent_dimension,) or scales.shape != (7,)
                or not torch.isfinite(center).all() or not torch.isfinite(scales).all()
                or (scales <= 0).any() or fit_metadata.get("split") != "train"
                or fit_metadata.get("generated_samples_used_for_fit") is not False):
            raise ValueError("invalid macro train-only normalization")
        self.config, self.fit_metadata = config, dict(fit_metadata)
        self.register_buffer("center", center.clone())
        self.register_buffer("scales", scales.clone())

    @classmethod
    def fit(cls, config, values, *, sample_ids, split="train"):
        values = _coefficients(values, config).double()
        ids = list(map(int, sample_ids))
        if split != "train" or len(values) < 2 or len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("macro fit requires distinct training answers only")
        u, local, chol = split_macro(values, config)
        center_chol = torch.zeros_like(chol[:1])
        center_chol[..., 0] = chol[..., 0].mean(0, keepdim=True)
        center = join_macro(u.mean(0, keepdim=True), torch.zeros_like(local[:1]), center_chol)
        blocks = split_macro(values-center, config)
        scales = torch.cat([block.square().mean(dim=tuple(i for i in range(block.ndim) if i != 1)).sqrt()
                            for block in blocks]).clamp_min(1.e-5)
        return cls(config, center[0], scales, {"split": "train", "sample_ids": ids,
            "generated_samples_used_for_fit": False, "absolute_scale_floor": 1.e-5,
            "centering": "time_dependent_global_U_and_logCholesky_DC_only",
            "schema": "vpfp_coherent_macro_train_block_normalization_v1"})

    def _transform(self, values, inverse):
        values = _coefficients(values, self.config)
        centered = values if inverse else values-self.center.to(values)
        u, local, chol = split_macro(centered, self.config)
        scales = self.scales.to(values)
        if not inverse:
            scales = scales.reciprocal()
        result = join_macro(u*scales[:2][None,:,None],
            local*scales[2:4][None,:,None,None], chol*scales[4:][None,:,None,None])
        return result+self.center.to(values) if inverse else result

    def encode(self, values):
        return self._transform(values, False)

    def decode(self, values):
        return self._transform(values, True)

    def to_payload(self):
        return {"schema": "vpfp_coherent_macro_normalizer_v1", "config": self.config.to_dict(),
                "center": self.center.detach().cpu().tolist(), "scales": self.scales.detach().cpu().tolist(),
                "fit_metadata": self.fit_metadata}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("schema") != "vpfp_coherent_macro_normalizer_v1":
            raise ValueError("unknown macro normalizer schema")
        return cls(VPFPCoherentMacroConfig.from_dict(payload["config"]), payload["center"],
                   payload["scales"], payload["fit_metadata"])


class VPFPCoherentMacroEDM(VPFPStructuredEDM):
    checkpoint_schema = "vpfp_coherent_macro_diffusion_checkpoint_v1"

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.transform = VPFPCoherentMacroTransform(config)
        self.input = nn.Conv3d(7, config.width, 1)
        self.context = nn.Sequential(nn.Linear(18, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.time_position = nn.Linear(3, config.width)
        self.blocks = nn.ModuleList(_SpatiotemporalBlock(config.width,
            config.temporal_dilation_cycle[i % len(config.temporal_dilation_cycle)]) for i in range(config.depth))
        self.output = nn.Conv3d(config.width, 7, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class VPFPMacroConditionalEDM(VPFPValueControlEDM):
    """Condition tensor packs [nu, bath_T, clean/generated normalized_m].

    The inherited EDM loss and per-case Heun sampling operate exclusively in
    micro white coordinates. The macro tensor is held constant along that
    trajectory. Training callers use same-row clean m; generation uses p(m).
    """
    checkpoint_schema = "vpfp_macro_conditional_diffusion_checkpoint_v1"

    def loss(self, clean_normalized_coefficients, conditions, *, generator=None,
             sigma=None, noise=None, residual_metric=None):
        if residual_metric is not None:
            raise ValueError("old full/value residual metrics do not apply to conditional white coordinates")
        return super().loss(clean_normalized_coefficients, conditions, generator=generator,
                            sigma=sigma, noise=noise, residual_metric=None)

    def __init__(self, config, macro_config):
        super().__init__(config)
        if (config.control_count != macro_config.control_count or config.grid_size != macro_config.grid_size
                or config.collision_nu_range != macro_config.collision_nu_range
                or config.temperature_range != macro_config.temperature_range):
            raise ValueError("macro and micro physical layouts differ")
        self.macro_config = macro_config
        self.macro_transform = VPFPCoherentMacroTransform(macro_config)
        self.macro_input = nn.Conv3d(7, config.width, 1)
        self.coherent_input = nn.Conv3d(7, config.width, 1)

    def _conditions(self, conditions, count, prototype):
        condition = torch.as_tensor(conditions, dtype=prototype.dtype, device=prototype.device)
        if condition.shape != (count, 2+self.macro_config.latent_dimension) or not torch.isfinite(condition).all():
            raise ValueError("conditional micro model needs physical conditions plus explicit macro coordinates")
        return VPFPStructuredEDM._conditions(self, condition[:, :2], count, prototype)

    def denoise(self, noisy_normalized_coefficients, sigma, conditions):
        parameter = next(self.parameters())
        noisy = _coefficients(noisy_normalized_coefficients, self.config, device=parameter.device, dtype=parameter.dtype)
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        if sigma.shape != (len(noisy),) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("invalid EDM noise level")
        condition = self._conditions(conditions, len(noisy), noisy)
        macro = torch.as_tensor(conditions, dtype=noisy.dtype, device=noisy.device)[:, 2:]
        denominator = torch.sqrt(sigma.square()+self.config.sigma_data**2)
        field = self.transform.to_field(noisy/denominator[:, None])
        frequencies = torch.exp(torch.linspace(0., math.log(100.), 8, device=noisy.device, dtype=noisy.dtype))
        phase = sigma.log()[:, None]*frequencies[None]
        embedding = self.context(torch.cat((phase.sin(), phase.cos(), condition), dim=1))
        tau = torch.linspace(0., 1., self.config.control_count, device=noisy.device, dtype=noisy.dtype)
        position = self.time_position(torch.stack((tau, (math.pi*tau).sin(), (math.pi*tau).cos()), dim=1))
        hidden = (self.input(field)+position.T[None,:,:,None,None]
            +self.macro_input(self.macro_transform.to_field(macro))
            +self.coherent_input(self.macro_transform.to_field(macro, coherent_only=True)))
        for block in self.blocks:
            hidden = block(hidden, embedding)
        residual = self.transform.to_coefficients(self.output(F.silu(hidden)))
        return (self.config.sigma_data**2/denominator.square())[:, None]*noisy + (
            sigma*self.config.sigma_data/denominator)[:, None]*residual


@torch.no_grad()
def sample_hierarchical(macro_model, macro_normalizer, micro_model, micro_normalizer,
                        conditions, seeds, *, steps=64):
    """One joint draw, two decoder identities downstream; no oracle m at inference."""
    if macro_model.config != macro_normalizer.config or micro_model.macro_config != macro_model.config:
        raise ValueError("hierarchical model/normalizer configuration mismatch")
    seeds = _seeds(seeds)
    macro_seeds = [int.from_bytes(sha256(f"vpfp-coherent-macro:{seed}".encode()).digest()[:8], "little") % (2**63) for seed in seeds]
    micro_seeds = [int.from_bytes(sha256(f"vpfp-coherent-micro:{seed}".encode()).digest()[:8], "little") % (2**63) for seed in seeds]
    white_m = macro_model.sample(conditions, macro_seeds, steps=steps)
    physical = torch.as_tensor(conditions, dtype=white_m.dtype, device=white_m.device)
    white_z = micro_model.sample(torch.cat((physical, white_m), dim=1), micro_seeds, steps=steps)
    return {"macro_coordinates": macro_normalizer.decode(white_m),
            "full_coefficients": micro_normalizer.decode(white_z),
            "macro_noise_seeds": macro_seeds, "micro_noise_seeds": micro_seeds,
            "decoder_pairing": "same_joint_draw_not_two_independent_models",
            "scientific_qualification": False}
