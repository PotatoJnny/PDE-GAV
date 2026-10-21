"""Zero-start derivative conditioning for the existing complete-state EDM.

Inputs are the SAME normalized macro coordinates as the old micro model.
These are derivatives of condition-coordinate fields, NOT physical velocity
or stress jets: centering/scaling has not been undone, log-Cholesky is not C,
and z-dependent density centering is deliberately unavailable here. No clean
z, new noise, source penalty, mode truncation or output smoothing is used.

This independent research module is not registered in any checkpoint loader
or trainer. A future launcher must bind ``identity()`` including the explicit
spatial periods to its physical decoder before training/loading this schema.
"""
from __future__ import annotations

import math

# Preserve the frozen VPFP import order (NumPy before Torch).
import numpy as np  # noqa: F401
import torch
from torch import nn
from torch.nn import functional as F

from .coherent_diffusion import VPFPCoherentMacroTransform, join_macro, split_macro
from .innovation_diffusion import VPFPInnovationConditionalEDM
from .structured_diffusion import _coefficients
from .value_control_diffusion import value_control_slopes


CHECKPOINT_SCHEMA = "vpfp_macro_jet_innovation_conditional_edm_checkpoint_v1"
JET_SCHEMA = "vpfp_normalized_macro_condition_coordinate_jets_v1"


class MacroConditionCoordinateJets(nn.Module):
    """[B,Dm] -> [B,21,T,X,Y], ordered (dt[7], dx[7], dy[7]).

    Time jets are the original one-sided/centered value-control slopes at
    knots, divided by physical time duration. They are the knot derivatives
    of that coordinate's native cubic interpolant, not exact derivatives of
    an unknown continuous trajectory. Spatial derivatives use all original
    real Fourier pairs with frequencies 2*pi*k/L; time is NOT periodic.
    """

    def __init__(self, macro_config, *, spatial_domain_lengths):
        super().__init__()
        lengths = tuple(float(x) for x in spatial_domain_lengths)
        if len(lengths) != 2 or not all(math.isfinite(x) and x > 0 for x in lengths):
            raise ValueError("two explicit positive finite spatial periods required")
        self.config, self.spatial_domain_lengths = macro_config, lengths
        self.transform = VPFPCoherentMacroTransform(macro_config)
        pairs = [(kx, ky) for kx in range(-macro_config.maximum_mode, macro_config.maximum_mode+1)
                 for ky in range(-macro_config.maximum_mode, macro_config.maximum_mode+1)
                 if kx > 0 or (kx == 0 and ky > 0)]
        self.register_buffer("wave_numbers", torch.tensor(
            [[2*math.pi*kx/lengths[0], 2*math.pi*ky/lengths[1]] for kx, ky in pairs],
            dtype=torch.float64))

    def identity(self):
        return {"schema": JET_SCHEMA, "macro_config": self.config.to_dict(),
                "spatial_domain_lengths": list(self.spatial_domain_lengths),
                "input": "existing_normalized_macro_coordinates_not_physical_macro",
                "channels": ["dt_7", "dx_7", "dy_7"],
                "time_derivative": "native_value_control_knot_slopes_divided_by_time_duration",
                "spatial_derivative": "all_real_Fourier_cos_sin_pairs_2pi_k_over_L",
                "physical_jets": False, "z_dependency": False,
                "coordinate_count_changed": False, "source_or_regime_qualification": False}

    def forward(self, normalized_macro):
        values = _coefficients(normalized_macro, self.config)
        u, local, chol = split_macro(values, self.config)
        duration = self.config.time_domain[1]-self.config.time_domain[0]
        dt = join_macro(value_control_slopes(u[..., None], self.config)[..., 0]/duration,
                        value_control_slopes(local, self.config)/duration,
                        value_control_slopes(chol, self.config)/duration)

        def spatial_coefficients(axis):
            omega = self.wave_numbers[:, axis].to(values)

            def rotate(non_dc):
                pairs = non_dc.reshape(*non_dc.shape[:-1], -1, 2)
                # d(a*cos+b*sin) = (omega*b)*cos + (-omega*a)*sin.
                return torch.stack((pairs[..., 1]*omega, -pairs[..., 0]*omega), -1).flatten(-2)

            return join_macro(torch.zeros_like(u), rotate(local),
                              torch.cat((torch.zeros_like(chol[..., :1]), rotate(chol[..., 1:])), -1))

        result = torch.cat([self.transform.to_field(v) for v in
                            (dt, spatial_coefficients(0), spatial_coefficients(1))], dim=1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite macro condition-coordinate jets")
        return result


class VPFPMacroJetInnovationEDM(VPFPInnovationConditionalEDM):
    """Old joint micro denoiser plus a zero-initialized 21->width jet branch.

    Inherits the original score loss, noise law, sampler and active coordinate
    inventory. Added features enter the SAME hidden state, not an independently
    sampled solution. No bias: a constant condition cannot add a spurious jet.
    """
    checkpoint_schema = CHECKPOINT_SCHEMA

    def __init__(self, config, macro_config, *, spatial_domain_lengths):
        if tuple(config.time_domain) != tuple(macro_config.time_domain):
            raise ValueError("macro and micro physical time domains differ")
        super().__init__(config, macro_config)
        self.macro_jets = MacroConditionCoordinateJets(
            macro_config, spatial_domain_lengths=spatial_domain_lengths)
        parameter = next(self.parameters())
        devices = [] if parameter.device.type == "cpu" else [parameter.device.index]
        # Adding a zero branch must not advance the old initialization RNG.
        with torch.random.fork_rng(devices=devices):
            self.jet_input = nn.Conv3d(21, config.width, 1, bias=False,
                                      device=parameter.device, dtype=parameter.dtype)
            nn.init.zeros_(self.jet_input.weight)

    def identity(self):
        return {"checkpoint_schema": self.checkpoint_schema, "config": self.config.to_dict(),
                "jet_conditioning": self.macro_jets.identity(),
                "branch": "zero_initialized_additive_21_to_width_bias_free",
                "noise_and_loss": "unchanged_VPFPInnovationConditionalEDM",
                "training_or_physics_qualification": False}

    def denoise(self, noisy_normalized_coefficients, sigma, conditions):
        # Keep the frozen parent's arithmetic/order verbatim except for the
        # final additive jet term in hidden. Tests bind zero-branch forward,
        # all common parameter/input gradients, loss and RNG to that parent.
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
        hidden = hidden+self.jet_input(self.macro_jets(macro))
        for block in self.blocks:
            hidden = block(hidden, embedding)
        residual = self.transform.to_coefficients(self.output(F.silu(hidden)))
        return (self.config.sigma_data**2/denominator.square())[:, None]*noisy + (
            sigma*self.config.sigma_data/denominator)[:, None]*residual
