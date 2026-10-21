"""Fixed Sobolev quadratic on complete VPFP coefficient residuals.

This is NOT distribution matching of nonlinear physical f/rho/kinetic energy.
It measures derivatives of the linear pre-gauge rho Fourier field and the
unnormalized Hermite amplitude channels. No derivative is penalized toward
zero: the input must be a prediction-minus-target residual (also valid for
flow-matching velocity residuals). Fixed positive-definite weighting preserves
the population conditional-mean denoiser/vector-field optimum; finite network
capacity and optimization can still change the fitted estimator.

For fixed observed conditioning q, let mu=E[target|q]. Then
E[||prediction-target||_W^2|q] = ||prediction-mu||_W^2 + constant.
Since W is positive definite on ACTIVE coefficients, mu is the unique optimum.
The same argument applies to a velocity target conditioned on a flow state and
flow time. The noise law, EDM preconditioner and reverse sampler are unchanged.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np
import torch
from torch import nn

from .structured_diffusion import (
    VPFPCoefficientNormalizer, VPFPStructuredEDMConfig, _coefficients, _split,
)


def _time_factors(config):
    """Exact square roots of segment mass / physical-time derivative Grams.

    Columns are (left value, right value, left d/dtau, right d/dtau).
    Four-point Gauss-Legendre exactly integrates cubic squared (degree six).
    Positive quadrature makes the computed quadratic manifestly nonnegative.
    Integrals are temporal averages: integral_0^1 (...) d tau, not sums of
    arbitrary control differences. Physical d/dt includes 1/(t1-t0).
    """
    nodes, weights = np.polynomial.legendre.leggauss(4)
    u = torch.tensor((nodes + 1.) / 2., dtype=torch.float64)
    weights = torch.tensor(weights / 2., dtype=torch.float64)
    h = config.slope_interval
    duration = config.time_domain[1] - config.time_domain[0]
    values = torch.stack((2*u**3 - 3*u**2 + 1, -2*u**3 + 3*u**2,
                          h*(u**3 - 2*u**2 + u), h*(u**3 - u**2)), dim=1)
    derivatives = torch.stack(((6*u**2 - 6*u)/h, (-6*u**2 + 6*u)/h,
                               3*u**2 - 4*u + 1, 3*u**2 - 2*u), dim=1) / duration
    root_measure = (weights * h).sqrt()[:, None]
    return values * root_measure, derivatives * root_measure


def _fourier_weights(maximum, lengths, *, constant, normalized):
    pairs = [(0, 0)] if constant else []
    for kx in range(-maximum, maximum + 1):
        for ky in range(-maximum, maximum + 1):
            if kx > 0 or (kx == 0 and ky > 0):
                pairs.extend(((kx, ky), (kx, ky)))  # cos, sin; native ordering
    modes = torch.tensor(pairs, dtype=torch.float64)
    mass = torch.full((len(pairs),), 1. if normalized else .5, dtype=torch.float64)
    if constant:
        mass[0] = 1.
    frequencies = modes * torch.tensor([2*math.pi/value for value in lengths], dtype=torch.float64)
    return mass, mass * frequencies[:, 0].square(), mass * frequencies[:, 1].square()


class VPFPSobolevMetric(nn.Module):
    """Base block MSE plus fixed trace-normalized derivative quadratics.

    group_scales are SNAPSHOTTED from the train-only normalizer. Expectations
    of each nonzero derivative component under iid N(0,1) ACTIVE normalized
    coefficients equal one. No target/prediction/batch statistic changes W.
    Spatial x+y uses one joint trace, preserving anisotropic length ratios.
    Temporal trace normalization cancels a uniform duration rescaling, and
    either trace normalization cancels a uniform field-amplitude scale; raw
    energies/traces retain those physical factors and are reported explicitly.
    The identity base on active coefficients makes W positive definite, even
    though derivatives alone have constant-field nullspaces. Inactive legacy
    coordinates are frozen, not random variables of this metric.
    """

    def __init__(self, config: VPFPStructuredEDMConfig,
                 normalizer: VPFPCoefficientNormalizer, *, spatial_weight=.1,
                 temporal_weight=.1, spatial_lengths=(2*math.pi, 2*math.pi),
                 latent_input_scale=2., structured_local_scale=1.):
        super().__init__()
        if normalizer.config != config:
            raise ValueError("Sobolev config differs from the frozen normalizer")
        for name, value in (("spatial_weight", spatial_weight), ("temporal_weight", temporal_weight)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        lengths = tuple(float(value) for value in spatial_lengths)
        if len(lengths) != 2 or not all(math.isfinite(value) and value > 0 for value in lengths):
            raise ValueError("spatial_lengths must contain two positive physical lengths")
        for name, value in (("latent_input_scale", latent_input_scale),
                            ("structured_local_scale", structured_local_scale)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.config = config
        self.spatial_weight = float(spatial_weight)
        self.temporal_weight = float(temporal_weight)
        self.total_scale = 1. + self.spatial_weight + self.temporal_weight
        if not math.isfinite(self.total_scale):
            raise ValueError("combined Sobolev weights overflow their fixed scale")
        self.spatial_lengths = lengths
        self.latent_input_scale = float(latent_input_scale)
        self.structured_local_scale = float(structured_local_scale)
        self.normalizer_sha256 = sha256(json.dumps(normalizer.to_payload(),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        scales = normalizer.group_scales.detach().to(device="cpu", dtype=torch.float64).clone()
        decode = scales / scales.new_tensor((1., config.slope_interval))
        # Match the linear part of the direct fixed decoder; later rho gauge,
        # unit-norm Hermite projection and square map are deliberately absent.
        decode[0] *= self.latent_input_scale
        decode[1:] *= self.latent_input_scale * self.structured_local_scale / math.sqrt(config.kinetic_modes)
        self.register_buffer("decode_factors", decode)
        mass_factor, time_factor = _time_factors(config)
        self.register_buffer("mass_factor", mass_factor)
        self.register_buffer("time_factor", time_factor)
        for group, maximum, constant, normalized in (
                ("rho", config.rho_max_mode, False, False),
                ("kinetic", config.kinetic_max_mode, True, True)):
            weights = _fourier_weights(maximum, lengths, constant=constant, normalized=normalized)
            for name, values in zip(("mass", "x", "y"), weights):
                self.register_buffer(f"{group}_{name}_weights", values)
            factors = decode[:1] if group == "rho" else decode[1:]
            # Each interval has two independently-noised endpoint values and
            # two slopes. Shared nodes contribute in both adjoining integrals.
            traces = []
            for temporal, spatial in ((mass_factor, weights[1] + weights[2]), (time_factor, weights[0])):
                diagonal = temporal.square().sum(dim=0)
                trace = ((diagonal[:2].sum()*factors[:, 0].square()
                          + diagonal[2:].sum()*factors[:, 1].square()).mean()
                         * (config.control_count - 1) * spatial.sum())
                traces.append(trace)
            trace_values = torch.stack(traces)
            nonzero_spatial = group == "rho" or config.kinetic_max_mode > 0
            if (not torch.isfinite(trace_values).all() or trace_values[1] <= 0
                    or (nonzero_spatial and trace_values[0] <= 0)):
                raise ValueError("Sobolev operator trace overflows or underflows its fixed FP64 scale")
            self.register_buffer(f"{group}_traces", trace_values)
        # Remember structural zeros before any caller .to(dtype) conversion.
        # In particular, a true DC-only kinetic spatial trace may be zero;
        # underflow of a nonzero rho/spatial/time trace may NEVER mean DC.
        self._buffer_nonzero_counts = {
            name: int(torch.count_nonzero(buffer).item())
            for name, buffer in self.named_buffers()
        }

    @property
    def enabled(self):
        return self.spatial_weight != 0. or self.temporal_weight != 0.

    @staticmethod
    def _normalized(energy, trace):
        # A kinetic DC-only layout has an identically zero spatial operator.
        return energy / torch.where(trace > 0, trace, trace.new_ones(()))

    def _checked_cast(self, name, prototype):
        """Fail closed outside the actual arithmetic dtype's supported range.

        Constructor validation is FP64; it does not imply representability in
        FP32. Reject subnormals as well as zero/inf conversions, since GPU FTZ
        can erase subnormal factors even when the CPU conversion retained them.
        This does not rescale W or clip a trace to make training continue.
        """
        converted = getattr(self, name).to(prototype)
        finite = torch.isfinite(converted).all()
        normal_or_zero = ((converted == 0) | (converted.abs() >= torch.finfo(prototype.dtype).tiny)).all()
        preserved_nonzeros = torch.count_nonzero(converted) == self._buffer_nonzero_counts[name]
        if not bool((finite & normal_or_zero & preserved_nonzeros).item()):
            raise ValueError(
                f"Sobolev buffer {name} is not representable as finite normal {prototype.dtype} values; "
                "use wider precision or rescale the fixed physical units")
        return converted

    def components(self, residual):
        """Return differentiable per-case terms, never an average over cases.

        rho_total / kinetic_total combine (base + weighted derivative terms)
        divided by fixed (1 + spatial_weight + temporal_weight);
        total applies config.macro_loss_fraction. *_spatial/temporal are fixed
        trace-normalized energies, whereas *_raw_x/y/temporal retain units.
        """
        values = _coefficients(residual, self.config)
        if values.dtype not in (torch.float32, torch.float64):
            raise ValueError("Sobolev residual quadratics require float32 or float64")
        rho, kinetic = _split(values, self.config)
        output = {"rho_base": rho.square().flatten(1).mean(1),
                  "kinetic_base": kinetic.square().flatten(1).mean(1)}
        mass_factor = self._checked_cast("mass_factor", values)
        time_factor = self._checked_cast("time_factor", values)
        decode = self._checked_cast("decode_factors", values)
        for group, coefficients, factors in (("rho", rho[:, None], decode[:1]),
                                             ("kinetic", kinetic, decode[1:])):
            raw = coefficients * factors[None, :, :, None, None]
            local = torch.stack((raw[:, :, 0, :-1], raw[:, :, 0, 1:],
                                 raw[:, :, 1, :-1], raw[:, :, 1, 1:]), dim=-1)
            mass_samples = local @ mass_factor.T
            time_samples = local @ time_factor.T
            # Sum time quadrature and intervals, average Hermite channels.
            mass_by_mode = mass_samples.square().sum(dim=(-1, 2)).mean(dim=1)
            time_by_mode = time_samples.square().sum(dim=(-1, 2)).mean(dim=1)
            for axis in ("x", "y"):
                output[f"{group}_raw_{axis}"] = (mass_by_mode * self._checked_cast(f"{group}_{axis}_weights", values)).sum(1)
            output[f"{group}_raw_temporal"] = (time_by_mode * self._checked_cast(f"{group}_mass_weights", values)).sum(1)
            traces = self._checked_cast(f"{group}_traces", values)
            output[f"{group}_spatial"] = self._normalized(
                output[f"{group}_raw_x"] + output[f"{group}_raw_y"], traces[0])
            output[f"{group}_temporal"] = self._normalized(output[f"{group}_raw_temporal"], traces[1])
            # Preserve exact original base values when both weights are zero.
            output[f"{group}_total"] = output[f"{group}_base"]
            if self.spatial_weight:
                output[f"{group}_total"] = output[f"{group}_total"] + self.spatial_weight * output[f"{group}_spatial"]
            if self.temporal_weight:
                output[f"{group}_total"] = output[f"{group}_total"] + self.temporal_weight * output[f"{group}_temporal"]
            if self.enabled:
                output[f"{group}_total"] = output[f"{group}_total"] / self.total_scale
        fraction = self.config.macro_loss_fraction
        for component in ("base", "spatial", "temporal", "total"):
            output[component] = fraction * output[f"rho_{component}"] + (1-fraction) * output[f"kinetic_{component}"]
        if not bool(torch.isfinite(torch.stack(tuple(output.values()), dim=-1)).all().item()):
            raise ValueError(
                f"Sobolev quadratic energy overflowed {values.dtype}; "
                "use wider precision or rescale the fixed physical units")
        return output

    def forward(self, residual):
        return self.components(residual)["total"]

    def report(self):
        return {"schema": "vpfp_fixed_linear_sobolev_residual_v1",
            "normalizer_sha256": self.normalizer_sha256,
            "spatial_weight": self.spatial_weight, "temporal_weight": self.temporal_weight,
            "total_scale_divisor": self.total_scale,
            "white_noise_total_expectation": "one_when_both_group_derivative_operators_are_nonzero; DC_only_spatial_operator_contributes_zero",
            "spatial_lengths": list(self.spatial_lengths), "time_domain": list(self.config.time_domain),
            "delta_tau": self.config.slope_interval, "latent_input_scale": self.latent_input_scale,
            "structured_local_scale": self.structured_local_scale,
            "rho_spatial_temporal_traces": self.rho_traces.detach().cpu().tolist(),
            "kinetic_spatial_temporal_traces": self.kinetic_traces.detach().cpu().tolist(),
            "integral": "mean_over_physical_space_and_time_and_kinetic_channels",
            "time_quadrature": "four_point_Gauss_Legendre_exact_for_piecewise_cubic_squared",
            "normalization": "fixed_operator_trace_under_iid_unit_active_normalized_coefficient_noise",
            "population_optimum": "unchanged_conditional_mean_for_denoiser_or_FM_velocity_target",
            "duration_scaling": "raw_temporal_energy_and_trace_both_scale_as_duration_inverse_squared",
            "linear_fields": ["pre_positivity_gauge_rho_Fourier_field", "unnormalized_Hermite_amplitude_channels"],
            "prediction_or_batch_dependent_weights": False,
            "arithmetic_range_policy": "all nonzero buffers must remain finite normal values in the residual dtype; all per-case energies must be finite",
            "unsupported_dtype_scale_policy": "fail_closed_without_clipping_or_reclassifying_nonzero_derivatives_as_DC",
            "inactive_policy": "exact_zero_excluded_from_active_SPD_domain",
            "physical_derivative_distribution_matching": False,
            "scientific_distribution_qualification": False}


__all__ = ["VPFPSobolevMetric"]
