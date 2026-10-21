"""Raw-value innovations about global-U SPACE advection, not VPFP transport.

No velocity pushforward, Hermite reprojection, clipping or suppressed residual.
All raw amplitude/rho values remain; only the original A decoder applies its
rho gauge, self-standardization and complete floor. Scaling/training is external.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import torch
from torch import nn
from .moving_frame import MovingFrameMacroConfig, half_modes, split_macro_controls
from .value_control_diffusion import VPFPValueEDMConfig

CONFIG_SCHEMA = "vpfp_bulk_space_transport_innovation_config_v1"
SCHEMA = "vpfp_bulk_space_transport_innovation_transform_v1"


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class VPFPBulkTransportInnovationConfig:
    value_config: VPFPValueEDMConfig
    macro_config: MovingFrameMacroConfig
    blocks: str = "both"
    schema: str = CONFIG_SCHEMA

    def __post_init__(self):
        if not isinstance(self.value_config, VPFPValueEDMConfig):
            raise TypeError("explicit raw native VALUE config required")
        if not isinstance(self.macro_config, MovingFrameMacroConfig):
            raise TypeError("actual A MovingFrameMacroConfig required, not network config")
        if type(self.blocks) is not str or self.blocks not in ("rho", "kinetic", "both"):
            raise ValueError("blocks must be rho, kinetic or both")
        if self.schema != CONFIG_SCHEMA:
            raise ValueError("unknown bulk transport innovation schema")
        if (self.value_config.control_count != self.macro_config.control_count
                or tuple(self.value_config.time_domain) != tuple(self.macro_config.time_domain)):
            raise ValueError("macro and value native time nodes/domain must match exactly")
        _json(self.to_dict())

    @property
    def selected_blocks(self):
        return ("rho", "kinetic") if self.blocks == "both" else (self.blocks,)

    @property
    def time_domain(self):
        return self.macro_config.time_domain

    @property
    def spatial_domain_lengths(self):
        return self.macro_config.spatial_domain_lengths

    def to_dict(self):
        return _json({"schema": self.schema, "value_config": self.value_config.to_dict(),
            "macro_config": self.macro_config.to_dict(), "blocks": self.blocks})

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema", "value_config", "macro_config", "blocks"}:
            raise ValueError("unexpected bulk transport config fields")
        result = cls(VPFPValueEDMConfig.from_dict(payload["value_config"]),
            MovingFrameMacroConfig.from_dict(payload["macro_config"]),
            blocks=payload["blocks"], schema=payload["schema"])
        if _canonical(payload) != _canonical(result.to_dict()):
            raise ValueError("complete canonical raw VALUE and A geometry configuration required")
        return result


class VPFPBulkTransportInnovationTransform(nn.Module):
    """a0,r_i=a_i-Phi_i(m)a_(i-1), preserving raw value layout/dtype/device.

    m is complete PHYSICAL [B,M], not normalized conditions. Phases and cubic
    integrals use explicit FP64; rotations/differences/left-to-right inverse use
    value dtype. No parameters, buffers, fit or RNG; .to() sets no arithmetic.
    """
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, VPFPBulkTransportInnovationConfig):
            raise TypeError("explicit VPFPBulkTransportInnovationConfig required")
        self.config = config

    @staticmethod
    def _tensor(value, width, name, *, output=False):
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.device.type not in ("cpu", "cuda")
                or value.dtype not in (torch.float32, torch.float64)
                or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != width):
            raise ValueError(f"{name} must be nonempty dense FP32/FP64 [B,{width}] on CPU/CUDA")
        if not bool(torch.isfinite(value).all()):
            if output:
                raise FloatingPointError(f"{name} overflow/nonfinite; no clipping or precision promotion")
            raise ValueError(f"{name} must be finite")
        return value

    def interval_displacements(self, physical_macro):
        """Native cubic U integral, explicitly FP64 [B,T-1,2].

        H*[h*(Ui+Ui1)/2+h^2*(s_tau_i-s_tau_i1)/12]; original endpoint-one-sided
        and interior-centered slopes. Not a hidden OU interpolation which
        produced these nodes. Local velocity/C are not predicted.
        """
        cfg = self.config.macro_config
        self._tensor(physical_macro, cfg.macro_dimension, "physical macro")
        u = split_macro_controls(physical_macro, cfg)[0].double()
        h = 1. / (cfg.control_count - 1)
        slopes = torch.cat(((u[..., 1:2] - u[..., :1]) / h,
            (u[..., 2:] - u[..., :-2]) / (2 * h),
            (u[..., -1:] - u[..., -2:-1]) / h), -1)
        displacement = (cfg.time_domain[1] - cfg.time_domain[0]) * (
            (.5 * u[..., :-1] + .5 * u[..., 1:]) * h
            + (slopes[..., :-1] - slopes[..., 1:]) * (h * h / 12.))
        if not bool(torch.isfinite(displacement).all()):
            raise FloatingPointError("bulk interval integral overflow; no wrapping or clipping")
        return displacement.transpose(1, 2)

    def _rotate(self, values, displacement, maximum_mode, *, constant):
        # values [B,T,K] or [B,H,T,K], displacement [B,T,2].
        modes = torch.tensor(half_modes(maximum_mode), dtype=torch.float64,
            device=values.device).reshape(-1, 2)
        if not len(modes):
            return values.clone()
        lengths = torch.tensor(self.config.spatial_domain_lengths,
            dtype=torch.float64, device=values.device)
        angles = 2 * math.pi * (displacement[..., None, :] * (modes / lengths)).sum(-1)
        if not bool(torch.isfinite(angles).all()):
            raise FloatingPointError("bulk Fourier phase overflow; no clipping")
        cosine, sine = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
        if values.ndim == 4:
            cosine, sine = cosine[:, None], sine[:, None]
        start = 1 if constant else 0
        pairs = values[..., start:].reshape(*values.shape[:-1], len(modes), 2)
        c, s = pairs[..., 0], pairs[..., 1]
        # a(x-delta): (c cos(theta)-s sin(theta), c sin(theta)+s cos(theta)).
        rotated = torch.stack((c * cosine - s * sine, c * sine + s * cosine), -1).flatten(-2)
        return torch.cat((values[..., :1], rotated), -1) if constant else rotated

    def _group(self, values, displacements, maximum_mode, constant, inverse):
        if not inverse:
            predicted = self._rotate(values[..., :-1, :], displacements,
                maximum_mode, constant=constant)
            return torch.cat((values[..., :1, :], values[..., 1:, :] - predicted), -2)
        current, rows = values[..., :1, :], [values[..., :1, :]]
        for index in range(values.shape[-2] - 1):
            current = self._rotate(current, displacements[:, index:index + 1],
                maximum_mode, constant=constant) + values[..., index + 1:index + 2, :]
            rows.append(current)
        return torch.cat(rows, -2)

    def _transform(self, values, physical_macro, inverse):
        cfg = self.config.value_config
        self._tensor(values, cfg.latent_dimension, "innovation" if inverse else "raw value")
        self._tensor(physical_macro, self.config.macro_config.macro_dimension, "physical macro")
        if len(values) != len(physical_macro) or values.device != physical_macro.device:
            raise ValueError("values and physical macro must have identical batch/device")
        displacements = self.interval_displacements(physical_macro)
        rho = values[:, :cfg.rho_dimension].reshape(len(values), cfg.control_count, cfg.rho_modes)
        kinetic = values[:, cfg.rho_dimension:].reshape(len(values), cfg.hermite_channels,
            cfg.control_count, cfg.kinetic_modes)
        if "rho" in self.config.selected_blocks:
            rho = self._group(rho, displacements, cfg.rho_max_mode, False, inverse)
        if "kinetic" in self.config.selected_blocks:
            kinetic = self._group(kinetic, displacements, cfg.kinetic_max_mode, True, inverse)
        result = torch.cat((rho.flatten(1), kinetic.flatten(1)), 1)
        return self._tensor(result, cfg.latent_dimension,
            "decoded values" if inverse else "innovations", output=True)

    def encode(self, raw_values, physical_macro):
        return self._transform(raw_values, physical_macro, False)

    def decode(self, coordinates, physical_macro):
        return self._transform(coordinates, physical_macro, True)

    def forward(self, raw_values, physical_macro):
        return self.encode(raw_values, physical_macro)

    def to_payload(self):
        return {"schema": SCHEMA, "config": self.config.to_dict(),
            "input_schema": "vpfp_raw_independent_value_controls_v1",
            "output_schema": "vpfp_full_raw_bulk_space_transport_innovations_v1",
            "dimension": self.config.value_config.latent_dimension,
            "selected_blocks": list(self.config.selected_blocks),
            "predictor": "a(x)-field translated to a(x-integral_of_native_cubic_U_dt)",
            "coordinate_order": "rho[B,T,R],kinetic[B,H,T,K]; first time is a0",
            "unselected_blocks": "identity", "normalization": "none_train_only_scales_are_external",
            "physical_macro_required": "actual_A_geometric_config_and_unnormalized_complete_m",
            "geometry_binding": "exact_supplied_A_time_domain_and_spatial_lengths_no_2pi_fallback",
            "macro_dependency": "global_U_only; local_u,C,rho-weighted_offset_not_predicted",
            "residual": "all coordinates retained; no timestep division or residual suppression",
            "arithmetic": {"phase_and_U_integral": "explicit_float64",
                "value_rotation_difference_inverse": "input_float32_or_float64",
                "inverse": "fixed_left_to_right_Phi_times_previous_plus_residual",
                "real_arithmetic_invertible_given_m": True, "conditional_jacobian_determinant": 1,
                "Phi_orthogonal_in_real_arithmetic": True,
                "floating_point_bitwise_roundtrip_guaranteed": False},
            "raw_amplitude_gauge": "preserved_not_reinterpreted_as_physical_error",
            "floor": "unchanged_entire_floor_in_original_A_decoder",
            "full_state_support": True, "fit_required": False,
            "velocity_pushforward": False, "finite_Hermite_reprojection": False,
            "exact_VPFP_transport": False, "approximate_prior": "bulk_space_advection_only",
            "physics_or_distribution_admission": False}

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError("invalid bulk transport transform payload")
        transform = cls(VPFPBulkTransportInnovationConfig.from_dict(payload["config"]))
        if _canonical(payload) != _canonical(transform.to_payload()):
            raise ValueError("bulk transport payload conventions/schema/layout differ")
        return transform
