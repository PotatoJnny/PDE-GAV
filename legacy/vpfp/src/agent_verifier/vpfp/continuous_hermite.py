from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_v2 import (
    VPFPStandardEquationConfig,
    _ResidualMLP,
    _broadcast_rows,
    _real_fourier_half_modes,
    _time_feature_dim,
    _torch,
)
from agent_verifier.vpfp.continuous_flow import EQUATION_CONDITION_KEYS
from agent_verifier.vpfp.hermite_density import (
    SQUARED_HERMITE_REPRESENTATION,
    torch_squared_hermite_density,
    total_degree_mode_pairs,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPContinuousHermiteConfig:
    latent_dim: int = 128
    latent_input_scale: float = 2.0
    micro_latent_dim: int = 32
    context_dim: int = 128
    hidden_width: int = 192
    hidden_layers: int = 5
    rho_max_mode_x: int = 6
    rho_max_mode_y: int = 6
    rho_contrast: float = 0.80
    rho_positivity_gauge: str = "coordinate_l1_v1"
    spatial_feature_modes: int = 6
    time_feature_modes: int = 4
    hermite_total_degree: int = 8
    coefficient_width: int = 192
    coefficient_blocks: int = 3
    structured_spectral_latent: bool = False
    structured_rho_time_order: int = 2
    structured_kinetic_global_dim: int = 64
    structured_local_channels: int = 48
    structured_local_spatial_mode: int = 4
    structured_local_time_order: int = 2
    structured_time_basis: str = "global_chebyshev_v1"
    structured_time_control_count: int = 0
    structured_local_scale: float = 1.0
    structured_direct_coefficients: bool = False
    structured_residual_scale: float = 0.10
    coefficient_anchor: float = 0.10
    velocity_floor_mixture: float = 1.0e-5
    micro_scale: float = 0.08
    micro_latent_radius: float = 3.0
    collision_nu_range: tuple[float, float] = (0.02, 0.08)
    temperature_range: tuple[float, float] = (0.8, 1.2)
    time_domain: tuple[float, float] = (0.0, 0.35)
    dtype: str = "float32"
    representation: str = SQUARED_HERMITE_REPRESENTATION
    equation: VPFPStandardEquationConfig = VPFPStandardEquationConfig()

    def validate(self) -> None:
        self.equation.validate()
        if min(
            self.latent_dim,
            self.micro_latent_dim,
            self.context_dim,
            self.hidden_width,
            self.hidden_layers,
            self.coefficient_width,
            self.coefficient_blocks,
        ) < 1:
            raise ValueError("continuous-Hermite network dimensions must be positive")
        if self.latent_input_scale <= 0.0:
            raise ValueError("latent_input_scale must be positive")
        if min(self.rho_max_mode_x, self.rho_max_mode_y) < 1:
            raise ValueError("rho Fourier mode cutoffs must be positive")
        if not 0.0 < self.rho_contrast < 1.0:
            raise ValueError("rho_contrast must lie in (0,1)")
        if self.rho_positivity_gauge not in {
            "coordinate_l1_v1",
            "phase_invariant_group_l1_v2",
        }:
            raise ValueError("unsupported rho positivity gauge")
        if min(self.spatial_feature_modes, self.time_feature_modes) < 1:
            raise ValueError("coordinate feature mode counts must be positive")
        if self.hermite_total_degree < 0:
            raise ValueError("hermite_total_degree must be nonnegative")
        if min(
            self.structured_rho_time_order,
            self.structured_local_spatial_mode,
            self.structured_local_time_order,
            self.structured_time_control_count,
        ) < 0:
            raise ValueError("structured spectral orders must be nonnegative")
        if self.structured_time_basis not in {
            "global_chebyshev_v1",
            "piecewise_cubic_hermite_v1",
        }:
            raise ValueError("unsupported structured time basis")
        if (
            self.structured_time_basis == "piecewise_cubic_hermite_v1"
            and self.structured_time_control_count < 2
        ):
            raise ValueError(
                "piecewise cubic Hermite time basis requires at least "
                "two controls"
            )
        if self.structured_local_scale <= 0.0:
            raise ValueError("structured_local_scale must be positive")
        if self.structured_residual_scale < 0.0:
            raise ValueError("structured_residual_scale must be nonnegative")
        if self.structured_spectral_latent:
            if min(
                self.structured_kinetic_global_dim,
                self.structured_local_channels,
            ) < 1:
                raise ValueError(
                    "structured latent dimensions must be positive"
                )
            layout = self.structured_latent_layout()
            if self.latent_dim != layout["total_latent_dim"]:
                raise ValueError(
                    "latent_dim does not match structured spectral layout: "
                    f"{self.latent_dim} != {layout['total_latent_dim']}"
                )
            if self.structured_direct_coefficients and (
                self.structured_local_channels
                != layout["hermite_mode_count"]
            ):
                raise ValueError(
                    "direct structured coefficients require one local "
                    "channel per Hermite mode"
                )
        elif self.structured_direct_coefficients:
            raise ValueError(
                "direct structured coefficients require "
                "structured_spectral_latent"
            )
        if self.coefficient_anchor <= 0.0:
            raise ValueError("coefficient_anchor must be positive")
        if not 0.0 <= self.velocity_floor_mixture < 1.0:
            raise ValueError("velocity_floor_mixture must lie in [0,1)")
        if self.micro_scale < 0.0:
            raise ValueError("micro_scale must be nonnegative")
        if self.micro_latent_radius <= 0.0:
            raise ValueError("micro_latent_radius must be positive")
        if self.collision_nu_range[1] <= self.collision_nu_range[0]:
            raise ValueError("collision_nu_range must be increasing")
        if self.temperature_range[1] <= self.temperature_range[0]:
            raise ValueError("temperature_range must be increasing")
        if self.time_domain[1] <= self.time_domain[0]:
            raise ValueError("time_domain must satisfy t1 > t0")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def structured_latent_layout(self) -> dict[str, int]:
        rho_mode_count = len(
            _real_fourier_half_modes(
                self.rho_max_mode_x,
                self.rho_max_mode_y,
            )
        )
        spatial_feature_count = (
            2 * int(self.structured_local_spatial_mode) + 1
        ) ** 2
        rho_time_feature_count = self.structured_rho_time_feature_count()
        local_time_feature_count = self.structured_local_time_feature_count()
        rho_latent_dim = rho_time_feature_count * 2 * rho_mode_count
        local_latent_dim = (
            int(self.structured_local_channels)
            * local_time_feature_count
            * spatial_feature_count
        )
        hermite_mode_count = len(
            total_degree_mode_pairs(self.hermite_total_degree)
        )
        total_latent_dim = (
            rho_latent_dim
            + int(self.structured_kinetic_global_dim)
            + local_latent_dim
        )
        return {
            "rho_mode_count": rho_mode_count,
            "rho_time_feature_count": rho_time_feature_count,
            "rho_latent_dim": rho_latent_dim,
            "kinetic_global_latent_dim": int(
                self.structured_kinetic_global_dim
            ),
            "local_spatial_feature_count": spatial_feature_count,
            "local_time_feature_count": local_time_feature_count,
            "local_latent_dim": local_latent_dim,
            "hermite_mode_count": hermite_mode_count,
            "total_latent_dim": total_latent_dim,
        }

    def structured_rho_time_feature_count(self) -> int:
        if self.structured_time_basis == "piecewise_cubic_hermite_v1":
            return 2 * int(self.structured_time_control_count)
        return int(self.structured_rho_time_order) + 1

    def structured_local_time_feature_count(self) -> int:
        if self.structured_time_basis == "piecewise_cubic_hermite_v1":
            return 2 * int(self.structured_time_control_count)
        return int(self.structured_local_time_order) + 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["collision_nu_range"] = [
            float(value) for value in self.collision_nu_range
        ]
        payload["temperature_range"] = [
            float(value) for value in self.temperature_range
        ]
        payload["time_domain"] = [float(value) for value in self.time_domain]
        payload["equation"] = self.equation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPContinuousHermiteConfig":
        payload = dict(data)
        for key in ("collision_nu_range", "temperature_range", "time_domain"):
            if key in payload:
                payload[key] = tuple(float(value) for value in payload[key])
        payload["equation"] = VPFPStandardEquationConfig.from_dict(
            payload.get("equation")
        )
        return cls(**payload)



def translate_structured_spectral_latent(
    latent,
    config: VPFPContinuousHermiteConfig | dict[str, Any],
    translation_fraction,
):
    """Translate represented fields by x/L fractions on the spatial torus."""

    cfg = (
        config
        if isinstance(config, VPFPContinuousHermiteConfig)
        else VPFPContinuousHermiteConfig.from_dict(dict(config))
    )
    cfg.validate()
    if not cfg.structured_spectral_latent:
        raise ValueError("translation requires a structured spectral latent")
    if cfg.rho_positivity_gauge != "phase_invariant_group_l1_v2":
        raise ValueError(
            "exact latent translation requires phase-invariant rho gauge"
        )
    torch = _torch()
    values = torch.as_tensor(latent)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None]
    if values.ndim != 2 or int(values.shape[1]) != int(cfg.latent_dim):
        raise ValueError("latent shape does not match structured Hermite config")
    shifts = torch.as_tensor(
        translation_fraction,
        dtype=values.dtype,
        device=values.device,
    )
    if shifts.ndim == 1:
        shifts = shifts[None]
    if shifts.ndim != 2 or int(shifts.shape[1]) != 2:
        raise ValueError("translation_fraction must have shape [2] or [B,2]")
    if int(shifts.shape[0]) == 1:
        shifts = shifts.expand(int(values.shape[0]), -1)
    if int(shifts.shape[0]) != int(values.shape[0]):
        raise ValueError("translation batch does not match latent batch")

    output = values.clone()
    layout = cfg.structured_latent_layout()
    rho_end = int(layout["rho_latent_dim"])
    rho_modes = _real_fourier_half_modes(
        cfg.rho_max_mode_x, cfg.rho_max_mode_y
    )
    rho = output[:, :rho_end].reshape(
        int(values.shape[0]),
        int(layout["rho_time_feature_count"]),
        len(rho_modes),
        2,
    )
    _rotate_fourier_pairs(torch, rho, shifts, rho_modes)

    local_start = rho_end + int(layout["kinetic_global_latent_dim"])
    local_modes = _real_fourier_half_modes(
        cfg.structured_local_spatial_mode,
        cfg.structured_local_spatial_mode,
    )
    local = output[:, local_start:].reshape(
        int(values.shape[0]),
        int(cfg.structured_local_channels),
        int(layout["local_time_feature_count"]),
        int(layout["local_spatial_feature_count"]),
    )
    local_pairs = local[..., 1:].reshape(
        *local.shape[:-1], len(local_modes), 2
    )
    _rotate_fourier_pairs(torch, local_pairs, shifts, local_modes)
    return output[0] if squeeze else output


def _rotate_fourier_pairs(torch, pairs, shifts, modes) -> None:
    wave_numbers = torch.as_tensor(
        modes, dtype=pairs.dtype, device=pairs.device
    )
    phase = 2.0 * torch.pi * (shifts @ wave_numbers.T)
    prefix = (int(pairs.shape[0]),) + (1,) * (pairs.ndim - 3)
    cosine = torch.cos(phase).reshape(*prefix, len(modes))
    sine = torch.sin(phase).reshape(*prefix, len(modes))
    first = pairs[..., 0].clone()
    second = pairs[..., 1].clone()
    pairs[..., 0] = first * cosine + second * sine
    pairs[..., 1] = -first * sine + second * cosine

class VPFPContinuousHermiteField:
    @staticmethod
    def build(config: VPFPContinuousHermiteConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPContinuousHermiteConfig)
            else VPFPContinuousHermiteConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPContinuousHermiteField(_torch(), cfg)


class _VPFPContinuousHermiteField(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    """Continuous VPFP field with exact Poisson and squared-Hermite velocity density."""

    def __init__(self, torch_module, config: VPFPContinuousHermiteConfig):
        super().__init__()
        torch = torch_module
        self.torch = torch
        self.config = config
        self.hermite_modes = total_degree_mode_pairs(
            config.hermite_total_degree
        )
        self.rho_half_modes = _real_fourier_half_modes(
            config.rho_max_mode_x,
            config.rho_max_mode_y,
        )
        time_dim = _time_feature_dim(config.time_feature_modes)
        coordinate_dim = time_dim + 4 * int(config.spatial_feature_modes)
        self.structured_layout = (
            config.structured_latent_layout()
            if config.structured_spectral_latent
            else None
        )
        self.local_half_modes = (
            _real_fourier_half_modes(
                config.structured_local_spatial_mode,
                config.structured_local_spatial_mode,
            )
            if config.structured_spectral_latent
            else ()
        )
        if self.structured_layout is None:
            context_latent_dim = int(config.latent_dim)
            self.rho_network = _ResidualMLP(
                torch,
                config.latent_dim
                + len(EQUATION_CONDITION_KEYS)
                + time_dim,
                2 * len(self.rho_half_modes),
                config.hidden_width,
                config.hidden_layers,
            )
        else:
            context_latent_dim = int(
                config.structured_kinetic_global_dim
            )
            if not config.structured_direct_coefficients:
                context_latent_dim += int(
                    config.structured_local_channels
                )
            self.rho_network = None
        common_input = (
            context_latent_dim
            + len(EQUATION_CONDITION_KEYS)
            + coordinate_dim
        )
        self.context_network = _ResidualMLP(
            torch,
            common_input,
            config.context_dim,
            config.hidden_width,
            config.hidden_layers,
        )
        self.micro_context_network = _ResidualMLP(
            torch,
            common_input + config.micro_latent_dim,
            config.context_dim,
            config.hidden_width,
            max(2, config.hidden_layers // 2),
        )
        self.coefficient_network = _ResidualMLP(
            torch,
            config.context_dim,
            len(self.hermite_modes),
            config.coefficient_width,
            config.coefficient_blocks,
        )
        kappa = []
        lx, ly = config.equation.x_domain
        for kx, ky in self.rho_half_modes:
            kappa.append(
                (
                    2.0 * np.pi * float(kx) / float(lx),
                    2.0 * np.pi * float(ky) / float(ly),
                )
            )
        self.register_buffer(
            "_rho_kappa",
            torch.as_tensor(kappa),
            persistent=False,
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(
        self,
        points,
        latent,
        micro_latent=None,
        equation_condition=None,
    ):
        return self.fields(
            points,
            latent,
            micro_latent=micro_latent,
            equation_condition=equation_condition,
        )

    def fields(
        self,
        points,
        latent,
        *,
        micro_latent=None,
        equation_condition=None,
    ) -> dict[str, Any]:
        torch = self.torch
        parameter = next(self.parameters())
        query = torch.as_tensor(
            points, dtype=parameter.dtype, device=parameter.device
        )
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or int(query.shape[-1]) != 5:
            raise ValueError(
                "points must have shape [N,5] in (t,x,y,vx,vy) order"
            )
        count = int(query.shape[0])
        latent_rows = _broadcast_rows(
            torch,
            latent,
            count=count,
            width=self.config.latent_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            name="latent",
        )
        latent_scale = float(self.config.latent_input_scale)
        if (
            self.structured_layout is not None
            and count > 1
            and int(latent_rows.stride(0)) == 0
        ):
            latent_rows = (latent_scale * latent_rows[:1]).expand(count, -1)
        else:
            latent_rows = latent_scale * latent_rows
        condition_rows, condition_features = self._equation_condition_rows(
            equation_condition,
            count=count,
        )
        time_features, coordinate_features, tau = self._coordinate_features(
            query
        )
        local_features = None
        if self.structured_layout is None:
            raw_rho_coefficients = self.rho_network(
                torch.cat(
                    (latent_rows, condition_features, time_features),
                    dim=-1,
                )
            )
            context_latent_features = latent_rows
        else:
            (
                raw_rho_coefficients,
                kinetic_global_features,
                local_features,
            ) = self._structured_latent_features(
                query,
                latent_rows,
                tau,
            )
            context_parts = [kinetic_global_features]
            if not self.config.structured_direct_coefficients:
                context_parts.append(local_features)
            context_latent_features = torch.cat(
                context_parts,
                dim=-1,
            )
        rho, phi, electric, rho_coefficients = self._rho_poisson(
            query,
            raw_rho_coefficients,
        )
        micro_envelope = torch.zeros_like(tau)
        direct_projection = (
            self.config.structured_direct_coefficients
            and float(self.config.structured_residual_scale) == 0.0
        )
        if direct_projection:
            raw_coefficients = local_features
        else:
            common_features = torch.cat(
                (
                    context_latent_features,
                    condition_features,
                    coordinate_features,
                ),
                dim=-1,
            )
            context = self.context_network(common_features)
            if micro_latent is not None and self.config.micro_scale > 0.0:
                micro_rows = self._bounded_micro_latent(
                    _broadcast_rows(
                        torch,
                        micro_latent,
                        count=count,
                        width=self.config.micro_latent_dim,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        name="micro_latent",
                    )
                )
                micro_envelope = torch.sin(0.5 * torch.pi * tau).square()
                context = context + (
                    float(self.config.micro_scale)
                    * micro_envelope
                    * self.micro_context_network(
                        torch.cat((common_features, micro_rows), dim=-1)
                    )
                )
            coefficient_residual = self.coefficient_network(context)
            if self.config.structured_direct_coefficients:
                raw_coefficients = local_features + (
                    float(self.config.structured_residual_scale)
                    * coefficient_residual
                )
            else:
                raw_coefficients = coefficient_residual
        if self.config.structured_direct_coefficients:
            coefficients = raw_coefficients
        else:
            coefficients = torch.cat(
                (
                    torch.nn.functional.softplus(raw_coefficients[:, :1])
                    + float(self.config.coefficient_anchor),
                    raw_coefficients[:, 1:],
                ),
                dim=-1,
            )
        coefficient_norm = torch.linalg.vector_norm(
            coefficients,
            dim=-1,
            keepdim=True,
        )
        normalized_coefficients = coefficients / coefficient_norm.clamp_min(
            torch.finfo(coefficients.dtype).tiny
        )
        if self.config.structured_direct_coefficients:
            ground_state = torch.zeros_like(normalized_coefficients)
            ground_state[:, 0] = 1.0
            normalized_coefficients = torch.where(
                coefficient_norm > torch.finfo(coefficients.dtype).eps,
                normalized_coefficients,
                ground_state,
            )
        g = torch_squared_hermite_density(
            torch,
            query[:, 3:5],
            normalized_coefficients,
            temperature=condition_rows[:, 1],
            modes=self.hermite_modes,
            floor_mixture=self.config.velocity_floor_mixture,
        )
        maxwellian = self.maxwellian(
            query[:, 3],
            query[:, 4],
            condition_rows[:, 1],
        )
        return {
            "rho": rho,
            "g": g,
            "f": rho * g,
            "phi": phi,
            "electric": electric,
            "maxwellian": maxwellian,
            "hermite_coefficients": normalized_coefficients,
            "rho_coefficients": rho_coefficients,
            "micro_envelope": micro_envelope,
            "equation_condition": condition_rows,
        }

    def maxwellian(self, vx, vy, temperature=None):
        theta = (
            self.torch.as_tensor(
                float(self.config.equation.temperature),
                dtype=vx.dtype,
                device=vx.device,
            )
            if temperature is None
            else self.torch.as_tensor(
                temperature,
                dtype=vx.dtype,
                device=vx.device,
            )
        )
        return self.torch.exp(
            -0.5 * (vx.square() + vy.square()) / theta
        ) / (2.0 * self.torch.pi * theta)

    def equation_metadata(self) -> dict[str, Any]:
        return {
            **self.public_equation_metadata(),
            "representation": self.config.representation,
            "normalization": {
                "rho_spatial_mean": float(
                    self.config.equation.background_density
                ),
                "rho_positivity": "analytic_bounded_fourier_coefficients",
                "rho_positivity_gauge": self.config.rho_positivity_gauge,
                "g_velocity_integral": 1.0,
                "g_normalization_backend": (
                    "orthonormal_Hermite_Parseval_identity_on_R2"
                ),
            },
            "rho_fourier_half_mode_count": len(self.rho_half_modes),
            "velocity_representation": "conditional_squared_Hermite_expansion",
            "velocity_hermite_total_degree": int(
                self.config.hermite_total_degree
            ),
            "velocity_hermite_mode_count": len(self.hermite_modes),
            "intrinsic_latent_dim": int(self.config.latent_dim),
            "structured_spectral_latent": bool(
                self.config.structured_spectral_latent
            ),
            "structured_latent_layout": self.structured_layout,
        }

    def public_equation_metadata(self) -> dict[str, Any]:
        return {
            **self.config.equation.to_dict(),
            "domain": {
                "space": "two_dimensional_torus",
                "velocity": "R^2",
                "velocity_truncation_role": "numerical_query_window_only",
                "time": [float(value) for value in self.config.time_domain],
            },
            "collision_operator": (
                "nu * div_v(v*f + temperature*grad_v(f))"
            ),
            "poisson_equation": (
                "-Delta_x(phi)=rho-background_density; E=-grad_x(phi)"
            ),
            "normalization": {
                "rho_spatial_mean": float(
                    self.config.equation.background_density
                ),
                "g_velocity_integral": 1.0,
            },
        }

    def _structured_latent_features(self, points, latent_rows, tau):
        if self.structured_layout is None:
            raise RuntimeError("structured latent layout is not enabled")
        torch = self.torch
        layout = self.structured_layout
        count = int(points.shape[0])
        shared_latent = count > 1 and int(latent_rows.stride(0)) == 0
        latent_values = latent_rows[:1] if shared_latent else latent_rows
        latent_count = 1 if shared_latent else count
        cursor = 0

        rho_end = cursor + int(layout["rho_latent_dim"])
        rho_values = latent_values[:, cursor:rho_end].reshape(
            latent_count,
            int(layout["rho_time_feature_count"]),
            2 * len(self.rho_half_modes),
        )
        rho_time_basis, rho_time_normalization = self._structured_time_basis(
            tau,
            int(self.config.structured_rho_time_order),
        )
        if shared_latent:
            raw_rho_coefficients = torch.einsum(
                "nt,tm->nm", rho_time_basis, rho_values[0]
            )
        else:
            raw_rho_coefficients = torch.einsum(
                "nt,ntm->nm", rho_time_basis, rho_values
            )
        raw_rho_coefficients = raw_rho_coefficients / float(
            rho_time_normalization
        )
        cursor = rho_end

        kinetic_end = (
            cursor + int(self.config.structured_kinetic_global_dim)
        )
        kinetic_global = latent_values[:, cursor:kinetic_end]
        if shared_latent:
            kinetic_global = kinetic_global.expand(count, -1)
        cursor = kinetic_end

        local_values = latent_values[:, cursor:].reshape(
            latent_count,
            int(self.config.structured_local_channels),
            int(layout["local_time_feature_count"]),
            int(layout["local_spatial_feature_count"]),
        )
        local_time_basis, local_time_normalization = self._structured_time_basis(
            tau,
            int(self.config.structured_local_time_order),
        )
        local_spatial_basis = self._local_spatial_basis(points)
        if shared_latent:
            local_by_point_time = torch.einsum(
                "cts,ns->nct", local_values[0], local_spatial_basis
            )
            local_features = torch.sum(
                local_by_point_time * local_time_basis[:, None, :], dim=-1
            )
        else:
            local_features = torch.einsum(
                "nt,ncts,ns->nc",
                local_time_basis,
                local_values,
                local_spatial_basis,
            )
        local_features = (
            float(self.config.structured_local_scale)
            * local_features
            / np.sqrt(float(local_spatial_basis.shape[-1]))
            / float(local_time_normalization)
        )
        return raw_rho_coefficients, kinetic_global, local_features

    def _structured_time_basis(self, tau, order: int):
        if self.config.structured_time_basis == "global_chebyshev_v1":
            basis = self._chebyshev_basis(tau, order)
            return basis, np.sqrt(float(basis.shape[-1]))
        control_count = int(self.config.structured_time_control_count)
        unit_position = self.torch.clamp(
            tau[:, 0], min=0.0, max=1.0
        ) * float(control_count - 1)
        lower = self.torch.floor(unit_position).to(
            dtype=self.torch.long
        ).clamp(max=control_count - 2)
        local = unit_position - lower.to(dtype=unit_position.dtype)
        local_squared = local.square()
        local_cubed = local_squared * local
        h00 = 2.0 * local_cubed - 3.0 * local_squared + 1.0
        h10 = local_cubed - 2.0 * local_squared + local
        h01 = -2.0 * local_cubed + 3.0 * local_squared
        h11 = local_cubed - local_squared
        value_basis = self.torch.zeros(
            (int(tau.shape[0]), control_count),
            dtype=tau.dtype,
            device=tau.device,
        )
        slope_basis = self.torch.zeros_like(value_basis)
        rows = self.torch.arange(int(tau.shape[0]), device=tau.device)
        value_basis[rows, lower] = h00
        value_basis[rows, lower + 1] = h01
        interval = 1.0 / float(control_count - 1)
        slope_basis[rows, lower] = interval * h10
        slope_basis[rows, lower + 1] = interval * h11
        return self.torch.cat((value_basis, slope_basis), dim=-1), 1.0

    def _chebyshev_basis(self, tau, order: int):
        scaled = 2.0 * tau - 1.0
        values = [self.torch.ones_like(scaled)]
        if int(order) >= 1:
            values.append(scaled)
        for _ in range(1, int(order)):
            values.append(2.0 * scaled * values[-1] - values[-2])
        return self.torch.cat(values, dim=-1)

    def _local_spatial_basis(self, points):
        torch = self.torch
        x = points[:, 1:2]
        y = points[:, 2:3]
        lx, ly = self.config.equation.x_domain
        values = [torch.ones_like(x)]
        for kx, ky in self.local_half_modes:
            phase = 2.0 * torch.pi * (
                float(kx) * x / float(lx)
                + float(ky) * y / float(ly)
            )
            values.extend(
                (
                    np.sqrt(2.0) * torch.cos(phase),
                    np.sqrt(2.0) * torch.sin(phase),
                )
            )
        basis = torch.cat(values, dim=-1)
        expected = int(
            self.structured_layout["local_spatial_feature_count"]
        )
        if int(basis.shape[-1]) != expected:
            raise RuntimeError("structured spatial basis size mismatch")
        return basis

    def _bounded_micro_latent(self, micro_latent):
        radius = float(self.config.micro_latent_radius)
        norm = self.torch.linalg.vector_norm(
            micro_latent,
            dim=-1,
            keepdim=True,
        )
        return micro_latent / self.torch.clamp(norm / radius, min=1.0)

    def _equation_condition_rows(self, equation_condition, *, count: int):
        parameter = next(self.parameters())
        if equation_condition is None:
            physical = self.torch.as_tensor(
                [
                    float(self.config.equation.collision_nu),
                    float(self.config.equation.temperature),
                ],
                dtype=parameter.dtype,
                device=parameter.device,
            )
        else:
            physical = equation_condition
        rows = _broadcast_rows(
            self.torch,
            physical,
            count=int(count),
            width=len(EQUATION_CONDITION_KEYS),
            dtype=parameter.dtype,
            device=parameter.device,
            name="equation_condition",
        )
        if bool(self.torch.any(rows[:, 0] < 0.0)):
            raise ValueError("conditioned collision_nu must be nonnegative")
        if bool(self.torch.any(rows[:, 1] <= 0.0)):
            raise ValueError("conditioned temperature must be positive")
        lower = self.torch.as_tensor(
            [
                self.config.collision_nu_range[0],
                self.config.temperature_range[0],
            ],
            dtype=rows.dtype,
            device=rows.device,
        )
        upper = self.torch.as_tensor(
            [
                self.config.collision_nu_range[1],
                self.config.temperature_range[1],
            ],
            dtype=rows.dtype,
            device=rows.device,
        )
        normalized = 2.0 * (rows - lower) / (upper - lower) - 1.0
        return rows, normalized

    def _coordinate_features(self, points):
        torch = self.torch
        t = points[:, 0:1]
        t0, t1 = self.config.time_domain
        tau = (t - float(t0)) / float(t1 - t0)
        time_features = [tau]
        for mode in range(1, self.config.time_feature_modes + 1):
            phase = float(mode) * torch.pi * tau
            time_features.extend((torch.sin(phase), torch.cos(phase)))
        time_values = torch.cat(time_features, dim=-1)
        spatial = []
        x = points[:, 1:2]
        y = points[:, 2:3]
        lx, ly = self.config.equation.x_domain
        for mode in range(1, self.config.spatial_feature_modes + 1):
            x_phase = 2.0 * torch.pi * float(mode) * x / float(lx)
            y_phase = 2.0 * torch.pi * float(mode) * y / float(ly)
            spatial.extend(
                (
                    torch.sin(x_phase),
                    torch.cos(x_phase),
                    torch.sin(y_phase),
                    torch.cos(y_phase),
                )
            )
        return (
            time_values,
            torch.cat((time_values, *spatial), dim=-1),
            tau,
        )

    def _rho_poisson(self, points, raw_coefficients):
        torch = self.torch
        count = int(points.shape[0])
        raw = raw_coefficients.reshape(
            count,
            len(self.rho_half_modes),
            2,
        )
        if self.config.rho_positivity_gauge == "phase_invariant_group_l1_v2":
            denominator = 1.0 + torch.sum(
                torch.linalg.vector_norm(raw, dim=-1),
                dim=1,
                keepdim=True,
            )[..., None]
        else:
            denominator = 1.0 + torch.sum(
                torch.abs(raw),
                dim=(1, 2),
                keepdim=True,
            )
        scale = (
            float(self.config.rho_contrast)
            * float(self.config.equation.background_density)
        )
        coefficients = scale * raw / denominator
        kappa = self._rho_kappa.to(
            dtype=points.dtype,
            device=points.device,
        )
        phase = (
            points[:, 1:2] * kappa[None, :, 0]
            + points[:, 2:3] * kappa[None, :, 1]
        )
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        perturbation = torch.sum(
            coefficients[:, :, 0] * cosine
            + coefficients[:, :, 1] * sine,
            dim=-1,
        )
        rho = (
            float(self.config.equation.background_density)
            + perturbation
        )
        kappa_squared = torch.sum(kappa.square(), dim=-1)
        phi = torch.sum(
            (
                coefficients[:, :, 0] * cosine
                + coefficients[:, :, 1] * sine
            )
            / kappa_squared[None, :],
            dim=-1,
        )
        electric_x = torch.sum(
            (
                coefficients[:, :, 0] * kappa[None, :, 0] * sine
                - coefficients[:, :, 1] * kappa[None, :, 0] * cosine
            )
            / kappa_squared[None, :],
            dim=-1,
        )
        electric_y = torch.sum(
            (
                coefficients[:, :, 0] * kappa[None, :, 1] * sine
                - coefficients[:, :, 1] * kappa[None, :, 1] * cosine
            )
            / kappa_squared[None, :],
            dim=-1,
        )
        return (
            rho,
            phi,
            torch.stack((electric_x, electric_y), dim=-1),
            coefficients,
        )
