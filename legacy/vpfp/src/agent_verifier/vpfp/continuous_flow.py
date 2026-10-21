from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_v2 import (
    STANDARD_VPFP_EQUATION_VERSION,
    VPFPStandardEquationConfig,
    _ResidualMLP,
    _broadcast_rows,
    _real_fourier_half_modes,
    _time_feature_dim,
    _torch,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


CONTINUOUS_FLOW_REPRESENTATION = (
    "bounded_fourier_rho_conditional_fullspace_velocity_flow_v3"
)


EQUATION_CONDITION_KEYS = ("collision_nu", "temperature")


@dataclass(frozen=True)
class VPFPContinuousFlowConfig:
    latent_dim: int = 256
    latent_input_scale: float = 2.0
    micro_latent_dim: int = 64
    context_dim: int = 128
    hidden_width: int = 256
    hidden_layers: int = 6
    rho_max_mode_x: int = 10
    rho_max_mode_y: int = 10
    rho_contrast: float = 0.80
    spatial_feature_modes: int = 8
    time_feature_modes: int = 8
    flow_components: int = 4
    flow_layers: int = 10
    flow_width: int = 192
    flow_blocks: int = 3
    flow_max_log_scale: float = 1.5
    velocity_floor_mixture: float = 1.0e-4
    micro_scale: float = 0.08
    micro_latent_radius: float = 3.0
    collision_nu_range: tuple[float, float] = (0.02, 0.08)
    temperature_range: tuple[float, float] = (0.8, 1.2)
    time_domain: tuple[float, float] = (0.0, 0.35)
    dtype: str = "float32"
    representation: str = CONTINUOUS_FLOW_REPRESENTATION
    equation: VPFPStandardEquationConfig = VPFPStandardEquationConfig()

    def validate(self) -> None:
        self.equation.validate()
        if min(
            self.latent_dim,
            self.micro_latent_dim,
            self.context_dim,
            self.hidden_width,
            self.hidden_layers,
            self.flow_components,
            self.flow_layers,
            self.flow_width,
            self.flow_blocks,
        ) < 1:
            raise ValueError("continuous-flow network dimensions must be positive")
        if self.latent_input_scale <= 0.0:
            raise ValueError("latent_input_scale must be positive")
        if min(self.rho_max_mode_x, self.rho_max_mode_y) < 1:
            raise ValueError("rho Fourier mode cutoffs must be positive")
        if not 0.0 < self.rho_contrast < 1.0:
            raise ValueError("rho_contrast must lie in (0, 1)")
        if min(self.spatial_feature_modes, self.time_feature_modes) < 1:
            raise ValueError("coordinate feature mode counts must be positive")
        if self.flow_max_log_scale <= 0.0:
            raise ValueError("flow_max_log_scale must be positive")
        if not 0.0 <= self.velocity_floor_mixture < 1.0:
            raise ValueError("velocity_floor_mixture must lie in [0, 1)")
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
    def from_dict(cls, data: dict[str, Any]) -> "VPFPContinuousFlowConfig":
        payload = dict(data)
        for key in ("collision_nu_range", "temperature_range"):
            if key in payload:
                payload[key] = tuple(float(value) for value in payload[key])
        if "time_domain" in payload:
            payload["time_domain"] = tuple(
                float(value) for value in payload["time_domain"]
            )
        payload["equation"] = VPFPStandardEquationConfig.from_dict(
            payload.get("equation")
        )
        return cls(**payload)


class VPFPContinuousFlowField:
    @staticmethod
    def build(config: VPFPContinuousFlowConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPContinuousFlowConfig)
            else VPFPContinuousFlowConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPContinuousFlowField(_torch(), cfg)


class _AffineCoupling(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(
        self,
        torch_module,
        *,
        context_dim: int,
        active_index: int,
        width: int,
        blocks: int,
        max_log_scale: float,
    ):
        super().__init__()
        self.active_index = int(active_index)
        self.passive_index = 1 - int(active_index)
        self.max_log_scale = float(max_log_scale)
        self.conditioner = _ResidualMLP(
            torch_module,
            int(context_dim) + 1,
            2,
            int(width),
            int(blocks),
        )

    def parameters_from(self, context, passive):
        raw = self.conditioner(
            _torch().cat((context, passive), dim=-1)
        )
        shift = raw[:, :1]
        log_scale = self.max_log_scale * _torch().tanh(raw[:, 1:2])
        return shift, log_scale

    def inverse(self, velocity, context):
        passive = velocity[:, self.passive_index : self.passive_index + 1]
        active = velocity[:, self.active_index : self.active_index + 1]
        shift, log_scale = self.parameters_from(context, passive)
        transformed = (active - shift) * _torch().exp(-log_scale)
        if self.active_index == 0:
            output = _torch().cat((transformed, passive), dim=-1)
        else:
            output = _torch().cat((passive, transformed), dim=-1)
        return output, -log_scale[:, 0]

    def forward_transform(self, velocity, context):
        passive = velocity[:, self.passive_index : self.passive_index + 1]
        active = velocity[:, self.active_index : self.active_index + 1]
        shift, log_scale = self.parameters_from(context, passive)
        transformed = active * _torch().exp(log_scale) + shift
        if self.active_index == 0:
            return _torch().cat((transformed, passive), dim=-1)
        return _torch().cat((passive, transformed), dim=-1)


class _VPFPContinuousFlowField(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    """Continuous VPFP field with exact torus Poisson and normalized R2 velocity flow."""

    def __init__(self, torch_module, config: VPFPContinuousFlowConfig):
        super().__init__()
        torch = torch_module
        self.torch = torch
        self.config = config
        self.rho_half_modes = _real_fourier_half_modes(
            config.rho_max_mode_x,
            config.rho_max_mode_y,
        )
        time_dim = _time_feature_dim(config.time_feature_modes)
        coordinate_dim = time_dim + 4 * int(config.spatial_feature_modes)
        self.rho_network = _ResidualMLP(
            torch,
            config.latent_dim + len(EQUATION_CONDITION_KEYS) + time_dim,
            2 * len(self.rho_half_modes),
            config.hidden_width,
            config.hidden_layers,
        )
        self.context_network = _ResidualMLP(
            torch,
            config.latent_dim + len(EQUATION_CONDITION_KEYS) + coordinate_dim,
            config.context_dim,
            config.hidden_width,
            config.hidden_layers,
        )
        self.micro_context_network = _ResidualMLP(
            torch,
            (
                config.latent_dim
                + config.micro_latent_dim
                + len(EQUATION_CONDITION_KEYS)
                + coordinate_dim
            ),
            config.context_dim,
            config.hidden_width,
            max(2, config.hidden_layers // 2),
        )
        self.flow_components = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        _AffineCoupling(
                            torch,
                            context_dim=config.context_dim,
                            active_index=layer % 2,
                            width=config.flow_width,
                            blocks=config.flow_blocks,
                            max_log_scale=config.flow_max_log_scale,
                        )
                        for layer in range(config.flow_layers)
                    ]
                )
                for _ in range(config.flow_components)
            ]
        )
        self.mixture_network = _ResidualMLP(
            torch,
            config.context_dim,
            config.flow_components,
            max(32, config.flow_width // 2),
            max(1, config.flow_blocks - 1),
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
        self, points, latent, *, micro_latent=None, equation_condition=None
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
        latent_rows = float(self.config.latent_input_scale) * latent_rows
        condition_rows, condition_features = self._equation_condition_rows(
            equation_condition, count=count
        )
        time_features, coordinate_features, tau = self._coordinate_features(
            query
        )
        raw_rho_coefficients = self.rho_network(
            torch.cat((latent_rows, condition_features, time_features), dim=-1)
        )
        rho, phi, electric, rho_coefficients = self._rho_poisson(
            query, raw_rho_coefficients
        )
        context = self.context_network(
            torch.cat(
                (latent_rows, condition_features, coordinate_features), dim=-1
            )
        )
        micro_envelope = torch.zeros_like(tau)
        if micro_latent is not None and self.config.micro_scale > 0.0:
            micro_rows = self._bounded_micro_latent(_broadcast_rows(
                torch,
                micro_latent,
                count=count,
                width=self.config.micro_latent_dim,
                dtype=parameter.dtype,
                device=parameter.device,
                name="micro_latent",
            ))
            micro_envelope = torch.sin(0.5 * torch.pi * tau).square()
            context = context + (
                float(self.config.micro_scale)
                * micro_envelope
                * self.micro_context_network(
                    torch.cat(
                        (
                            latent_rows,
                            micro_rows,
                            condition_features,
                            coordinate_features,
                        ),
                        dim=-1,
                    )
                )
            )
        flow_density, flow_log_density, flow_weights = self._velocity_density(
            query[:, 3:5], context, condition_rows[:, 1]
        )
        maxwellian = self.maxwellian(
            query[:, 3], query[:, 4], condition_rows[:, 1]
        )
        floor = float(self.config.velocity_floor_mixture)
        g = (1.0 - floor) * flow_density + floor * maxwellian
        f = rho * g
        return {
            "rho": rho,
            "g": g,
            "f": f,
            "phi": phi,
            "electric": electric,
            "maxwellian": maxwellian,
            "flow_log_density": flow_log_density,
            "flow_component_weights": flow_weights,
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
                temperature, dtype=vx.dtype, device=vx.device
            )
        )
        return self.torch.exp(
            -0.5 * (vx.square() + vy.square()) / theta
        ) / (2.0 * self.torch.pi * theta)

    def sample_velocity(
        self,
        points_txy,
        latent,
        *,
        micro_latent=None,
        equation_condition=None,
        generator=None,
    ):
        torch = self.torch
        parameter = next(self.parameters())
        txy = torch.as_tensor(
            points_txy, dtype=parameter.dtype, device=parameter.device
        )
        if txy.ndim == 1:
            txy = txy[None, :]
        if txy.ndim != 2 or int(txy.shape[-1]) != 3:
            raise ValueError("points_txy must have shape [N,3]")
        query = torch.cat(
            (txy, torch.zeros((txy.shape[0], 2), dtype=txy.dtype, device=txy.device)),
            dim=-1,
        )
        latent_rows = _broadcast_rows(
            torch,
            latent,
            count=int(query.shape[0]),
            width=self.config.latent_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            name="latent",
        )
        latent_rows = float(self.config.latent_input_scale) * latent_rows
        condition_rows, condition_features = self._equation_condition_rows(
            equation_condition, count=int(query.shape[0])
        )
        _, coordinate_features, tau = self._coordinate_features(query)
        context = self.context_network(
            torch.cat(
                (latent_rows, condition_features, coordinate_features), dim=-1
            )
        )
        if micro_latent is not None and self.config.micro_scale > 0.0:
            micro_rows = self._bounded_micro_latent(_broadcast_rows(
                torch,
                micro_latent,
                count=int(query.shape[0]),
                width=self.config.micro_latent_dim,
                dtype=parameter.dtype,
                device=parameter.device,
                name="micro_latent",
            ))
            envelope = torch.sin(0.5 * torch.pi * tau).square()
            context = context + (
                float(self.config.micro_scale)
                * envelope
                * self.micro_context_network(
                    torch.cat(
                        (
                            latent_rows,
                            micro_rows,
                            condition_features,
                            coordinate_features,
                        ),
                        dim=-1,
                    )
                )
            )
        theta = condition_rows[:, 1]
        base_velocity = torch.randn(
            (query.shape[0], 2),
            dtype=query.dtype,
            device=query.device,
            generator=generator,
        ) * torch.sqrt(theta)[:, None]
        mixture_weights = torch.softmax(self.mixture_network(context), dim=-1)
        selected = torch.multinomial(
            mixture_weights, 1, replacement=True, generator=generator
        )[:, 0]
        velocity = torch.zeros_like(base_velocity)
        for component_index, couplings in enumerate(self.flow_components):
            component_velocity = base_velocity
            for coupling in couplings:
                component_velocity = coupling.forward_transform(
                    component_velocity, context
                )
            velocity = torch.where(
                (selected == component_index)[:, None],
                component_velocity,
                velocity,
            )
        floor = float(self.config.velocity_floor_mixture)
        if floor > 0.0:
            choose_floor = torch.rand(
                (query.shape[0],),
                dtype=query.dtype,
                device=query.device,
                generator=generator,
            ) < floor
            floor_velocity = torch.randn(
                (query.shape[0], 2),
                dtype=query.dtype,
                device=query.device,
                generator=generator,
            ) * torch.sqrt(theta)[:, None]
            velocity = torch.where(
                choose_floor[:, None], floor_velocity, velocity
            )
        return velocity

    def equation_metadata(self) -> dict[str, Any]:
        return {
            **self.config.equation.to_dict(),
            "representation": self.config.representation,
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
                "rho_positivity": "analytic_l1_bounded_fourier_coefficients",
                "g_velocity_integral": 1.0,
                "g_normalization_backend": (
                    "conditional_normalizing_flow_change_of_variables"
                ),
            },
            "rho_fourier_half_mode_count": len(self.rho_half_modes),
            "velocity_representation": "conditional_affine_normalizing_flow",
            "velocity_flow_components": int(self.config.flow_components),
            "velocity_flow_layers": int(self.config.flow_layers),
            "intrinsic_latent_dim": int(self.config.latent_dim),
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

    def _velocity_density(self, velocity, context, temperature):
        theta = self.torch.as_tensor(
            temperature, dtype=velocity.dtype, device=velocity.device
        )
        component_logs = []
        for couplings in self.flow_components:
            transformed = velocity
            log_determinant = self.torch.zeros_like(velocity[:, 0])
            for coupling in reversed(couplings):
                transformed, contribution = coupling.inverse(
                    transformed, context
                )
                log_determinant = log_determinant + contribution
            base_log_density = (
                -0.5 * self.torch.sum(transformed.square(), dim=-1) / theta
                - self.torch.log(2.0 * self.torch.pi * theta)
            )
            component_logs.append(base_log_density + log_determinant)
        component_log_density = self.torch.stack(component_logs, dim=-1)
        log_weights = self.torch.log_softmax(
            self.mixture_network(context), dim=-1
        )
        log_density = self.torch.logsumexp(
            log_weights + component_log_density, dim=-1
        )
        return self.torch.exp(log_density), log_density, self.torch.exp(log_weights)

    def _bounded_micro_latent(self, micro_latent):
        radius = float(self.config.micro_latent_radius)
        norm = self.torch.linalg.vector_norm(
            micro_latent, dim=-1, keepdim=True
        )
        scale = self.torch.clamp(norm / radius, min=1.0)
        return micro_latent / scale

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
        raw = raw_coefficients.reshape(count, len(self.rho_half_modes), 2)
        denominator = 1.0 + torch.sum(torch.abs(raw), dim=(1, 2), keepdim=True)
        scale = (
            float(self.config.rho_contrast)
            * float(self.config.equation.background_density)
        )
        coefficients = scale * raw / denominator
        kappa = self._rho_kappa.to(
            dtype=points.dtype, device=points.device
        )
        phase = (
            points[:, 1:2] * kappa[None, :, 0]
            + points[:, 2:3] * kappa[None, :, 1]
        )
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        rho_perturbation = torch.sum(
            coefficients[:, :, 0] * cosine
            + coefficients[:, :, 1] * sine,
            dim=-1,
        )
        rho = (
            float(self.config.equation.background_density)
            + rho_perturbation
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
