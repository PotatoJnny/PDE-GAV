from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


STANDARD_VPFP_EQUATION_VERSION = "vpfp_ou_poisson_torus_rv_v2"
CONTINUOUS_VPFP_REPRESENTATION = "positive_fourier_fullspace_velocity_neural_field_v2"
VELOCITY_SQUARE_HERMITE = "square_hermite"
VELOCITY_DENSITY_DICTIONARY = "normalized_gaussian_density_dictionary"


@dataclass(frozen=True)
class VPFPStandardEquationConfig:
    """Dimensionless electrostatic VPFP equation on T^2_x times R^2_v."""

    collision_nu: float = 0.05
    temperature: float = 1.0
    background_density: float = 1.0
    x_domain: tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi)

    def validate(self) -> None:
        if self.collision_nu < 0.0:
            raise ValueError("collision_nu must be nonnegative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.background_density <= 0.0:
            raise ValueError("background_density must be positive")
        if min(self.x_domain) <= 0.0:
            raise ValueError("x_domain lengths must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["x_domain"] = [float(value) for value in self.x_domain]
        payload["equation_version"] = STANDARD_VPFP_EQUATION_VERSION
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VPFPStandardEquationConfig":
        payload = dict(data or {})
        payload.pop("equation_version", None)
        if "x_domain" in payload:
            payload["x_domain"] = tuple(float(value) for value in payload["x_domain"])
        return cls(**payload)


@dataclass(frozen=True)
class VPFPContinuousFieldConfig:
    latent_dim: int = 32
    micro_latent_dim: int = 16
    hidden_width: int = 192
    hidden_layers: int = 5
    rho_max_mode_x: int = 3
    rho_max_mode_y: int = 3
    velocity_total_degree: int = 8
    velocity_representation: str = VELOCITY_SQUARE_HERMITE
    velocity_density_atom_count: int = 64
    density_atom_center_scale: float = 2.0
    density_atom_std_range: tuple[float, float] = (0.35, 1.50)
    micro_density_atom_fraction: float = 0.25
    spatial_feature_modes: int = 4
    time_feature_modes: int = 4
    time_domain: tuple[float, float] = (0.0, 0.5)
    rho_floor: float = 0.02
    velocity_floor_mixture: float = 1.0e-4
    coefficient_floor: float = 0.25
    micro_min_velocity_degree: int = 4
    micro_scale: float = 0.08
    dtype: str = "float64"
    representation: str = CONTINUOUS_VPFP_REPRESENTATION
    equation: VPFPStandardEquationConfig = VPFPStandardEquationConfig()

    @property
    def velocity_basis_count(self) -> int:
        if self.velocity_representation == VELOCITY_DENSITY_DICTIONARY:
            return int(self.velocity_density_atom_count)
        degree = int(self.velocity_total_degree)
        return (degree + 1) * (degree + 2) // 2

    def validate(self) -> None:
        self.equation.validate()
        if self.latent_dim < 1 or self.micro_latent_dim < 1:
            raise ValueError("latent dimensions must be positive")
        if self.hidden_width < 4 or self.hidden_layers < 1:
            raise ValueError("hidden network must have positive capacity")
        if self.velocity_representation not in {
            VELOCITY_SQUARE_HERMITE,
            VELOCITY_DENSITY_DICTIONARY,
        }:
            raise ValueError("unsupported velocity_representation")
        if self.velocity_density_atom_count < 1:
            raise ValueError("velocity_density_atom_count must be positive")
        if self.density_atom_center_scale < 0.0:
            raise ValueError("density_atom_center_scale must be nonnegative")
        if not (0.0 < self.density_atom_std_range[0] <= self.density_atom_std_range[1]):
            raise ValueError("density_atom_std_range must be positive and ordered")
        if not 0.0 < self.micro_density_atom_fraction <= 1.0:
            raise ValueError("micro_density_atom_fraction must lie in (0, 1]")
        if min(self.rho_max_mode_x, self.rho_max_mode_y) < 0:
            raise ValueError("rho Fourier cutoffs must be nonnegative")
        if self.velocity_total_degree < 0:
            raise ValueError("velocity_total_degree must be nonnegative")
        if self.spatial_feature_modes < 1 or self.time_feature_modes < 0:
            raise ValueError("coordinate feature mode counts are invalid")
        if self.time_domain[1] <= self.time_domain[0]:
            raise ValueError("time_domain must satisfy t1 > t0")
        if not 0.0 <= self.rho_floor < self.equation.background_density:
            raise ValueError("rho_floor must lie below background_density")
        if not 0.0 < self.velocity_floor_mixture < 1.0:
            raise ValueError("velocity_floor_mixture must lie in (0, 1)")
        if self.coefficient_floor <= 0.0 or self.micro_scale < 0.0:
            raise ValueError("coefficient_floor and micro_scale must be nonnegative")
        if self.velocity_representation == VELOCITY_SQUARE_HERMITE and not (
            0 <= self.micro_min_velocity_degree <= self.velocity_total_degree + 1
        ):
            raise ValueError("micro_min_velocity_degree is outside the Hermite range")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["time_domain"] = [float(value) for value in self.time_domain]
        payload["density_atom_std_range"] = [
            float(value) for value in self.density_atom_std_range
        ]
        payload["equation"] = self.equation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPContinuousFieldConfig":
        payload = dict(data)
        if "time_domain" in payload:
            payload["time_domain"] = tuple(float(value) for value in payload["time_domain"])
        if "density_atom_std_range" in payload:
            payload["density_atom_std_range"] = tuple(
                float(value) for value in payload["density_atom_std_range"]
            )
        payload["equation"] = VPFPStandardEquationConfig.from_dict(payload.get("equation"))
        return cls(**payload)


class VPFPContinuousField:
    @staticmethod
    def build(config: VPFPContinuousFieldConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPContinuousFieldConfig)
            else VPFPContinuousFieldConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPContinuousField(_torch(), cfg)


class _ResidualMLP((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch_module, input_dim: int, output_dim: int, width: int, layers: int):
        super().__init__()
        torch = torch_module
        self.input = torch.nn.Linear(input_dim, width)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(width, width),
                    torch.nn.SiLU(),
                    torch.nn.Linear(width, width),
                )
                for _ in range(layers)
            ]
        )
        self.output = torch.nn.Linear(width, output_dim)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, values):
        torch = _torch()
        hidden = torch.nn.functional.silu(self.input(values))
        block_scale = float(max(len(self.blocks), 1)) ** -0.5
        for block in self.blocks:
            hidden = hidden + block_scale * block(hidden)
        return self.output(hidden)


class _VPFPContinuousField((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    """Positive, normalized continuous VPFP hidden field.

    rho is a squared real Fourier field with fixed spatial mean. g is either a
    Maxwellian-weighted squared Hermite expansion or a simplex mixture of
    normalized full-space Gaussian density atoms. Basis size is a capacity
    control, not a fixed low-dimensional physical ansatz.
    """

    def __init__(self, torch_module, config: VPFPContinuousFieldConfig):
        super().__init__()
        self.torch = torch_module
        self.config = config
        torch = torch_module

        self.rho_half_modes = _real_fourier_half_modes(
            config.rho_max_mode_x,
            config.rho_max_mode_y,
        )
        self.velocity_multi_indices = (
            _hermite_multi_indices(config.velocity_total_degree)
            if config.velocity_representation == VELOCITY_SQUARE_HERMITE
            else []
        )
        rho_output_dim = 1 + 2 * len(self.rho_half_modes)
        velocity_output_dim = config.velocity_basis_count
        time_dim = _time_feature_dim(config.time_feature_modes)
        coordinate_dim = time_dim + 4 * config.spatial_feature_modes

        self.rho_network = _ResidualMLP(
            torch,
            config.latent_dim + time_dim,
            rho_output_dim,
            config.hidden_width,
            config.hidden_layers,
        )
        self.velocity_network = _ResidualMLP(
            torch,
            config.latent_dim + coordinate_dim,
            velocity_output_dim,
            config.hidden_width,
            config.hidden_layers,
        )
        self.micro_network = _ResidualMLP(
            torch,
            config.latent_dim + config.micro_latent_dim + coordinate_dim,
            velocity_output_dim,
            config.hidden_width,
            max(2, config.hidden_layers // 2),
        )

        if config.velocity_representation == VELOCITY_DENSITY_DICTIONARY:
            atom_means, atom_stds = _density_dictionary_initial_values(config)
            self.density_atom_means = torch.nn.Parameter(
                torch.as_tensor(atom_means)
            )
            self.density_atom_log_stds = torch.nn.Parameter(
                torch.log(torch.as_tensor(atom_stds))
            )
        complex_modes, convolution_modes, pair_groups = _fourier_convolution_structure(
            self.rho_half_modes
        )
        self.register_buffer(
            "_complex_modes",
            torch.as_tensor(complex_modes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_convolution_modes",
            torch.as_tensor(convolution_modes, dtype=torch.int64),
            persistent=False,
        )
        self._pair_groups = pair_groups
        if config.velocity_representation == VELOCITY_SQUARE_HERMITE:
            micro_mask = [
                float(ax + ay >= config.micro_min_velocity_degree)
                for ax, ay in self.velocity_multi_indices
            ]
        else:
            active = max(
                1,
                int(np.ceil(config.velocity_basis_count * config.micro_density_atom_fraction)),
            )
            micro_mask = [0.0] * (config.velocity_basis_count - active) + [1.0] * active
        self.register_buffer(
            "_micro_mask",
            torch.as_tensor(micro_mask),
            persistent=False,
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(self, points, latent, micro_latent=None):
        return self.fields(points, latent, micro_latent=micro_latent)

    def fields(self, points, latent, *, micro_latent=None) -> dict[str, Any]:
        torch = self.torch
        parameter = next(self.parameters())
        query = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or int(query.shape[-1]) != 5:
            raise ValueError(f"points must have shape [N,5] in (t,x,y,vx,vy) order, got {tuple(query.shape)}")
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
        time_features, coordinate_features, tau = self._coordinate_features(query)

        rho_raw = self.rho_network(torch.cat((latent_rows, time_features), dim=-1))
        rho_coefficients = torch.cat(
            (
                torch.nn.functional.softplus(rho_raw[:, :1]) + self.config.coefficient_floor,
                rho_raw[:, 1:],
            ),
            dim=-1,
        )
        rho, phi, electric, rho_unit_coefficients = self._rho_poisson(
            query,
            rho_coefficients,
        )

        velocity_raw = self.velocity_network(
            torch.cat((latent_rows, coordinate_features), dim=-1)
        )
        if self.config.velocity_representation == VELOCITY_SQUARE_HERMITE:
            velocity_parameters = torch.cat(
                (
                    torch.nn.functional.softplus(velocity_raw[:, :1])
                    + self.config.coefficient_floor,
                    velocity_raw[:, 1:],
                ),
                dim=-1,
            )
        else:
            velocity_parameters = velocity_raw
        micro_envelope = torch.zeros_like(tau)
        if micro_latent is not None and self.config.micro_scale > 0.0:
            micro_rows = _broadcast_rows(
                torch,
                micro_latent,
                count=count,
                width=self.config.micro_latent_dim,
                dtype=parameter.dtype,
                device=parameter.device,
                name="micro_latent",
            )
            micro_raw = self.micro_network(
                torch.cat((latent_rows, micro_rows, coordinate_features), dim=-1)
            )
            # The perturbation and its first time derivative vanish at t=t0.
            micro_envelope = torch.sin(0.5 * torch.pi * tau).square()
            velocity_parameters = velocity_parameters + (
                self.config.micro_scale
                * micro_envelope
                * micro_raw
                * self._micro_mask.to(dtype=parameter.dtype)[None, :]
            )

        maxwellian = self.maxwellian(query[:, 3], query[:, 4])
        if self.config.velocity_representation == VELOCITY_SQUARE_HERMITE:
            coefficient_norm = torch.linalg.vector_norm(
                velocity_parameters,
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(parameter.dtype).eps)
            velocity_coefficients = velocity_parameters / coefficient_norm
            hermite = self._velocity_basis(query[:, 3], query[:, 4])
            amplitude = torch.sum(velocity_coefficients * hermite, dim=-1)
            floor = float(self.config.velocity_floor_mixture)
            g = maxwellian * (floor + (1.0 - floor) * amplitude.square())
        else:
            velocity_coefficients = torch.softmax(velocity_parameters, dim=-1)
            atoms = self._density_atom_values(query[:, 3], query[:, 4])
            g = torch.sum(velocity_coefficients * atoms, dim=-1)
        f = rho * g
        return {
            "rho": rho,
            "g": g,
            "f": f,
            "phi": phi,
            "electric": electric,
            "maxwellian": maxwellian,
            "rho_coefficients": rho_unit_coefficients,
            "velocity_coefficients": velocity_coefficients,
            "micro_envelope": micro_envelope,
        }

    def maxwellian(self, vx, vy):
        torch = self.torch
        theta = float(self.config.equation.temperature)
        return torch.exp(-0.5 * (vx.square() + vy.square()) / theta) / (
            2.0 * torch.pi * theta
        )

    def density_atom_stds(self):
        if self.config.velocity_representation != VELOCITY_DENSITY_DICTIONARY:
            raise RuntimeError("density_atom_stds is available only for the density dictionary")
        return self.torch.exp(self.density_atom_log_stds).clamp_min(1.0e-3)

    def _density_atom_values(self, vx, vy):
        torch = self.torch
        velocity = torch.stack((vx, vy), dim=-1)
        means = self.density_atom_means
        stds = self.density_atom_stds()
        centered = (
            velocity[:, None, :] - means[None, :, :]
        ) / stds[None, :, :]
        normalization = 2.0 * torch.pi * stds[:, 0] * stds[:, 1]
        return torch.exp(-0.5 * torch.sum(centered.square(), dim=-1)) / (
            normalization[None, :]
        )

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
            "collision_operator": "nu * div_v(v*f + temperature*grad_v(f))",
            "poisson_equation": "-Delta_x(phi)=rho-background_density; E=-grad_x(phi)",
            "normalization": {
                "rho_spatial_mean": float(self.config.equation.background_density),
                "g_velocity_integral": 1.0,
                "g_normalization_backend": "analytic_full_space",
            },
            "rho_fourier_half_mode_count": len(self.rho_half_modes),
            "velocity_representation": self.config.velocity_representation,
            "velocity_basis_count": self.config.velocity_basis_count,
        }

    def public_equation_metadata(self) -> dict[str, Any]:
        """Return PDE data without revealing the hidden representation."""

        return {
            **self.config.equation.to_dict(),
            "domain": {
                "space": "two_dimensional_torus",
                "velocity": "R^2",
                "velocity_truncation_role": "numerical_query_window_only",
                "time": [float(value) for value in self.config.time_domain],
            },
            "collision_operator": "nu * div_v(v*f + temperature*grad_v(f))",
            "poisson_equation": "-Delta_x(phi)=rho-background_density; E=-grad_x(phi)",
            "normalization": {
                "rho_spatial_mean": float(self.config.equation.background_density),
                "g_velocity_integral": 1.0,
            },
        }

    def _coordinate_features(self, points):
        torch = self.torch
        equation = self.config.equation
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
        lx, ly = equation.x_domain
        for mode in range(1, self.config.spatial_feature_modes + 1):
            x_phase = 2.0 * torch.pi * float(mode) * x / float(lx)
            y_phase = 2.0 * torch.pi * float(mode) * y / float(ly)
            spatial.extend(
                (torch.sin(x_phase), torch.cos(x_phase), torch.sin(y_phase), torch.cos(y_phase))
            )
        return time_values, torch.cat((time_values, *spatial), dim=-1), tau

    def _rho_poisson(self, points, coefficients):
        torch = self.torch
        constant = coefficients[:, :1]
        pairs = coefficients[:, 1:].reshape(int(points.shape[0]), -1, 2)
        phase_rows = []
        for kx, ky in self.rho_half_modes:
            phase_rows.append(
                2.0
                * torch.pi
                * (
                    float(kx) * points[:, 1] / float(self.config.equation.x_domain[0])
                    + float(ky) * points[:, 2] / float(self.config.equation.x_domain[1])
                )
            )
        if phase_rows:
            phases = torch.stack(phase_rows, dim=-1)
            psi = constant[:, 0] + np.sqrt(2.0) * torch.sum(
                pairs[:, :, 0] * torch.cos(phases)
                + pairs[:, :, 1] * torch.sin(phases),
                dim=-1,
            )
        else:
            psi = constant[:, 0]
        norm_squared = torch.sum(coefficients.square(), dim=-1).clamp_min(
            torch.finfo(coefficients.dtype).eps
        )
        scale = float(self.config.equation.background_density - self.config.rho_floor)
        rho = float(self.config.rho_floor) + scale * psi.square() / norm_squared

        complex_coefficients = [torch.complex(constant[:, 0], torch.zeros_like(constant[:, 0]))]
        inv_sqrt_two = float(2.0**-0.5)
        for mode_index in range(len(self.rho_half_modes)):
            cosine = pairs[:, mode_index, 0]
            sine = pairs[:, mode_index, 1]
            positive = torch.complex(cosine, -sine) * inv_sqrt_two
            complex_coefficients.extend((positive, torch.conj(positive)))
        complex_coefficients_tensor = torch.stack(complex_coefficients, dim=-1)
        products = (
            complex_coefficients_tensor[:, :, None]
            * complex_coefficients_tensor[:, None, :]
        ).reshape(int(points.shape[0]), -1)
        convolution = torch.stack(
            [torch.sum(products[:, indices], dim=-1) for indices in self._pair_groups],
            dim=-1,
        )
        rho_hat = scale * convolution / norm_squared[:, None]
        zero_index = next(
            index
            for index, mode in enumerate(self._convolution_modes.tolist())
            if mode == [0, 0]
        )
        zero_addition = torch.zeros_like(rho_hat)
        zero_addition[:, zero_index] = float(self.config.rho_floor)
        rho_hat = rho_hat + zero_addition

        modes = self._convolution_modes.to(dtype=points.dtype)
        kappa_x = 2.0 * torch.pi * modes[:, 0] / float(self.config.equation.x_domain[0])
        kappa_y = 2.0 * torch.pi * modes[:, 1] / float(self.config.equation.x_domain[1])
        kappa_squared = kappa_x.square() + kappa_y.square()
        nonzero = kappa_squared > 0.0
        query_phase = (
            points[:, 1:2] * kappa_x[None, :]
            + points[:, 2:3] * kappa_y[None, :]
        )
        exponential = torch.exp(torch.complex(torch.zeros_like(query_phase), query_phase))
        phi_hat = torch.where(
            nonzero[None, :],
            rho_hat / torch.where(nonzero, kappa_squared, torch.ones_like(kappa_squared))[None, :],
            torch.zeros_like(rho_hat),
        )
        phi = torch.real(torch.sum(phi_hat * exponential, dim=-1))
        electric_x = torch.real(
            torch.sum((-1j * kappa_x[None, :]) * phi_hat * exponential, dim=-1)
        )
        electric_y = torch.real(
            torch.sum((-1j * kappa_y[None, :]) * phi_hat * exponential, dim=-1)
        )
        unit_coefficients = coefficients / torch.sqrt(norm_squared[:, None])
        return rho, phi, torch.stack((electric_x, electric_y), dim=-1), unit_coefficients

    def _velocity_basis(self, vx, vy):
        theta_sqrt = float(self.config.equation.temperature) ** 0.5
        hx = _normalized_probabilists_hermite(self.torch, vx / theta_sqrt, self.config.velocity_total_degree)
        hy = _normalized_probabilists_hermite(self.torch, vy / theta_sqrt, self.config.velocity_total_degree)
        return self.torch.stack(
            [hx[:, ax] * hy[:, ay] for ax, ay in self.velocity_multi_indices],
            dim=-1,
        )


def manufacture_standard_vpfp_source(
    model,
    points,
    latent,
    *,
    micro_latent=None,
    equation_condition=None,
    create_graph: bool = False,
) -> dict[str, Any]:
    """Evaluate hidden fields and manufacture S with pointwise autograd.

    The represented equation is

      d_t f + v dot grad_x f + E dot grad_v f
        = nu div_v(v f + theta grad_v f) + S.
    """

    torch = _torch()
    parameter = next(model.parameters())
    query = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
    if query.ndim == 1:
        query = query[None, :]
    query = query.detach().clone().requires_grad_(True)
    field_kwargs = {"micro_latent": micro_latent}
    if equation_condition is not None:
        field_kwargs["equation_condition"] = equation_condition
    values = model.fields(query, latent, **field_kwargs)
    f = values["f"]
    gradient = torch.autograd.grad(
        f.sum(),
        query,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    second_vx = torch.autograd.grad(
        gradient[:, 3].sum(),
        query,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0][:, 3]
    second_vy = torch.autograd.grad(
        gradient[:, 4].sum(),
        query,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )[0][:, 4]
    condition_values = values.get("equation_condition")
    if condition_values is None:
        theta = float(model.config.equation.temperature)
        collision_nu = float(model.config.equation.collision_nu)
    else:
        theta = condition_values[:, 1]
        collision_nu = condition_values[:, 0]
    collision = (
        2.0 * f
        + query[:, 3] * gradient[:, 3]
        + query[:, 4] * gradient[:, 4]
        + theta * (second_vx + second_vy)
    )
    electric = values["electric"]
    source = (
        gradient[:, 0]
        + query[:, 3] * gradient[:, 1]
        + query[:, 4] * gradient[:, 2]
        + electric[:, 0] * gradient[:, 3]
        + electric[:, 1] * gradient[:, 4]
        - collision_nu * collision
    )
    return {
        **values,
        "source": source,
        "f_gradient": gradient,
        "f_velocity_laplacian": second_vx + second_vy,
        "collision": collision,
        "points": query,
    }


def _broadcast_rows(torch, value, *, count: int, width: int, dtype, device, name: str):
    rows = torch.as_tensor(value, dtype=dtype, device=device)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or int(rows.shape[-1]) != int(width):
        raise ValueError(f"{name} must have shape [{width}] or [N,{width}], got {tuple(rows.shape)}")
    if int(rows.shape[0]) == 1:
        rows = rows.expand(count, -1)
    if int(rows.shape[0]) != count:
        raise ValueError(f"{name} row count {int(rows.shape[0])} does not match query count {count}")
    return rows


def _time_feature_dim(modes: int) -> int:
    return 1 + 2 * int(modes)


def _real_fourier_half_modes(max_x: int, max_y: int) -> list[tuple[int, int]]:
    return [
        (kx, ky)
        for kx in range(-int(max_x), int(max_x) + 1)
        for ky in range(-int(max_y), int(max_y) + 1)
        if kx > 0 or (kx == 0 and ky > 0)
    ]


def _hermite_multi_indices(total_degree: int) -> list[tuple[int, int]]:
    return [
        (ax, degree - ax)
        for degree in range(int(total_degree) + 1)
        for ax in range(degree + 1)
    ]


def _fourier_convolution_structure(half_modes: list[tuple[int, int]]):
    complex_modes = [(0, 0)]
    for mode in half_modes:
        complex_modes.extend((mode, (-mode[0], -mode[1])))
    sums = sorted(
        {
            (left[0] + right[0], left[1] + right[1])
            for left in complex_modes
            for right in complex_modes
        }
    )
    sum_to_index = {mode: index for index, mode in enumerate(sums)}
    groups: list[list[int]] = [[] for _ in sums]
    width = len(complex_modes)
    for left_index, left in enumerate(complex_modes):
        for right_index, right in enumerate(complex_modes):
            mode = (left[0] + right[0], left[1] + right[1])
            groups[sum_to_index[mode]].append(left_index * width + right_index)
    return complex_modes, sums, groups


def _normalized_probabilists_hermite(torch, values, degree: int):
    rows = [torch.ones_like(values)]
    if degree == 0:
        return torch.stack(rows, dim=-1)
    rows.append(values)
    for order in range(1, int(degree)):
        rows.append(
            values * rows[-1] / float(order + 1) ** 0.5
            - float(order / (order + 1)) ** 0.5 * rows[-2]
        )
    return torch.stack(rows, dim=-1)


def _density_dictionary_initial_values(config: VPFPContinuousFieldConfig):
    """Initialize learnable R^2 Gaussian densities without defining a box."""

    count = int(config.velocity_density_atom_count)
    means = np.zeros((count, 2), dtype=float)
    if count > 1:
        index = np.arange(1, count, dtype=float)
        radius = float(config.density_atom_center_scale) * np.sqrt(
            (index - 0.5) / float(count - 1)
        )
        angle = index * np.pi * (3.0 - np.sqrt(5.0))
        means[1:, 0] = radius * np.cos(angle)
        means[1:, 1] = radius * np.sin(angle)
    largest = float(config.density_atom_std_range[1])
    smallest = float(config.density_atom_std_range[0])
    scales = np.geomspace(largest, smallest, count)
    stds = np.repeat(scales[:, None], 2, axis=1)
    return means, stds


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for continuous VPFP fields") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
