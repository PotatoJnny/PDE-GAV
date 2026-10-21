from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from agent_verifier.vpfp.continuous_flow import (
    EQUATION_CONDITION_KEYS,
    VPFPContinuousFlowConfig,
    VPFPContinuousFlowField,
)
from agent_verifier.vpfp.continuous_hermite import (
    VPFPContinuousHermiteConfig,
    VPFPContinuousHermiteField,
)
from agent_verifier.vpfp.hermite_density import (
    squared_hermite_projection_diagnostic,
    torch_squared_hermite_density,
)
from agent_verifier.vpfp.io import load_case


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


FLOW_MOMENT_POWERS = tuple(
    (power_x, degree - power_x)
    for degree in range(1, 5)
    for power_x in range(degree + 1)
)
FLOW_CHARACTERISTIC_WAVES = (
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
    (0.5, -0.5),
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.5),
    (0.5, 1.0),
)
FLOW_TAIL_RADII = (2.0, 3.0, 4.0)
FLOW_OBSERVATION_FEATURE_KEYS = (
    "log_rho",
    "rho_anomaly",
    *(f"moment_{power_x}_{power_y}" for power_x, power_y in FLOW_MOMENT_POWERS),
    *(
        name
        for wave_x, wave_y in FLOW_CHARACTERISTIC_WAVES
        for name in (
            f"characteristic_cos_{wave_x:g}_{wave_y:g}",
            f"characteristic_sin_{wave_x:g}_{wave_y:g}",
        )
    ),
    "velocity_entropy",
    *(f"tail_mass_radius_{radius:g}" for radius in FLOW_TAIL_RADII),
)


@dataclass(frozen=True)
class VPFPFlowObservationEncoderConfig:
    input_channels: int = len(FLOW_OBSERVATION_FEATURE_KEYS)
    latent_dim: int = 256
    width: int = 192
    point_layers: int = 3
    attention_heads: int = 8
    query_tokens: int = 16
    self_attention_layers: int = 2
    output_head_width: int = 0
    time_feature_modes: int = 4
    spatial_feature_modes: int = 4
    time_domain: tuple[float, float] = (0.0, 0.35)
    x_domain: tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi)
    collision_nu_range: tuple[float, float] = (0.02, 0.08)
    temperature_range: tuple[float, float] = (0.8, 1.2)
    dtype: str = "float32"
    representation: str = "vpfp_solution_observation_point_set_encoder_v1"

    def validate(self) -> None:
        if min(
            self.input_channels,
            self.latent_dim,
            self.width,
            self.point_layers,
            self.attention_heads,
            self.query_tokens,
            self.self_attention_layers,
            self.time_feature_modes,
            self.spatial_feature_modes,
        ) < 1:
            raise ValueError("observation encoder dimensions must be positive")
        if self.output_head_width < 0:
            raise ValueError("output_head_width must be nonnegative")
        if self.width % self.attention_heads:
            raise ValueError("encoder width must be divisible by attention_heads")
        if self.time_domain[1] <= self.time_domain[0]:
            raise ValueError("time_domain must be increasing")
        if min(self.x_domain) <= 0.0:
            raise ValueError("x_domain lengths must be positive")
        if self.collision_nu_range[1] <= self.collision_nu_range[0]:
            raise ValueError("collision_nu_range must be increasing")
        if self.temperature_range[1] <= self.temperature_range[0]:
            raise ValueError("temperature_range must be increasing")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "time_domain",
            "x_domain",
            "collision_nu_range",
            "temperature_range",
        ):
            payload[key] = [float(value) for value in payload[key]]
        return payload

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "VPFPFlowObservationEncoderConfig":
        payload = dict(data)
        for key in (
            "time_domain",
            "x_domain",
            "collision_nu_range",
            "temperature_range",
        ):
            if key in payload:
                payload[key] = tuple(float(value) for value in payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class VPFPFlowReconstructionLossConfig:
    rho_log_weight: float = 1.0
    rho_relative_weight: float = 0.5
    velocity_nll_weight: float = 0.25
    velocity_hellinger_weight: float = 1.0
    f_relative_weight: float = 0.5
    slice_density_weight: float = 1.0
    moment_weight: float = 0.75
    window_normalization_weight: float = 0.25
    rho_slice_weight: float = 1.0
    rho_spectral_weight: float = 0.25
    rho_spectral_frequency_scale: float = 0.10
    characteristic_weight: float = 0.50
    hermite_coefficient_weight: float = 1.00
    hermite_frequency_scale: float = 0.05
    latent_l2_weight: float = 1.0e-4
    latent_swap_weight: float = 0.10
    latent_swap_margin: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VPFPFlowObservationEncoder:
    @staticmethod
    def build(config: VPFPFlowObservationEncoderConfig | dict[str, Any]):
        cfg = (
            config
            if isinstance(config, VPFPFlowObservationEncoderConfig)
            else VPFPFlowObservationEncoderConfig.from_dict(dict(config))
        )
        cfg.validate()
        return _VPFPFlowObservationEncoder(_torch(), cfg)


class _VPFPFlowObservationEncoder(
    (_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)
):
    def __init__(self, torch_module, config: VPFPFlowObservationEncoderConfig):
        super().__init__()
        torch = torch_module
        self.torch = torch
        self.config = config
        coordinate_dim = (
            1
            + 2 * int(config.time_feature_modes)
            + 4 * int(config.spatial_feature_modes)
        )
        point_input_dim = 2 * int(config.input_channels) + coordinate_dim + 2
        point_layers: list[Any] = []
        in_features = point_input_dim
        for _ in range(int(config.point_layers)):
            point_layers.extend(
                (
                    torch.nn.Linear(in_features, config.width),
                    torch.nn.LayerNorm(config.width),
                    torch.nn.SiLU(),
                )
            )
            in_features = config.width
        self.point_network = torch.nn.Sequential(*point_layers)
        self.query_tokens = torch.nn.Parameter(
            torch.randn(config.query_tokens, config.width)
            / np.sqrt(float(config.width))
        )
        self.cross_attention = torch.nn.MultiheadAttention(
            config.width,
            config.attention_heads,
            batch_first=True,
        )
        self.cross_norm = torch.nn.LayerNorm(config.width)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.token_network = torch.nn.TransformerEncoder(
            layer, num_layers=config.self_attention_layers
        )
        head_width = (
            2 * int(config.latent_dim)
            if int(config.output_head_width) == 0
            else int(config.output_head_width)
        )
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                config.query_tokens * config.width,
                head_width,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(head_width, config.latent_dim),
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def forward(
        self,
        coordinates,
        features,
        equation_condition,
        *,
        point_mask=None,
        channel_mask=None,
    ):
        torch = self.torch
        parameter = next(self.parameters())
        coordinates = torch.as_tensor(
            coordinates, dtype=parameter.dtype, device=parameter.device
        )
        features = torch.as_tensor(
            features, dtype=parameter.dtype, device=parameter.device
        )
        if coordinates.ndim == 2:
            coordinates = coordinates[None]
        if features.ndim == 2:
            features = features[None]
        if (
            coordinates.ndim != 3
            or coordinates.shape[-1] != 3
            or features.ndim != 3
            or coordinates.shape[:2] != features.shape[:2]
            or features.shape[-1] != self.config.input_channels
        ):
            raise ValueError(
                "encoder expects coordinates [B,N,3] and features [B,N,C]"
            )
        batch, point_count, _ = features.shape
        if point_mask is None:
            point_mask = torch.ones(
                (batch, point_count), dtype=torch.bool, device=features.device
            )
        else:
            point_mask = torch.as_tensor(
                point_mask, dtype=torch.bool, device=features.device
            )
        if point_mask.shape != (batch, point_count):
            raise ValueError("point_mask must have shape [B,N]")
        if bool(torch.any(torch.sum(point_mask, dim=1) == 0)):
            raise ValueError("each observation set must contain at least one point")
        if channel_mask is None:
            channel_mask = torch.ones_like(features)
        else:
            channel_mask = torch.as_tensor(
                channel_mask, dtype=features.dtype, device=features.device
            )
        if channel_mask.shape != features.shape:
            raise ValueError("channel_mask must have shape [B,N,C]")
        condition = self._condition_features(
            equation_condition, batch=batch
        )[:, None, :].expand(-1, point_count, -1)
        position = self._coordinate_features(coordinates)
        point_input = torch.cat(
            (features * channel_mask, channel_mask, position, condition),
            dim=-1,
        )
        point_embeddings = self.point_network(point_input)
        point_embeddings = torch.where(
            point_mask[:, :, None], point_embeddings, torch.zeros_like(point_embeddings)
        )
        queries = self.query_tokens[None].expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            queries,
            point_embeddings,
            point_embeddings,
            key_padding_mask=~point_mask,
            need_weights=False,
        )
        tokens = self.cross_norm(queries + attended)
        return self.head(self.token_network(tokens))

    def _condition_features(self, condition, *, batch: int):
        torch = self.torch
        parameter = next(self.parameters())
        rows = torch.as_tensor(
            condition, dtype=parameter.dtype, device=parameter.device
        )
        if rows.ndim == 1:
            rows = rows[None]
        if rows.shape == (1, len(EQUATION_CONDITION_KEYS)):
            rows = rows.expand(batch, -1)
        if rows.shape != (batch, len(EQUATION_CONDITION_KEYS)):
            raise ValueError("equation_condition must have shape [B,2] or [2]")
        lower = torch.as_tensor(
            (
                self.config.collision_nu_range[0],
                self.config.temperature_range[0],
            ),
            dtype=rows.dtype,
            device=rows.device,
        )
        upper = torch.as_tensor(
            (
                self.config.collision_nu_range[1],
                self.config.temperature_range[1],
            ),
            dtype=rows.dtype,
            device=rows.device,
        )
        return 2.0 * (rows - lower) / (upper - lower) - 1.0

    def _coordinate_features(self, coordinates):
        torch = self.torch
        t0, t1 = self.config.time_domain
        tau = (coordinates[..., 0:1] - float(t0)) / float(t1 - t0)
        output = [tau]
        for mode in range(1, self.config.time_feature_modes + 1):
            phase = float(mode) * torch.pi * tau
            output.extend((torch.sin(phase), torch.cos(phase)))
        for axis, length in zip((1, 2), self.config.x_domain, strict=True):
            value = coordinates[..., axis : axis + 1]
            for mode in range(1, self.config.spatial_feature_modes + 1):
                phase = 2.0 * torch.pi * float(mode) * value / float(length)
                output.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(output, dim=-1)


def load_flow_reference_case(case_dir: str | Path) -> dict[str, Any]:
    case = load_case(case_dir)
    public = case["public_arrays"]
    hidden = case["hidden_arrays"]
    rho = np.asarray(hidden["rho_true"], dtype=np.float32)
    g = np.asarray(hidden["g_true"], dtype=np.float32)
    weights = np.asarray(public["velocity_weights"], dtype=np.float32)
    vx = np.asarray(public["vx"], dtype=np.float32)
    vy = np.asarray(public["vy"], dtype=np.float32)
    if g.shape != rho.shape + (vx.size, vy.size):
        raise ValueError(f"incompatible rho/g shapes in {case_dir}")
    if weights.shape != (vx.size, vy.size):
        raise ValueError(f"incompatible velocity weights in {case_dir}")
    temperature = float(case["public_question"].get("temperature", 1.0))
    condition = np.asarray(
        (float(case["collision_nu"]), temperature), dtype=np.float32
    )
    output = {
        "case_id": case["case_id"],
        "case_dir": str(Path(case_dir).resolve()),
        "t": np.asarray(public["t"], dtype=np.float32),
        "x": np.asarray(public["x"], dtype=np.float32),
        "y": np.asarray(public["y"], dtype=np.float32),
        "vx": vx,
        "vy": vy,
        "velocity_weights": weights,
        "rho": rho,
        "g": g,
        "equation_condition": condition,
        "background_density": float(
            case["public_question"].get("background_density", 1.0)
        ),
    }
    observation = numerical_flow_observation_features(output)
    output["observation_coordinates"] = observation["coordinates"]
    output["observation_features"] = observation["features"]
    return output


def numerical_flow_observation_features(case: dict[str, Any]) -> dict[str, Any]:
    rho = np.asarray(case["rho"], dtype=np.float64)
    g = np.asarray(case["g"], dtype=np.float64)
    weights = np.asarray(case["velocity_weights"], dtype=np.float64)
    vx = np.asarray(case["vx"], dtype=np.float64)
    vy = np.asarray(case["vy"], dtype=np.float64)
    mass = np.sum(g * weights, axis=(-2, -1))
    normalized_g = g / np.maximum(mass[..., None, None], 1.0e-12)
    vx_grid = vx[:, None]
    vy_grid = vy[None, :]
    fields: list[np.ndarray] = [
        np.log(np.maximum(rho, 1.0e-12)),
        rho / float(case.get("background_density", 1.0)) - 1.0,
    ]
    for power_x, power_y in FLOW_MOMENT_POWERS:
        monomial = vx_grid**power_x * vy_grid**power_y
        fields.append(
            np.sum(normalized_g * weights * monomial, axis=(-2, -1))
        )
    for wave_x, wave_y in FLOW_CHARACTERISTIC_WAVES:
        phase = float(wave_x) * vx_grid + float(wave_y) * vy_grid
        fields.extend(
            (
                np.sum(normalized_g * weights * np.cos(phase), axis=(-2, -1)),
                np.sum(normalized_g * weights * np.sin(phase), axis=(-2, -1)),
            )
        )
    fields.append(
        -np.sum(
            normalized_g
            * weights
            * np.log(np.maximum(normalized_g, 1.0e-12)),
            axis=(-2, -1),
        )
    )
    radius = np.sqrt(vx_grid**2 + vy_grid**2)
    for threshold in FLOW_TAIL_RADII:
        fields.append(
            np.sum(
                normalized_g * weights * (radius >= float(threshold)),
                axis=(-2, -1),
            )
        )
    values = np.stack(fields, axis=-1)
    t, x, y = np.meshgrid(
        np.asarray(case["t"], dtype=np.float64),
        np.asarray(case["x"], dtype=np.float64),
        np.asarray(case["y"], dtype=np.float64),
        indexing="ij",
    )
    coordinates = np.stack((t, x, y), axis=-1)
    if values.shape[:-1] != coordinates.shape[:-1]:
        raise ValueError("observation feature and coordinate grids disagree")
    if values.shape[-1] != len(FLOW_OBSERVATION_FEATURE_KEYS):
        raise RuntimeError("unexpected observation feature count")
    if not np.all(np.isfinite(values)):
        raise ValueError("observation features contain non-finite values")
    return {
        "coordinates": coordinates.reshape(-1, 3).astype(np.float32),
        "features": values.reshape(-1, values.shape[-1]).astype(np.float32),
        "feature_keys": list(FLOW_OBSERVATION_FEATURE_KEYS),
    }


def fit_observation_normalization(
    feature_arrays: Iterable[np.ndarray], *, min_std: float = 1.0e-5
) -> dict[str, Any]:
    total_count = 0
    total = np.zeros(len(FLOW_OBSERVATION_FEATURE_KEYS), dtype=np.float64)
    total_square = np.zeros_like(total)
    for raw in feature_arrays:
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != total.size:
            raise ValueError("observation features must have shape [N,C]")
        total_count += values.shape[0]
        total += np.sum(values, axis=0)
        total_square += np.sum(values**2, axis=0)
    if total_count < 1:
        raise ValueError("cannot normalize an empty observation corpus")
    mean = total / float(total_count)
    variance = np.maximum(total_square / float(total_count) - mean**2, 0.0)
    std = np.maximum(np.sqrt(variance), float(min_std))
    return {
        "schema": "vpfp_flow_observation_normalization_v1",
        "keys": list(FLOW_OBSERVATION_FEATURE_KEYS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min_std": float(min_std),
        "sample_count": int(total_count),
    }


def normalize_observation_features(
    features, normalization: dict[str, Any]
):
    torch = _torch()
    values = torch.as_tensor(features)
    mean = torch.as_tensor(
        normalization["mean"], dtype=values.dtype, device=values.device
    )
    std = torch.as_tensor(
        normalization["std"], dtype=values.dtype, device=values.device
    )
    return (values - mean) / std


def prepare_hermite_coefficient_targets(
    cases: Sequence[dict[str, Any]],
    *,
    max_total_degree: int,
    max_design_condition_number: float = 1.0e3,
) -> dict[str, Any]:
    case_reports = []
    basis_count = None
    for case in cases:
        density = np.asarray(case["g"], dtype=np.float64)
        diagnostic = squared_hermite_projection_diagnostic(
            density,
            case["vx"],
            case["vy"],
            case["velocity_weights"],
            max_total_degree=int(max_total_degree),
            temperature=float(case["equation_condition"][1]),
        )
        if not bool(diagnostic["design_full_column_rank"]):
            raise ValueError(
                f"Hermite projection is rank deficient for {case['case_id']}"
            )
        condition_number = float(diagnostic["design_condition_number"])
        if (
            not np.isfinite(condition_number)
            or condition_number > float(max_design_condition_number)
        ):
            raise ValueError(
                "Hermite projection is ill-conditioned for "
                f"{case['case_id']}: {condition_number}"
            )
        coefficients = np.asarray(
            diagnostic["normalized_coefficients"],
            dtype=np.float32,
        ).reshape(density.shape[:-2] + (-1,))
        case["hermite_coefficient_targets"] = coefficients
        basis_count = int(diagnostic["basis_count"])
        case_reports.append(
            {
                "case_id": str(case["case_id"]),
                "design_condition_number": condition_number,
                "density_rel_l2": _numpy_metric_summary(
                    diagnostic["density_rel_l2"]
                ),
                "sqrt_density_rel_l2": _numpy_metric_summary(
                    diagnostic["sqrt_density_rel_l2"]
                ),
                "window_mass_abs_error": _numpy_metric_summary(
                    diagnostic["window_mass_abs_error"]
                ),
                "highest_degree_energy_fraction": _numpy_metric_summary(
                    diagnostic["highest_degree_energy_fraction"]
                ),
            }
        )
    return {
        "schema": "vpfp_hermite_coefficient_target_projection_v1",
        "max_total_degree": int(max_total_degree),
        "basis_count": int(basis_count or 0),
        "max_design_condition_number": float(
            max(row["design_condition_number"] for row in case_reports)
        ),
        "case_count": len(case_reports),
        "density_case_mean_summary": _numpy_metric_summary(
            [row["density_rel_l2"]["mean"] for row in case_reports]
        ),
        "density_case_p90_summary": _numpy_metric_summary(
            [row["density_rel_l2"]["p90"] for row in case_reports]
        ),
        "cases": case_reports,
    }


def prepare_structured_direct_projection_operators(
    model,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Factor the fixed-grid operators shared by direct projection cases."""

    config = model.config
    if not bool(config.structured_spectral_latent):
        raise ValueError("structured spectral latent is required")
    if not bool(config.structured_direct_coefficients):
        raise ValueError("direct structured coefficients are required")
    if float(config.structured_residual_scale) != 0.0:
        raise ValueError("analytic direct projection requires zero residual scale")

    time = np.asarray(case["t"], dtype=np.float64)
    x = np.asarray(case["x"], dtype=np.float64)
    y = np.asarray(case["y"], dtype=np.float64)
    t0, t1 = (float(value) for value in config.time_domain)
    tau = (time - t0) / (t1 - t0)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    cubic_time = (
        config.structured_time_basis == "piecewise_cubic_hermite_v1"
    )
    if cubic_time and int(config.structured_time_control_count) != time.size:
        raise ValueError(
            "piecewise cubic Hermite projection requires one control per "
            "trajectory output time"
        )

    local_spatial = _numpy_real_fourier_basis(
        x_grid,
        y_grid,
        model.local_half_modes,
        x_domain=config.equation.x_domain,
        include_constant=True,
        normalized=True,
    )
    local_spatial_projection = (
        float(config.structured_local_scale)
        * local_spatial
        / np.sqrt(float(local_spatial.shape[-1]))
    ).reshape(x.size * y.size, -1)
    if cubic_time:
        local_design = local_spatial_projection
    else:
        local_time = _numpy_chebyshev_basis(
            tau,
            int(config.structured_local_time_order),
        )
        local_design = np.einsum(
            "ta,xyb->txyab",
            local_time,
            local_spatial,
        ).reshape(time.size * x.size * y.size, -1)
        local_design *= float(config.structured_local_scale) / np.sqrt(
            float(local_design.shape[1])
        )

    rho_spatial = _numpy_real_fourier_basis(
        x_grid,
        y_grid,
        model.rho_half_modes,
        x_domain=config.equation.x_domain,
        include_constant=False,
        normalized=False,
    ).reshape(x.size * y.size, -1)
    if cubic_time:
        rho_time = np.eye(time.size, dtype=np.float64)
    else:
        rho_time = _numpy_chebyshev_basis(
            tau,
            int(config.structured_rho_time_order),
        ) / np.sqrt(float(int(config.structured_rho_time_order) + 1))
    return {
        "schema": "vpfp_structured_direct_projection_operators_v1",
        "structured_time_basis": str(config.structured_time_basis),
        "t": time.copy(),
        "tau": tau.copy(),
        "x": x.copy(),
        "y": y.copy(),
        "local_design": local_design,
        "local_pseudoinverse": np.linalg.pinv(local_design),
        "local_spatial_projection": local_spatial_projection,
        "local_spatial_pseudoinverse": np.linalg.pinv(
            local_spatial_projection
        ),
        "local_spatial_feature_count": int(local_spatial.shape[-1]),
        "rho_spatial": rho_spatial,
        "rho_spatial_pseudoinverse": np.linalg.pinv(rho_spatial),
        "rho_time": rho_time,
        "rho_time_pseudoinverse": np.linalg.pinv(rho_time),
        "local_design_condition_number": float(np.linalg.cond(local_design)),
        "rho_spatial_condition_number": float(np.linalg.cond(rho_spatial)),
        "rho_time_condition_number": float(np.linalg.cond(rho_time)),
    }


def fit_structured_direct_case_latent(
    model,
    case: dict[str, Any],
    *,
    projection_operators: dict[str, Any] | None = None,
    rho_feasible_fraction: float = 0.95,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project one gridded reference onto the canonical structured latent."""

    config = model.config
    if not bool(config.structured_spectral_latent):
        raise ValueError("structured spectral latent is required")
    if not bool(config.structured_direct_coefficients):
        raise ValueError("direct structured coefficients are required")
    if float(config.structured_residual_scale) != 0.0:
        raise ValueError("analytic direct projection requires zero residual scale")
    if "hermite_coefficient_targets" not in case:
        raise ValueError("Hermite coefficient targets must be prepared first")
    if not 0.0 < float(rho_feasible_fraction) < 1.0:
        raise ValueError("rho_feasible_fraction must lie strictly in (0,1)")

    time = np.asarray(case["t"], dtype=np.float64)
    x = np.asarray(case["x"], dtype=np.float64)
    y = np.asarray(case["y"], dtype=np.float64)
    rho = np.asarray(case["rho"], dtype=np.float64)
    target = np.asarray(
        case["hermite_coefficient_targets"],
        dtype=np.float64,
    )
    expected_shape = (
        time.size,
        x.size,
        y.size,
        len(model.hermite_modes),
    )
    if target.shape != expected_shape:
        raise ValueError(
            "Hermite coefficient target shape does not match reference grid"
        )
    if rho.shape != expected_shape[:3]:
        raise ValueError("rho shape does not match reference grid")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(rho)):
        raise ValueError("direct projection targets must be finite")

    operators = (
        prepare_structured_direct_projection_operators(model, case)
        if projection_operators is None
        else projection_operators
    )
    if operators.get("schema") != (
        "vpfp_structured_direct_projection_operators_v1"
    ):
        raise ValueError("unexpected direct projection operator schema")
    for key, values in (("t", time), ("x", x), ("y", y)):
        if not np.array_equal(operators[key], values):
            raise ValueError(
                "direct projection operators do not match the case grid"
            )
    cubic_time = (
        config.structured_time_basis == "piecewise_cubic_hermite_v1"
    )
    if cubic_time:
        target_by_time = target.reshape(time.size, x.size * y.size, -1)
        local_solution_by_time = np.einsum(
            "sn,tnc->tsc",
            operators["local_spatial_pseudoinverse"],
            target_by_time,
        )
        local_prediction = np.einsum(
            "ns,tsc->tnc",
            operators["local_spatial_projection"],
            local_solution_by_time,
        ).reshape(-1, target.shape[-1])
        local_value_controls = np.transpose(
            local_solution_by_time, (2, 0, 1)
        )
        local_slope_controls = _normalized_time_slopes(
            local_value_controls,
            operators["tau"],
            axis=1,
        )
        local_values = np.concatenate(
            (local_value_controls, local_slope_controls), axis=1
        )
    else:
        local_design = operators["local_design"]
        local_solution = (
            operators["local_pseudoinverse"]
            @ target.reshape(-1, target.shape[-1])
        )
        local_prediction = local_design @ local_solution
        local_values = local_solution.T.reshape(
            int(config.structured_local_channels),
            int(config.structured_local_time_order) + 1,
            int(operators["local_spatial_feature_count"]),
        )
    local_norm = np.linalg.norm(local_prediction, axis=-1, keepdims=True)
    normalized_local = np.divide(
        local_prediction,
        local_norm,
        out=np.zeros_like(local_prediction),
        where=local_norm > np.finfo(np.float64).eps,
    )
    zero_rows = local_norm[:, 0] <= np.finfo(np.float64).eps
    normalized_local[zero_rows, 0] = 1.0
    coefficient_rel_l2 = float(
        np.linalg.norm(normalized_local - target.reshape(-1, target.shape[-1]))
        / max(np.linalg.norm(target), np.finfo(np.float64).tiny)
    )

    rho_spatial = operators["rho_spatial"]
    background = float(config.equation.background_density)
    rho_anomaly = (rho - background).reshape(time.size, -1)
    desired_rho_coefficients = (
        operators["rho_spatial_pseudoinverse"] @ rho_anomaly.T
    ).T
    unconstrained_coefficient_gauge_norm = _rho_positivity_gauge_norm(
        desired_rho_coefficients,
        gauge=str(config.rho_positivity_gauge),
    )
    coefficient_scale = float(config.rho_contrast) * background
    feasibility_scale = np.minimum(
        1.0,
        (
            float(rho_feasible_fraction)
            * coefficient_scale
            / np.maximum(
                unconstrained_coefficient_gauge_norm,
                np.finfo(np.float64).tiny,
            )
        ),
    )
    desired_rho_coefficients *= feasibility_scale[:, None]
    coefficient_gauge_norm = _rho_positivity_gauge_norm(
        desired_rho_coefficients,
        gauge=str(config.rho_positivity_gauge),
    )
    raw_rho_coefficients = desired_rho_coefficients / (
        coefficient_scale - coefficient_gauge_norm[:, None]
    )
    rho_time = operators["rho_time"]
    if cubic_time:
        rho_values = np.concatenate(
            (
                raw_rho_coefficients,
                _normalized_time_slopes(
                    raw_rho_coefficients,
                    operators["tau"],
                    axis=0,
                ),
            ),
            axis=0,
        )
        fitted_raw_rho = raw_rho_coefficients
    else:
        rho_values = (
            operators["rho_time_pseudoinverse"] @ raw_rho_coefficients
        )
        fitted_raw_rho = rho_time @ rho_values
    fitted_rho_coefficients = (
        coefficient_scale
        * fitted_raw_rho
        / (
            1.0
            + _rho_positivity_gauge_norm(
                fitted_raw_rho,
                gauge=str(config.rho_positivity_gauge),
            )[:, None]
        )
    )
    fitted_rho = background + (
        rho_spatial @ fitted_rho_coefficients.T
    ).T.reshape(rho.shape)
    rho_rel_l2 = float(
        np.linalg.norm(fitted_rho - rho)
        / max(np.linalg.norm(rho), np.finfo(np.float64).tiny)
    )

    layout = config.structured_latent_layout()
    latent = np.concatenate(
        (
            rho_values.reshape(-1),
            np.zeros(
                int(layout["kinetic_global_latent_dim"]),
                dtype=np.float64,
            ),
            local_values.reshape(-1),
        )
    )
    if latent.size != int(layout["total_latent_dim"]):
        raise RuntimeError("analytic structured latent size mismatch")
    latent /= float(config.latent_input_scale)
    report = {
        "schema": "vpfp_structured_direct_case_projection_v1",
        "case_id": str(case.get("case_id", "")),
        "rho_rel_l2": rho_rel_l2,
        "hermite_coefficient_rel_l2": coefficient_rel_l2,
        "rho_spatial_condition_number": float(
            operators["rho_spatial_condition_number"]
        ),
        "rho_time_condition_number": float(
            operators["rho_time_condition_number"]
        ),
        "local_design_condition_number": float(
            operators["local_design_condition_number"]
        ),
        "rho_positivity_gauge": str(config.rho_positivity_gauge),
        "rho_coefficient_gauge_norm_max": float(
            np.max(coefficient_gauge_norm)
        ),
        "rho_unconstrained_coefficient_gauge_norm_max": float(
            np.max(unconstrained_coefficient_gauge_norm)
        ),
        "rho_feasible_fraction": float(rho_feasible_fraction),
        "rho_feasibility_scale_min": float(np.min(feasibility_scale)),
        "rho_clipped_time_count": int(np.sum(feasibility_scale < 1.0)),
    }
    return latent.astype(np.float32), report


def _normalized_time_slopes(
    values: np.ndarray,
    tau: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    """Finite-difference derivatives with respect to normalized time."""

    controls = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    coordinates = np.asarray(tau, dtype=np.float64)
    if controls.shape[0] != coordinates.size or coordinates.size < 2:
        raise ValueError("time controls and normalized coordinates disagree")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("normalized time coordinates must be increasing")
    slopes = np.empty_like(controls)
    slopes[0] = (controls[1] - controls[0]) / (
        coordinates[1] - coordinates[0]
    )
    slopes[-1] = (controls[-1] - controls[-2]) / (
        coordinates[-1] - coordinates[-2]
    )
    if coordinates.size > 2:
        denominator = coordinates[2:] - coordinates[:-2]
        reshape = (denominator.size,) + (1,) * (controls.ndim - 1)
        slopes[1:-1] = (controls[2:] - controls[:-2]) / denominator.reshape(
            reshape
        )
    return np.moveaxis(slopes, 0, axis)


def _rho_positivity_gauge_norm(
    coefficients: np.ndarray,
    *,
    gauge: str,
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape[-1] % 2:
        raise ValueError("rho Fourier coefficients must be cosine/sine pairs")
    if gauge == "coordinate_l1_v1":
        return np.sum(np.abs(values), axis=-1)
    if gauge == "phase_invariant_group_l1_v2":
        pairs = values.reshape(*values.shape[:-1], -1, 2)
        return np.sum(np.linalg.norm(pairs, axis=-1), axis=-1)
    raise ValueError(f"unsupported rho positivity gauge: {gauge!r}")


def _numpy_chebyshev_basis(values: np.ndarray, order: int) -> np.ndarray:
    scaled = 2.0 * np.asarray(values, dtype=np.float64) - 1.0
    basis = [np.ones_like(scaled)]
    if int(order) >= 1:
        basis.append(scaled)
    for _ in range(1, int(order)):
        basis.append(2.0 * scaled * basis[-1] - basis[-2])
    return np.stack(basis, axis=-1)


def _numpy_real_fourier_basis(
    x: np.ndarray,
    y: np.ndarray,
    modes,
    *,
    x_domain: tuple[float, float],
    include_constant: bool,
    normalized: bool,
) -> np.ndarray:
    values = []
    if include_constant:
        values.append(np.ones_like(x, dtype=np.float64))
    amplitude = np.sqrt(2.0) if normalized else 1.0
    lx, ly = (float(value) for value in x_domain)
    for kx, ky in modes:
        phase = 2.0 * np.pi * (
            float(kx) * x / lx + float(ky) * y / ly
        )
        values.extend(
            (
                amplitude * np.cos(phase),
                amplitude * np.sin(phase),
            )
        )
    if not values:
        return np.empty(x.shape + (0,), dtype=np.float64)
    return np.stack(values, axis=-1)


def sample_hermite_coefficient_batch(
    cases: Sequence[dict[str, Any]],
    case_indices: Sequence[int],
    *,
    slices_per_case: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    points = []
    targets = []
    repeated_indices = []
    conditions = []
    for raw_index in case_indices:
        case_index = int(raw_index)
        case = cases[case_index]
        if "hermite_coefficient_targets" not in case:
            raise ValueError("Hermite coefficient targets were not prepared")
        target = np.asarray(
            case["hermite_coefficient_targets"],
            dtype=np.float32,
        )
        flat_count = int(np.prod(target.shape[:-1]))
        selected = rng.integers(
            0,
            flat_count,
            size=int(slices_per_case),
        )
        t_index, x_index, y_index = np.unravel_index(
            selected,
            target.shape[:-1],
        )
        points.append(
            np.stack(
                (
                    case["t"][t_index],
                    case["x"][x_index],
                    case["y"][y_index],
                    np.zeros(selected.size, dtype=np.float32),
                    np.zeros(selected.size, dtype=np.float32),
                ),
                axis=-1,
            )
        )
        targets.append(target[t_index, x_index, y_index])
        repeated_indices.append(
            np.full(selected.size, case_index, dtype=np.int64)
        )
        conditions.append(
            np.broadcast_to(
                case["equation_condition"],
                (selected.size, 2),
            )
        )
    return {
        "points": np.concatenate(points, axis=0),
        "target_coefficients": np.concatenate(targets, axis=0),
        "case_indices": np.concatenate(repeated_indices, axis=0),
        "equation_condition": np.concatenate(conditions, axis=0),
    }


def hermite_coefficient_loss(
    model,
    latent_table,
    batch: dict[str, np.ndarray],
    *,
    config: VPFPFlowReconstructionLossConfig,
):
    torch = _torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    indices = torch.as_tensor(
        batch["case_indices"],
        dtype=torch.long,
        device=device,
    )
    prediction = model.fields(
        torch.as_tensor(batch["points"], dtype=dtype, device=device),
        latent_table[indices],
        equation_condition=torch.as_tensor(
            batch["equation_condition"],
            dtype=dtype,
            device=device,
        ),
    )
    if "hermite_coefficients" not in prediction:
        raise ValueError("model does not expose Hermite coefficients")
    predicted = prediction["hermite_coefficients"]
    target = torch.as_tensor(
        batch["target_coefficients"],
        dtype=dtype,
        device=device,
    )
    if predicted.shape != target.shape:
        raise ValueError("predicted and target Hermite shapes differ")
    mode_degree = torch.as_tensor(
        [first + second for first, second in model.hermite_modes],
        dtype=dtype,
        device=device,
    )
    frequency_weight = (
        1.0
        + float(config.hermite_frequency_scale)
        * mode_degree.square()
    )
    squared_difference = (predicted - target).square()
    weighted_distance_squared = torch.sum(
        squared_difference * frequency_weight[None, :],
        dim=-1,
    ) / torch.mean(frequency_weight)
    coefficient_loss = 0.5 * torch.mean(weighted_distance_squared)
    relative_l2 = torch.mean(
        torch.linalg.vector_norm(predicted - target, dim=-1)
    )
    cosine_error = torch.mean(
        1.0 - torch.sum(predicted * target, dim=-1)
    )
    total = float(config.hermite_coefficient_weight) * coefficient_loss
    return total, {
        "hermite_coefficient_loss": coefficient_loss,
        "hermite_coefficient_rel_l2": relative_l2,
        "hermite_coefficient_cosine_error": cosine_error,
    }


def _numpy_metric_summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("metric values must be finite and nonempty")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def sample_flow_reference_batch(
    cases: Sequence[dict[str, Any]],
    case_indices: Sequence[int],
    *,
    points_per_case: int,
    rng: np.random.Generator,
    density_sample_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
    if not 0.0 <= float(density_sample_fraction) <= 1.0:
        raise ValueError("density_sample_fraction must lie in [0,1]")
    rows = {
        "points": [],
        "target_rho": [],
        "target_g": [],
        "case_indices": [],
        "equation_condition": [],
        "density_sample_mask": [],
    }
    for raw_index in case_indices:
        case_index = int(raw_index)
        case = cases[case_index]
        count = int(points_per_case)
        t_index = rng.integers(0, case["rho"].shape[0], size=count)
        x_index = rng.integers(0, case["rho"].shape[1], size=count)
        y_index = rng.integers(0, case["rho"].shape[2], size=count)
        velocity_count = case["vx"].size * case["vy"].size
        density_count = int(round(count * float(density_sample_fraction)))
        velocity_flat = rng.integers(0, velocity_count, size=count)
        for row_index in range(density_count):
            density = np.maximum(
                case["g"][
                    t_index[row_index],
                    x_index[row_index],
                    y_index[row_index],
                ]
                * case["velocity_weights"],
                0.0,
            ).reshape(-1)
            probability = density / max(float(np.sum(density)), 1.0e-12)
            velocity_flat[row_index] = int(rng.choice(velocity_count, p=probability))
        vx_index, vy_index = np.unravel_index(
            velocity_flat, (case["vx"].size, case["vy"].size)
        )
        points = np.stack(
            (
                case["t"][t_index],
                case["x"][x_index],
                case["y"][y_index],
                case["vx"][vx_index],
                case["vy"][vy_index],
            ),
            axis=-1,
        )
        rows["points"].append(points)
        rows["target_rho"].append(
            case["rho"][t_index, x_index, y_index]
        )
        rows["target_g"].append(
            case["g"][t_index, x_index, y_index, vx_index, vy_index]
        )
        rows["case_indices"].append(np.full(count, case_index, dtype=np.int64))
        rows["equation_condition"].append(
            np.broadcast_to(case["equation_condition"], (count, 2))
        )
        mask = np.zeros(count, dtype=bool)
        mask[:density_count] = True
        rows["density_sample_mask"].append(mask)
    return {
        key: np.concatenate(value, axis=0)
        for key, value in rows.items()
    }


def flow_reference_point_loss(
    model,
    latent_table,
    batch: dict[str, np.ndarray],
    *,
    config: VPFPFlowReconstructionLossConfig,
):
    torch = _torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    indices = torch.as_tensor(
        batch["case_indices"], dtype=torch.long, device=device
    )
    latent = latent_table[indices]
    points = torch.as_tensor(batch["points"], dtype=dtype, device=device)
    condition = torch.as_tensor(
        batch["equation_condition"], dtype=dtype, device=device
    )
    target_rho = torch.as_tensor(
        batch["target_rho"], dtype=dtype, device=device
    )
    target_g = torch.as_tensor(batch["target_g"], dtype=dtype, device=device)
    density_mask = torch.as_tensor(
        batch["density_sample_mask"], dtype=torch.bool, device=device
    )
    uniform_mask = ~density_mask
    prediction = model.fields(
        points, latent, equation_condition=condition
    )
    epsilon = torch.finfo(dtype).eps
    rho_log = torch.mean(
        (
            torch.log(prediction["rho"].clamp_min(epsilon))
            - torch.log(target_rho.clamp_min(epsilon))
        ).square()
    )
    rho_relative = torch.mean(
        (prediction["rho"] - target_rho).square()
    ) / torch.mean(target_rho.square()).clamp_min(epsilon)
    if bool(torch.any(density_mask)):
        velocity_nll = -torch.mean(
            torch.log(prediction["g"][density_mask].clamp_min(1.0e-12))
        )
    else:
        velocity_nll = prediction["g"].sum() * 0.0
    comparison_mask = uniform_mask if bool(torch.any(uniform_mask)) else ~uniform_mask
    predicted_g = prediction["g"][comparison_mask]
    compared_g = target_g[comparison_mask]
    velocity_hellinger = torch.mean(
        (
            torch.sqrt(predicted_g.clamp_min(0.0))
            - torch.sqrt(compared_g.clamp_min(0.0))
        ).square()
    ) / torch.mean(compared_g).clamp_min(1.0e-8)
    target_f = target_rho[comparison_mask] * compared_g
    predicted_f = prediction["rho"][comparison_mask] * predicted_g
    f_relative = torch.mean((predicted_f - target_f).square()) / torch.mean(
        target_f.square()
    ).clamp_min(1.0e-8)
    latent_l2 = torch.mean(latent.square())
    total = (
        float(config.rho_log_weight) * rho_log
        + float(config.rho_relative_weight) * rho_relative
        + float(config.velocity_nll_weight) * velocity_nll
        + float(config.velocity_hellinger_weight) * velocity_hellinger
        + float(config.f_relative_weight) * f_relative
        + float(config.latent_l2_weight) * latent_l2
    )
    return total, {
        "point_loss": total,
        "rho_log_mse": rho_log,
        "rho_relative_l2": torch.sqrt(rho_relative),
        "velocity_nll": velocity_nll,
        "velocity_hellinger": velocity_hellinger,
        "f_relative_l2": torch.sqrt(f_relative),
        "latent_l2": latent_l2,
    }


def sample_flow_rho_slices(
    cases: Sequence[dict[str, Any]],
    case_indices: Sequence[int],
    *,
    slices_per_case: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int]:
    points = []
    target_rho = []
    repeated_case_indices = []
    conditions = []
    spatial_shape = None
    for raw_index in case_indices:
        case_index = int(raw_index)
        case = cases[case_index]
        current_shape = (int(case["x"].size), int(case["y"].size))
        if spatial_shape is None:
            spatial_shape = current_shape
        elif spatial_shape != current_shape:
            raise ValueError("rho slice batching requires a common spatial grid")
        x_grid, y_grid = np.meshgrid(
            case["x"], case["y"], indexing="ij"
        )
        count = x_grid.size
        for _ in range(int(slices_per_case)):
            time_index = int(rng.integers(case["rho"].shape[0]))
            points.append(
                np.stack(
                    (
                        np.full(count, case["t"][time_index]),
                        x_grid.reshape(-1),
                        y_grid.reshape(-1),
                        np.zeros(count),
                        np.zeros(count),
                    ),
                    axis=-1,
                )
            )
            target_rho.append(case["rho"][time_index].reshape(-1))
            repeated_case_indices.append(
                np.full(count, case_index, dtype=np.int64)
            )
            conditions.append(
                np.broadcast_to(case["equation_condition"], (count, 2))
            )
    if spatial_shape is None:
        raise ValueError("cannot sample rho slices from an empty batch")
    return {
        "points": np.concatenate(points, axis=0),
        "target_rho": np.stack(target_rho, axis=0),
        "case_indices": np.concatenate(repeated_case_indices, axis=0),
        "equation_condition": np.concatenate(conditions, axis=0),
        "spatial_nx": int(spatial_shape[0]),
        "spatial_ny": int(spatial_shape[1]),
    }


def flow_rho_slice_loss(
    model,
    latent_table,
    batch: dict[str, np.ndarray | int],
    *,
    config: VPFPFlowReconstructionLossConfig,
):
    torch = _torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    indices = torch.as_tensor(
        batch["case_indices"], dtype=torch.long, device=device
    )
    prediction = model.fields(
        torch.as_tensor(batch["points"], dtype=dtype, device=device),
        latent_table[indices],
        equation_condition=torch.as_tensor(
            batch["equation_condition"], dtype=dtype, device=device
        ),
    )
    nx = int(batch["spatial_nx"])
    ny = int(batch["spatial_ny"])
    predicted_rho = prediction["rho"].reshape(-1, nx, ny)
    target_rho = torch.as_tensor(
        batch["target_rho"], dtype=dtype, device=device
    ).reshape(-1, nx, ny)
    epsilon = torch.finfo(dtype).eps
    difference = predicted_rho - target_rho
    relative = torch.sum(difference.square(), dim=(-2, -1)) / torch.sum(
        target_rho.square(), dim=(-2, -1)
    ).clamp_min(epsilon)

    predicted_anomaly = (
        predicted_rho - torch.mean(predicted_rho, dim=(-2, -1), keepdim=True)
    )
    target_anomaly = (
        target_rho - torch.mean(target_rho, dim=(-2, -1), keepdim=True)
    )
    predicted_spectrum = torch.fft.fftn(
        predicted_anomaly, dim=(-2, -1), norm="ortho"
    )
    target_spectrum = torch.fft.fftn(
        target_anomaly, dim=(-2, -1), norm="ortho"
    )
    wave_x = torch.fft.fftfreq(nx, d=1.0 / float(nx), device=device)
    wave_y = torch.fft.fftfreq(ny, d=1.0 / float(ny), device=device)
    frequency_weight = 1.0 + float(
        config.rho_spectral_frequency_scale
    ) * (
        wave_x[:, None].square() + wave_y[None, :].square()
    )
    spectral_numerator = torch.sum(
        torch.abs(predicted_spectrum - target_spectrum).square()
        * frequency_weight,
        dim=(-2, -1),
    )
    spectral_denominator = torch.maximum(
        torch.sum(
            torch.abs(target_spectrum).square() * frequency_weight,
            dim=(-2, -1),
        ),
        1.0e-6 * torch.sum(target_rho.square(), dim=(-2, -1)),
    )
    spectral_relative = spectral_numerator / spectral_denominator
    total = (
        float(config.rho_slice_weight) * torch.mean(relative)
        + float(config.rho_spectral_weight)
        * torch.mean(spectral_relative)
    )
    return total, {
        "rho_slice_loss": total,
        "rho_slice_relative_l2": torch.sqrt(torch.mean(relative)),
        "rho_spectral_relative_l2": torch.sqrt(
            torch.mean(spectral_relative)
        ),
        "rho_slice_mean_max_abs": torch.max(
            torch.abs(
                torch.mean(predicted_rho, dim=(-2, -1))
                - torch.mean(target_rho, dim=(-2, -1))
            )
        ),
    }


def sample_flow_velocity_slices(
    cases: Sequence[dict[str, Any]],
    case_indices: Sequence[int],
    *,
    slices_per_case: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int]:
    points = []
    target_g = []
    weights = []
    repeated_case_indices = []
    conditions = []
    velocity_count = None
    monomials = None
    characteristic_features = None
    for raw_index in case_indices:
        case_index = int(raw_index)
        case = cases[case_index]
        current_velocity_count = case["vx"].size * case["vy"].size
        if velocity_count is None:
            velocity_count = current_velocity_count
            vx_grid, vy_grid = np.meshgrid(
                case["vx"], case["vy"], indexing="ij"
            )
            monomials = np.stack(
                [
                    vx_grid.reshape(-1) ** power_x
                    * vy_grid.reshape(-1) ** power_y
                    for power_x, power_y in FLOW_MOMENT_POWERS
                ],
                axis=-1,
            )
            characteristic_features = np.stack(
                [
                    trig(wave_x * vx_grid + wave_y * vy_grid).reshape(-1)
                    for wave_x, wave_y in FLOW_CHARACTERISTIC_WAVES
                    for trig in (np.cos, np.sin)
                ],
                axis=-1,
            )
        elif velocity_count != current_velocity_count:
            raise ValueError("velocity slice batching requires a common grid size")
        for _ in range(int(slices_per_case)):
            ti = int(rng.integers(case["rho"].shape[0]))
            xi = int(rng.integers(case["rho"].shape[1]))
            yi = int(rng.integers(case["rho"].shape[2]))
            vx_grid, vy_grid = np.meshgrid(
                case["vx"], case["vy"], indexing="ij"
            )
            count = vx_grid.size
            points.append(
                np.stack(
                    (
                        np.full(count, case["t"][ti]),
                        np.full(count, case["x"][xi]),
                        np.full(count, case["y"][yi]),
                        vx_grid.reshape(-1),
                        vy_grid.reshape(-1),
                    ),
                    axis=-1,
                )
            )
            target_g.append(case["g"][ti, xi, yi].reshape(-1))
            weights.append(case["velocity_weights"].reshape(-1))
            repeated_case_indices.append(
                np.full(count, case_index, dtype=np.int64)
            )
            conditions.append(
                np.broadcast_to(case["equation_condition"], (count, 2))
            )
    if (
        velocity_count is None
        or monomials is None
        or characteristic_features is None
    ):
        raise ValueError("cannot sample velocity slices from an empty batch")
    return {
        "points": np.concatenate(points, axis=0),
        "target_g": np.stack(target_g, axis=0),
        "weights": np.stack(weights, axis=0),
        "case_indices": np.concatenate(repeated_case_indices, axis=0),
        "equation_condition": np.concatenate(conditions, axis=0),
        "velocity_count": int(velocity_count),
        "monomials": monomials.astype(np.float32),
        "characteristic_features": characteristic_features.astype(np.float32),
    }


def flow_velocity_slice_loss(
    model,
    latent_table,
    batch: dict[str, np.ndarray | int],
    *,
    config: VPFPFlowReconstructionLossConfig,
):
    torch = _torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    indices = torch.as_tensor(
        batch["case_indices"], dtype=torch.long, device=device
    )
    prediction = model.fields(
        torch.as_tensor(batch["points"], dtype=dtype, device=device),
        latent_table[indices],
        equation_condition=torch.as_tensor(
            batch["equation_condition"], dtype=dtype, device=device
        ),
    )
    velocity_count = int(batch["velocity_count"])
    predicted_g = prediction["g"].reshape(-1, velocity_count)
    target_g = torch.as_tensor(
        batch["target_g"], dtype=dtype, device=device
    )
    weights = torch.as_tensor(batch["weights"], dtype=dtype, device=device)
    epsilon = torch.finfo(dtype).eps
    slice_density = torch.sum(
        (predicted_g - target_g).square() * weights, dim=-1
    ) / torch.sum(target_g.square() * weights, dim=-1).clamp_min(epsilon)
    window_mass = torch.sum(predicted_g * weights, dim=-1)
    normalization_loss = torch.mean((window_mass - 1.0).square())
    monomials = torch.as_tensor(
        batch["monomials"], dtype=dtype, device=device
    )
    predicted_moments = (predicted_g * weights) @ monomials
    target_moments = (target_g * weights) @ monomials
    moment_loss = torch.mean(
        (predicted_moments - target_moments).square()
        / (0.05 + target_moments.square())
    )
    characteristic_features = torch.as_tensor(
        batch["characteristic_features"], dtype=dtype, device=device
    )
    predicted_characteristics = (
        predicted_g * weights
    ) @ characteristic_features
    target_characteristics = (
        target_g * weights
    ) @ characteristic_features
    characteristic_loss = torch.mean(
        (predicted_characteristics - target_characteristics).square()
        / (0.05 + target_characteristics.square())
    )
    total = (
        float(config.slice_density_weight) * torch.mean(slice_density)
        + float(config.moment_weight) * moment_loss
        + float(config.window_normalization_weight) * normalization_loss
        + float(config.characteristic_weight) * characteristic_loss
    )
    return total, {
        "slice_loss": total,
        "slice_g_relative_l2": torch.sqrt(torch.mean(slice_density)),
        "moment_loss": moment_loss,
        "characteristic_loss": characteristic_loss,
        "window_normalization_loss": normalization_loss,
        "window_normalization_max_abs": torch.max(torch.abs(window_mass - 1.0)),
    }


def evaluate_flow_reference_case(
    model,
    latent,
    case: dict[str, Any],
    *,
    chunk_size: int = 32768,
) -> dict[str, float]:
    torch = _torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    t_grid, x_grid, y_grid = np.meshgrid(
        case["t"], case["x"], case["y"], indexing="ij"
    )
    txy = np.stack((t_grid, x_grid, y_grid), axis=-1).reshape(-1, 3)
    rho_points = np.concatenate(
        (txy, np.zeros((txy.shape[0], 2), dtype=np.float32)), axis=-1
    )
    condition = torch.as_tensor(
        case["equation_condition"], dtype=dtype, device=device
    )
    with torch.no_grad():
        rho_fields = model.fields(
            torch.as_tensor(rho_points, dtype=dtype, device=device),
            latent,
            equation_condition=condition,
        )
        rho_prediction = (
            rho_fields["rho"]
            .detach()
            .cpu()
            .numpy()
            .reshape(case["rho"].shape)
        )
        hermite_coefficients = rho_fields.get("hermite_coefficients")
    velocity_x, velocity_y = np.meshgrid(
        case["vx"], case["vy"], indexing="ij"
    )
    velocity = np.stack(
        (velocity_x.reshape(-1), velocity_y.reshape(-1)), axis=-1
    )
    phase_points = np.concatenate(
        (
            np.repeat(txy, velocity.shape[0], axis=0),
            np.tile(velocity, (txy.shape[0], 1)),
        ),
        axis=-1,
    )
    g_rows = []
    velocity_count = int(velocity.shape[0])
    with torch.no_grad():
        for start in range(0, phase_points.shape[0], int(chunk_size)):
            stop = min(
                start + int(chunk_size),
                phase_points.shape[0],
            )
            point_chunk = torch.as_tensor(
                phase_points[start:stop],
                dtype=dtype,
                device=device,
            )
            if (
                hermite_coefficients is not None
                and hasattr(model, "hermite_modes")
            ):
                coefficient_indices = (
                    torch.arange(start, stop, device=device)
                    // velocity_count
                )
                g_chunk = torch_squared_hermite_density(
                    torch,
                    point_chunk[:, 3:5],
                    hermite_coefficients[coefficient_indices],
                    temperature=condition[1],
                    modes=model.hermite_modes,
                    floor_mixture=float(
                        model.config.velocity_floor_mixture
                    ),
                )
            else:
                g_chunk = model.fields(
                    point_chunk,
                    latent,
                    equation_condition=condition,
                )["g"]
            g_rows.append(g_chunk.detach().cpu().numpy())
    g_prediction = np.concatenate(g_rows).reshape(case["g"].shape)
    rho_target = np.asarray(case["rho"], dtype=np.float64)
    g_target = np.asarray(case["g"], dtype=np.float64)
    weights = np.asarray(case["velocity_weights"], dtype=np.float64)
    f_target = rho_target[..., None, None] * g_target
    f_prediction = rho_prediction[..., None, None] * g_prediction

    def relative_l2(first, second, weight=None):
        difference = np.asarray(first, dtype=np.float64) - np.asarray(
            second, dtype=np.float64
        )
        if weight is None:
            numerator = np.sum(difference**2)
            denominator = np.sum(np.asarray(second, dtype=np.float64) ** 2)
        else:
            numerator = np.sum(difference**2 * weight)
            denominator = np.sum(
                np.asarray(second, dtype=np.float64) ** 2 * weight
            )
        return float(np.sqrt(numerator / max(denominator, 1.0e-20)))

    window_mass = np.sum(g_prediction * weights, axis=(-2, -1))
    log_mask = g_target > 1.0e-8
    return {
        "rho_rel_l2": relative_l2(rho_prediction, rho_target),
        "g_rel_l2": relative_l2(g_prediction, g_target, weights),
        "f_rel_l2": relative_l2(f_prediction, f_target, weights),
        "log_g_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        np.log(np.maximum(g_prediction[log_mask], 1.0e-12))
                        - np.log(g_target[log_mask])
                    )
                    ** 2
                )
            )
        ),
        "window_g_normalization_max_abs": float(
            np.max(np.abs(window_mass - 1.0))
        ),
    }


def fit_latent_normalization(
    latents, *, min_std: float = 1.0e-3
) -> dict[str, Any]:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("latents must have shape [N,D]")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.maximum(np.std(values, axis=0), float(min_std)).tolist(),
        "min_std": float(min_std),
    }


def save_flow_autoencoder_checkpoint(
    path: str | Path,
    *,
    decoder,
    encoder,
    feature_normalization: dict[str, Any],
    latent_normalization: dict[str, Any],
    case_latents,
    case_ids: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    train_config: dict[str, Any],
    step: int,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_continuous_flow_autoencoder_checkpoint_v1",
            "decoder_config": decoder.config.to_dict(),
            "encoder_config": encoder.config.to_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "feature_normalization": feature_normalization,
            "latent_normalization": latent_normalization,
            "case_latents": torch.as_tensor(case_latents).detach().cpu(),
            "case_ids": [str(value) for value in case_ids],
            "train_indices": [int(value) for value in train_indices],
            "validation_indices": [int(value) for value in validation_indices],
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_flow_autoencoder_checkpoint(
    path: str | Path, *, device: str = "cpu"
):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != "vpfp_continuous_flow_autoencoder_checkpoint_v1":
        raise ValueError("unexpected VPFP continuous-flow checkpoint schema")
    decoder = VPFPContinuousFlowField.build(
        VPFPContinuousFlowConfig.from_dict(payload["decoder_config"])
    ).to(device=device)
    encoder = VPFPFlowObservationEncoder.build(
        VPFPFlowObservationEncoderConfig.from_dict(payload["encoder_config"])
    ).to(device=device)
    decoder.load_state_dict(payload["decoder_state_dict"])
    encoder.load_state_dict(payload["encoder_state_dict"])
    decoder.eval()
    encoder.eval()
    return encoder, decoder, payload


def save_hermite_autoencoder_checkpoint(
    path: str | Path,
    *,
    decoder,
    encoder,
    feature_normalization: dict[str, Any],
    latent_normalization: dict[str, Any],
    case_latents,
    case_ids: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    train_config: dict[str, Any],
    step: int,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vpfp_continuous_hermite_autoencoder_checkpoint_v1",
            "decoder_config": decoder.config.to_dict(),
            "encoder_config": encoder.config.to_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "feature_normalization": feature_normalization,
            "latent_normalization": latent_normalization,
            "case_latents": torch.as_tensor(case_latents).detach().cpu(),
            "case_ids": [str(value) for value in case_ids],
            "train_indices": [int(value) for value in train_indices],
            "validation_indices": [int(value) for value in validation_indices],
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_hermite_autoencoder_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
):
    torch = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if (
        payload.get("schema")
        != "vpfp_continuous_hermite_autoencoder_checkpoint_v1"
    ):
        raise ValueError("unexpected VPFP continuous-Hermite checkpoint schema")
    decoder = VPFPContinuousHermiteField.build(
        VPFPContinuousHermiteConfig.from_dict(payload["decoder_config"])
    ).to(device=device)
    encoder = VPFPFlowObservationEncoder.build(
        VPFPFlowObservationEncoderConfig.from_dict(payload["encoder_config"])
    ).to(device=device)
    decoder.load_state_dict(payload["decoder_state_dict"])
    encoder.load_state_dict(payload["encoder_state_dict"])
    decoder.eval()
    encoder.eval()
    return encoder, decoder, payload


def summarize_metric_rows(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    output = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        }
    return output


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError(
            "torch is required for VPFP continuous-flow training"
        ) from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE


def flow_latent_swap_margin_loss(
    model,
    latent_table,
    batch: dict[str, np.ndarray],
    *,
    margin: float,
):
    """Require a case's latent to outperform a different latent at fixed queries."""

    torch = _torch()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    indices = torch.as_tensor(
        batch["case_indices"], dtype=torch.long, device=device
    )
    unique_indices = torch.unique(indices)
    if int(unique_indices.numel()) < 2:
        zero = torch.mean(latent_table[indices].square()) * 0.0
        return zero, {
            "latent_swap_loss": zero,
            "latent_swap_correct_error": zero,
            "latent_swap_wrong_error": zero,
            "latent_swap_error_gap": zero,
        }
    permuted = torch.roll(unique_indices, shifts=1)
    wrong_indices = indices.clone()
    for correct_index, wrong_index in zip(
        unique_indices, permuted, strict=True
    ):
        wrong_indices = torch.where(
            indices == correct_index, wrong_index, wrong_indices
        )
    points = torch.as_tensor(
        batch["points"], dtype=dtype, device=device
    )
    condition = torch.as_tensor(
        batch["equation_condition"], dtype=dtype, device=device
    )
    target_rho = torch.as_tensor(
        batch["target_rho"], dtype=dtype, device=device
    )
    target_g = torch.as_tensor(
        batch["target_g"], dtype=dtype, device=device
    )
    correct = model.fields(
        points,
        latent_table[indices],
        equation_condition=condition,
    )
    wrong = model.fields(
        points,
        latent_table[wrong_indices],
        equation_condition=condition,
    )

    def point_error(prediction):
        rho_error = (
            torch.log(prediction["rho"].clamp_min(1.0e-8))
            - torch.log(target_rho.clamp_min(1.0e-8))
        ).square()
        g_error = (
            torch.sqrt(prediction["g"].clamp_min(0.0))
            - torch.sqrt(target_g.clamp_min(0.0))
        ).square() / target_g.clamp_min(0.02)
        return rho_error + g_error

    correct_error = point_error(correct)
    wrong_error = point_error(wrong)
    swap_loss = torch.mean(
        torch.relu(float(margin) + correct_error - wrong_error)
    )
    correct_mean = torch.mean(correct_error)
    wrong_mean = torch.mean(wrong_error)
    return swap_loss, {
        "latent_swap_loss": swap_loss,
        "latent_swap_correct_error": correct_mean,
        "latent_swap_wrong_error": wrong_mean,
        "latent_swap_error_gap": wrong_mean - correct_mean,
    }
