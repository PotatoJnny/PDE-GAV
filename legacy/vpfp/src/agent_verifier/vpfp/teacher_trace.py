from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from agent_verifier.vpfp.high_fidelity import VPFPHighFidelityConfig


TRACE_SCHEMA = "vpfp_canonical_teacher_trace_v1"
TRACE_NORMALIZATION_SCHEMA = "vpfp_teacher_trace_normalization_v1"
RHO_TOKEN_DIM = 5
COMPONENT_TOKEN_DIM = 61
SOURCE_TOKEN_DIM = 17
SCALAR_TOKEN_DIM = 3


@dataclass(frozen=True)
class VPFPTeacherTraceConfig:
    rho_mode_count: int = 4
    maximum_component_count: int = 10
    component_weight_mode_count: int = 3
    component_parameter_mode_count: int = 2
    source_mode_count: int = 12
    maximum_spatial_mode: int = 4
    velocity_center_range: float = 2.0

    @classmethod
    def from_equation_config(
        cls, config: VPFPHighFidelityConfig
    ) -> "VPFPTeacherTraceConfig":
        config.validate()
        result = cls(
            rho_mode_count=int(config.rho_mode_count),
            maximum_component_count=int(config.g0_component_range[1]),
            component_weight_mode_count=int(config.g0_weight_mode_count),
            component_parameter_mode_count=int(config.g0_parameter_mode_count),
            source_mode_count=int(config.source_mode_count),
            maximum_spatial_mode=int(config.max_spatial_mode),
            velocity_center_range=float(config.velocity_center_range),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if min(
            int(self.rho_mode_count),
            int(self.maximum_component_count),
            int(self.component_weight_mode_count),
            int(self.component_parameter_mode_count),
            int(self.source_mode_count),
            int(self.maximum_spatial_mode),
        ) < 1:
            raise ValueError("teacher trace dimensions must be positive")
        if float(self.velocity_center_range) <= 0.0:
            raise ValueError("teacher trace velocity range must be positive")
        expected = (
            6
            + 5 * int(self.component_weight_mode_count)
            + 4 * 5 * int(self.component_parameter_mode_count)
        )
        if expected != COMPONENT_TOKEN_DIM:
            raise ValueError(
                "teacher trace component token dimension changed"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VPFPTeacherTraceConfig":
        result = cls(**dict(payload))
        result.validate()
        return result


def pack_teacher_trace_specs(
    specs: Sequence[Mapping[str, Any]],
    config: VPFPTeacherTraceConfig,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    config.validate()
    if not specs:
        raise ValueError("teacher trace needs at least one equation spec")
    batch = len(specs)
    rho = torch.zeros(
        (batch, int(config.rho_mode_count), RHO_TOKEN_DIM), dtype=dtype
    )
    rho_mask = torch.zeros(
        (batch, int(config.rho_mode_count)), dtype=torch.bool
    )
    components = torch.zeros(
        (
            batch,
            int(config.maximum_component_count),
            COMPONENT_TOKEN_DIM,
        ),
        dtype=dtype,
    )
    component_mask = torch.zeros(
        (batch, int(config.maximum_component_count)), dtype=torch.bool
    )
    sources = torch.zeros(
        (batch, int(config.source_mode_count), SOURCE_TOKEN_DIM), dtype=dtype
    )
    source_mask = torch.zeros(
        (batch, int(config.source_mode_count)), dtype=torch.bool
    )
    scalars = torch.zeros((batch, SCALAR_TOKEN_DIM), dtype=dtype)
    for case_index, spec in enumerate(specs):
        if spec.get("schema") != "vpfp_continuous_equation_spec_v3":
            raise ValueError("teacher trace requires rich v3 equation specs")
        rho_modes = list(spec["rho_modes"])
        if len(rho_modes) > int(config.rho_mode_count):
            raise ValueError("teacher trace rho mode capacity exceeded")
        for token_index, mode in enumerate(rho_modes):
            rho[case_index, token_index] = torch.tensor(
                _spatial_mode_features(mode, config), dtype=dtype
            )
            rho_mask[case_index, token_index] = True

        case_components = list(spec["g0_components"])
        if len(case_components) > int(config.maximum_component_count):
            raise ValueError("teacher trace component capacity exceeded")
        for token_index, component in enumerate(case_components):
            components[case_index, token_index] = torch.tensor(
                _component_features(component, config), dtype=dtype
            )
            component_mask[case_index, token_index] = True

        source_modes = list(spec["source_modes"])
        if len(source_modes) > int(config.source_mode_count):
            raise ValueError("teacher trace source capacity exceeded")
        for token_index, source in enumerate(source_modes):
            sources[case_index, token_index] = torch.tensor(
                _source_features(source, config), dtype=dtype
            )
            source_mask[case_index, token_index] = True
        scalars[case_index] = torch.tensor(
            (
                float(spec["collision_nu"]),
                float(spec["temperature"]),
                float(spec["background_density"]),
            ),
            dtype=dtype,
        )
    trace = {
        "schema": TRACE_SCHEMA,
        "config": config.to_dict(),
        "rho": rho.to(device),
        "rho_mask": rho_mask.to(device),
        "components": components.to(device),
        "component_mask": component_mask.to(device),
        "sources": sources.to(device),
        "source_mask": source_mask.to(device),
        "scalars": scalars.to(device),
    }
    validate_teacher_trace(trace, config)
    return trace


def fit_teacher_trace_normalization(
    trace: Mapping[str, Any],
    config: VPFPTeacherTraceConfig,
) -> dict[str, Any]:
    validate_teacher_trace(trace, config)
    result = {
        "schema": TRACE_NORMALIZATION_SCHEMA,
        "config": config.to_dict(),
    }
    for name, mask_name in (
        ("rho", "rho_mask"),
        ("components", "component_mask"),
        ("sources", "source_mask"),
    ):
        values = torch.as_tensor(trace[name], dtype=torch.float64)
        mask = torch.as_tensor(trace[mask_name], dtype=torch.bool)
        active = values[mask]
        if int(active.shape[0]) < 2:
            raise ValueError(f"teacher trace {name} lacks active tokens")
        result[name] = _fit_normalization(active)
    result["scalars"] = _fit_normalization(
        torch.as_tensor(trace["scalars"], dtype=torch.float64)
    )
    return result


def normalize_teacher_trace(
    trace: Mapping[str, Any],
    normalization: Mapping[str, Any],
    config: VPFPTeacherTraceConfig,
) -> dict[str, Any]:
    validate_teacher_trace(trace, config)
    if normalization.get("schema") != TRACE_NORMALIZATION_SCHEMA:
        raise ValueError("unexpected teacher trace normalization schema")
    if dict(normalization.get("config") or {}) != config.to_dict():
        raise ValueError("teacher trace normalization config changed")
    result = {
        "schema": TRACE_SCHEMA,
        "config": config.to_dict(),
    }
    for name, mask_name in (
        ("rho", "rho_mask"),
        ("components", "component_mask"),
        ("sources", "source_mask"),
    ):
        values = torch.as_tensor(trace[name])
        mask = torch.as_tensor(trace[mask_name], dtype=torch.bool)
        normalized = _apply_normalization(values, normalization[name])
        result[name] = normalized * mask[..., None].to(normalized.dtype)
        result[mask_name] = mask
    result["scalars"] = _apply_normalization(
        torch.as_tensor(trace["scalars"]), normalization["scalars"]
    )
    validate_teacher_trace(result, config)
    return result


def move_teacher_trace(
    trace: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in trace.items()
    }


def concatenate_teacher_traces(
    traces: Sequence[Mapping[str, Any]],
    config: VPFPTeacherTraceConfig,
) -> dict[str, Any]:
    if not traces:
        raise ValueError("cannot concatenate an empty teacher trace list")
    for trace in traces:
        validate_teacher_trace(trace, config)
    result = {
        "schema": TRACE_SCHEMA,
        "config": config.to_dict(),
    }
    for name in (
        "rho",
        "rho_mask",
        "components",
        "component_mask",
        "sources",
        "source_mask",
        "scalars",
    ):
        result[name] = torch.cat(
            [torch.as_tensor(trace[name]) for trace in traces], dim=0
        )
    validate_teacher_trace(result, config)
    return result


def index_teacher_trace(
    trace: Mapping[str, Any],
    indices,
    config: VPFPTeacherTraceConfig,
) -> dict[str, Any]:
    validate_teacher_trace(trace, config)
    result = {
        "schema": TRACE_SCHEMA,
        "config": config.to_dict(),
    }
    for name in (
        "rho",
        "rho_mask",
        "components",
        "component_mask",
        "sources",
        "source_mask",
        "scalars",
    ):
        result[name] = torch.as_tensor(trace[name])[indices]
    validate_teacher_trace(result, config)
    return result


def validate_teacher_trace(
    trace: Mapping[str, Any], config: VPFPTeacherTraceConfig
) -> None:
    config.validate()
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError("unexpected teacher trace schema")
    if dict(trace.get("config") or {}) != config.to_dict():
        raise ValueError("teacher trace config changed")
    rho = torch.as_tensor(trace["rho"])
    components = torch.as_tensor(trace["components"])
    sources = torch.as_tensor(trace["sources"])
    scalars = torch.as_tensor(trace["scalars"])
    batch = int(rho.shape[0])
    expected = {
        "rho": (batch, int(config.rho_mode_count), RHO_TOKEN_DIM),
        "rho_mask": (batch, int(config.rho_mode_count)),
        "components": (
            batch,
            int(config.maximum_component_count),
            COMPONENT_TOKEN_DIM,
        ),
        "component_mask": (
            batch,
            int(config.maximum_component_count),
        ),
        "sources": (batch, int(config.source_mode_count), SOURCE_TOKEN_DIM),
        "source_mask": (batch, int(config.source_mode_count)),
        "scalars": (batch, SCALAR_TOKEN_DIM),
    }
    for name, shape in expected.items():
        values = torch.as_tensor(trace[name])
        if tuple(values.shape) != shape:
            raise ValueError(f"teacher trace {name} shape changed")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError(f"teacher trace {name} is non-finite")
    for name in ("rho_mask", "component_mask", "source_mask"):
        if torch.as_tensor(trace[name]).dtype != torch.bool:
            raise ValueError(f"teacher trace {name} must be boolean")
    if int(components.shape[0]) != batch or int(sources.shape[0]) != batch:
        raise ValueError("teacher trace batch dimensions disagree")
    if int(scalars.shape[0]) != batch:
        raise ValueError("teacher trace scalar batch dimension changed")


def _spatial_mode_features(
    mode: Mapping[str, Any], config: VPFPTeacherTraceConfig
) -> tuple[float, ...]:
    phase = float(mode["phase"])
    scale = float(config.maximum_spatial_mode)
    return (
        float(mode["amplitude"]),
        float(mode["kx"]) / scale,
        float(mode["ky"]) / scale,
        math.sin(phase),
        math.cos(phase),
    )


def _component_features(
    component: Mapping[str, Any], config: VPFPTeacherTraceConfig
) -> tuple[float, ...]:
    center_scale = float(config.velocity_center_range)
    values = [
        float(component["base_logit"]),
        float(component["base_mean"][0]) / center_scale,
        float(component["base_mean"][1]) / center_scale,
        float(component["base_log_std"][0]),
        float(component["base_log_std"][1]),
        float(component["correlation"]),
    ]
    for name in (
        "weight_modes",
        "mean_x_modes",
        "mean_y_modes",
        "log_std_x_modes",
        "log_std_y_modes",
    ):
        modes = list(component[name])
        expected_count = (
            int(config.component_weight_mode_count)
            if name == "weight_modes"
            else int(config.component_parameter_mode_count)
        )
        if len(modes) != expected_count:
            raise ValueError("teacher trace component mode count changed")
        for mode in modes:
            values.extend(_spatial_mode_features(mode, config))
    if len(values) != COMPONENT_TOKEN_DIM:
        raise ValueError("teacher trace component feature count changed")
    return tuple(values)


def _source_features(
    source: Mapping[str, Any], config: VPFPTeacherTraceConfig
) -> tuple[float, ...]:
    phase = float(source["phase"])
    velocity_phase = float(source["velocity_phase"])
    mode_scale = float(config.maximum_spatial_mode)
    center_scale = float(config.velocity_center_range)
    values = (
        float(source["amplitude"]),
        float(source["kx"]) / mode_scale,
        float(source["ky"]) / mode_scale,
        float(source["omega"]),
        math.sin(phase),
        math.cos(phase),
        float(source["velocity_center"][0]) / center_scale,
        float(source["velocity_center"][1]) / center_scale,
        math.log(float(source["velocity_std"][0])),
        math.log(float(source["velocity_std"][1])),
        float(source["velocity_correlation"]),
        float(source["velocity_degree"][0]),
        float(source["velocity_degree"][1]),
        float(source["velocity_wave"][0]),
        float(source["velocity_wave"][1]),
        math.sin(velocity_phase),
        math.cos(velocity_phase),
    )
    if len(values) != SOURCE_TOKEN_DIM:
        raise ValueError("teacher trace source feature count changed")
    return values


def _fit_normalization(values: torch.Tensor) -> dict[str, list[float]]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _apply_normalization(values, payload) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    mean = torch.as_tensor(
        payload["mean"], dtype=tensor.dtype, device=tensor.device
    )
    scale = torch.as_tensor(
        payload["scale"], dtype=tensor.dtype, device=tensor.device
    )
    if int(mean.numel()) != int(tensor.shape[-1]):
        raise ValueError("teacher trace normalization width changed")
    return (tensor - mean) / scale
