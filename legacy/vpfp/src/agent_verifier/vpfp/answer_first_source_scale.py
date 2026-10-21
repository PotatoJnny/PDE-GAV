from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


SOURCE_ENVELOPE_SCHEMA = "vpfp_answer_first_induced_source_envelope_v1"


@dataclass(frozen=True)
class VPFPAnswerFirstSourceEnvelopeConfig:
    lower_quantile: float = 0.02
    upper_quantile: float = 0.98
    multiplicative_margin: float = 1.35
    minimum_scale: float = 1.0e-8
    minimum_passing_fraction: float = 0.95
    enforce_lower_rms: bool = True

    def validate(self) -> None:
        if not 0.0 <= float(self.lower_quantile) < float(
            self.upper_quantile
        ) <= 1.0:
            raise ValueError("source envelope quantiles are invalid")
        if float(self.multiplicative_margin) < 1.0:
            raise ValueError("source envelope margin must be at least one")
        if float(self.minimum_scale) <= 0.0:
            raise ValueError("minimum source scale must be positive")
        if not 0.0 < float(self.minimum_passing_fraction) <= 1.0:
            raise ValueError("source passing fraction must lie in (0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_first_source_scale_metrics(source_values) -> dict[str, np.ndarray]:
    """Return per-answer absolute induced-source statistics.

    These values are deliberately not residuals against zero or ratios to an
    equation-first source. In the answer-first construction this source is the
    exact public right-hand side associated with the generated hidden answer.
    """

    source = np.asarray(source_values, dtype=np.float64)
    if source.ndim == 1:
        source = source[None, :]
    if source.ndim < 2 or source.shape[0] < 1 or source.shape[1] < 1:
        raise ValueError("source values must have shape [case, point, ...]")
    source = source.reshape(source.shape[0], -1)
    if not np.all(np.isfinite(source)):
        raise ValueError("induced source must be finite")
    absolute = np.abs(source)
    rms = np.sqrt(np.mean(np.square(source), axis=1))
    safe_rms = np.maximum(rms, np.finfo(np.float64).tiny)
    q95 = np.quantile(absolute, 0.95, axis=1)
    q99 = np.quantile(absolute, 0.99, axis=1)
    maximum = np.max(absolute, axis=1)
    mean_absolute = np.mean(absolute, axis=1)
    return {
        "source_rms": rms,
        "source_abs_mean": mean_absolute,
        "source_abs_q95": q95,
        "source_abs_q99": q99,
        "source_abs_max": maximum,
        "source_abs_mean_over_rms": mean_absolute / safe_rms,
        "source_abs_q99_over_rms": q99 / safe_rms,
        "source_abs_max_over_rms": maximum / safe_rms,
    }


def fit_answer_first_source_envelope(
    source_values,
    *,
    config: VPFPAnswerFirstSourceEnvelopeConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = config or VPFPAnswerFirstSourceEnvelopeConfig()
    policy.validate()
    metrics = answer_first_source_scale_metrics(source_values)
    bounds = {}
    summaries = {}
    margin = float(policy.multiplicative_margin)
    floor = float(policy.minimum_scale)
    for name, values in metrics.items():
        lower_raw = float(np.quantile(values, float(policy.lower_quantile)))
        upper_raw = float(np.quantile(values, float(policy.upper_quantile)))
        lower = max(0.0, lower_raw / margin)
        if name == "source_rms" and bool(policy.enforce_lower_rms):
            lower = max(floor, lower)
        else:
            lower = 0.0
        upper = max(floor, upper_raw * margin)
        bounds[name] = {"minimum": lower, "maximum": upper}
        summaries[name] = {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
            "calibration_lower_quantile": lower_raw,
            "calibration_upper_quantile": upper_raw,
        }
    return {
        "schema": SOURCE_ENVELOPE_SCHEMA,
        "config": policy.to_dict(),
        "calibration_case_count": int(next(iter(metrics.values())).size),
        "bounds": bounds,
        "calibration_summary": summaries,
        "provenance": dict(provenance or {}),
        "semantics": {
            "primary_object": "generated_continuous_hidden_solution",
            "public_source": "exact_operator_image_of_generated_hidden_solution",
            "gate": "absolute_induced_source_scale_and_concentration",
            "not_a_gate": [
                "residual_to_zero",
                "matching_an_equation_first_source",
                "source_free_kinetic_dynamics",
                "equation_first_conservation_or_relaxation",
            ],
        },
    }


def audit_answer_first_source_scale(
    source_values,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_envelope(envelope)
    metrics = answer_first_source_scale_metrics(source_values)
    rows = {}
    failures = []
    required_fraction = float(
        envelope["config"].get("minimum_passing_fraction", 1.0)
    )
    for name, values in metrics.items():
        bound = dict(envelope["bounds"][name])
        minimum = float(bound["minimum"])
        maximum = float(bound["maximum"])
        passed = (values >= minimum) & (values <= maximum)
        failed_indices = np.flatnonzero(~passed)
        passing_fraction = float(np.mean(passed))
        metric_passed = passing_fraction >= required_fraction
        if not metric_passed:
            failures.append(name)
        rows[name] = {
            "minimum_allowed": minimum,
            "maximum_allowed": maximum,
            "observed_minimum": float(np.min(values)),
            "observed_median": float(np.median(values)),
            "observed_maximum": float(np.max(values)),
            "passing_fraction": passing_fraction,
            "required_passing_fraction": required_fraction,
            "passed": bool(metric_passed),
            "failed_case_indices": failed_indices.astype(int).tolist(),
        }
    return {
        "schema": "vpfp_answer_first_induced_source_scale_audit_v1",
        "case_count": int(next(iter(metrics.values())).size),
        "required_passing_fraction": required_fraction,
        "passed": not failures,
        "failed_metrics": failures,
        "metrics": rows,
        "answer_first_semantics_preserved": True,
    }


def differentiable_answer_first_source_penalty(
    source_values,
    envelope: Mapping[str, Any],
):
    """Soft differentiable penalty for guidance in the generated-answer latent."""

    _validate_envelope(envelope)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for source guidance") from exc
    source = torch.as_tensor(source_values)
    if source.ndim == 1:
        source = source[None, :]
    source = source.reshape(int(source.shape[0]), -1)
    if not bool(torch.all(torch.isfinite(source)).item()):
        raise ValueError("induced source must be finite")
    epsilon = torch.finfo(source.dtype).tiny
    absolute = torch.abs(source)
    rms = torch.sqrt(torch.mean(source.square(), dim=1)).clamp_min(epsilon)
    q99 = torch.quantile(absolute, 0.99, dim=1)
    maximum = torch.amax(absolute, dim=1)
    values = {
        "source_rms": rms,
        "source_abs_q99": q99,
        "source_abs_q99_over_rms": q99 / rms,
        "source_abs_max_over_rms": maximum / rms,
    }
    penalties = []
    diagnostics = {}
    for name, value in values.items():
        bound = dict(envelope["bounds"][name])
        lower = max(float(bound["minimum"]), float(epsilon))
        upper = max(float(bound["maximum"]), lower)
        lower_violation = torch.relu(
            torch.log(torch.as_tensor(lower, dtype=value.dtype, device=value.device))
            - torch.log(value.clamp_min(epsilon))
        )
        upper_violation = torch.relu(
            torch.log(value.clamp_min(epsilon))
            - torch.log(torch.as_tensor(upper, dtype=value.dtype, device=value.device))
        )
        metric_penalty = lower_violation.square() + upper_violation.square()
        penalties.append(metric_penalty)
        diagnostics[name] = value.detach()
    per_case = torch.stack(penalties, dim=1).mean(dim=1)
    return per_case.mean(), {
        "per_case_penalty": per_case,
        "metrics": diagnostics,
        "compares_to_equation_first_source": False,
    }


def _validate_envelope(envelope: Mapping[str, Any]) -> None:
    if envelope.get("schema") != SOURCE_ENVELOPE_SCHEMA:
        raise ValueError("unexpected answer-first source envelope schema")
    required = set(answer_first_source_scale_metrics(np.ones((1, 4))))
    bounds = dict(envelope.get("bounds") or {})
    missing = required - set(bounds)
    if missing:
        raise ValueError(f"source envelope is missing metrics: {sorted(missing)}")
    for name in required:
        minimum = float(bounds[name]["minimum"])
        maximum = float(bounds[name]["maximum"])
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("source envelope bounds must be finite")
        if minimum < 0.0 or maximum < minimum:
            raise ValueError("source envelope bounds are invalid")
