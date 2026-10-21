"""Physical geometry, norms and positive kinetic MMS perturbations.

These helpers complement ``sample_multiplicative_mms_perturbation``. They do
not identify latent coefficient distance with physical solution distance.
The PDE plugin still owns the child Poisson/field solve, manufactured source,
initial/boundary data, independent residual checks and Case materialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp, isfinite, pi
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .perturbation import (
    MultiplicativePerturbationSettings, PerturbationValidation,
    RelativeChangeRange, SmoothRandomFieldSettings, SmoothRandomFourierField,
    sample_multiplicative_mms_perturbation,
)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class PerturbationGeometry:
    """Ordered physical coordinates; ``None`` means a nonperiodic axis."""

    coordinates: tuple[str, ...]
    length_scales: tuple[float, ...]
    periods: tuple[float | None, ...]
    term_count: int = 32
    maximum_frequency_factor: float = 4.0

    def __post_init__(self):
        if not self.coordinates or len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError("coordinates must be nonempty and unique")
        if any(not isinstance(name, str) or not name for name in self.coordinates):
            raise ValueError("coordinate names must be nonempty text")
        if len(self.length_scales) != len(self.coordinates) or len(self.periods) != len(self.coordinates):
            raise ValueError("coordinates, length scales and periods must align")
        settings = SmoothRandomFieldSettings(tuple(self.length_scales), self.term_count, self.maximum_frequency_factor)
        for period in self.periods:
            if period is not None:
                _positive(period, "period")
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "length_scales", settings.length_scales)
        object.__setattr__(self, "periods", tuple(self.periods))

    def draw(self, seed: int) -> SmoothRandomFourierField:
        """Integer Fourier modes make periodic values AND derivatives match."""
        initial = SmoothRandomFourierField.sample(SmoothRandomFieldSettings(self.length_scales, self.term_count, self.maximum_frequency_factor), seed=seed)
        frequencies = []
        for row in initial.frequencies:
            converted = []
            for frequency, scale, period in zip(row, self.length_scales, self.periods):
                if period is None:
                    converted.append(frequency)
                else:
                    max_mode = int(self.maximum_frequency_factor * period / (2 * pi * scale))
                    mode = max(-max_mode, min(max_mode, round(frequency * period / (2 * pi))))
                    converted.append(2 * pi * mode / period)
            frequencies.append(tuple(converted))
        return SmoothRandomFourierField(initial.length_scales, tuple(frequencies), initial.phases, initial.weights, seed)


def weighted_relative_field_change(
    parent: Mapping[str, object],
    child: Mapping[str, object],
    *,
    quadrature_weights: Sequence[float],
    field_scales: Mapping[str, float],
    reference_floor: float = 1.0e-8,
) -> float:
    """Dimensionless joint L2 distance with explicit physical field scales.

    Values have shape ``[point, ...components]``. Each field is first divided
    by its fixed declared physical scale, then integrated by the same positive
    quadrature. The reference floor applies to this dimensionless norm and
    handles a zero parent. Scales and quadrature must be fixed for the entire
    comparison/qualification, not fitted independently to each child.
    """
    if not field_scales or set(parent) != set(field_scales) or set(child) != set(field_scales):
        raise ValueError("parent, child and field scales must have identical nonempty fields")
    weights = np.asarray(quadrature_weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("quadrature weights must be finite, positive and one-dimensional")
    floor = _positive(reference_floor, "reference floor")
    numerator = denominator = 0.0
    components = 0
    for name, raw_scale in field_scales.items():
        scale = _positive(raw_scale, f"{name} scale")
        left = np.asarray(parent[name], dtype=np.float64)
        right = np.asarray(child[name], dtype=np.float64)
        if left.ndim < 1 or left.shape != right.shape or left.shape[0] != len(weights):
            raise ValueError(f"{name} values must share shape [quadrature point, ...components]")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError(f"{name} values must be finite")
        left = left.reshape(len(weights), -1) / scale
        difference = right.reshape(len(weights), -1) / scale - left
        numerator += float(np.dot(weights, np.sum(difference ** 2, axis=1)))
        denominator += float(np.dot(weights, np.sum(left ** 2, axis=1)))
        components += left.shape[1]
    result = np.sqrt(numerator / max(denominator, floor ** 2 * weights.sum() * components))
    if not np.isfinite(result):
        raise ValueError("physical norm overflowed; rescale declared field units")
    return float(result)


@dataclass(frozen=True)
class PhysicalPerturbationPolicy:
    """One explicit admissible neighbourhood shared across campaign phases.

    Every named residual is nonnegative and dimensionless. Its definition,
    quadrature, units and scales are bound by ``metric_definition_sha256``.
    Missing checks fail closed. Tolerances are experiment choices: this SDK
    deliberately does not invent cross-PDE universal thresholds.
    """

    solution_change: RelativeChangeRange
    maximum_source_change: float
    maximum_residuals: Mapping[str, float]
    metric_definition_sha256: str

    def __post_init__(self):
        if not isinstance(self.solution_change, RelativeChangeRange):
            raise TypeError("solution_change must be RelativeChangeRange")
        _positive(self.maximum_source_change, "maximum source change")
        if len(self.metric_definition_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.metric_definition_sha256):
            raise ValueError("metric_definition_sha256 must be a SHA256")
        if not {"mms_residual", "boundary_residual"}.issubset(self.maximum_residuals):
            raise ValueError("MMS and boundary residual limits are required")
        limits = {}
        for name, value in self.maximum_residuals.items():
            if not isinstance(name, str) or not name or isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError("residual limits must be named, finite and nonnegative")
            limits[name] = float(value)
        object.__setattr__(self, "maximum_residuals", MappingProxyType(limits))

    @property
    def sha256(self) -> str:
        payload = {"solution_change": [self.solution_change.minimum, self.solution_change.maximum], "maximum_source_change": self.maximum_source_change, "maximum_residuals": dict(self.maximum_residuals), "metric_definition_sha256": self.metric_definition_sha256}
        return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def validate(self, *, solution_change: float, source_change: float, residuals: Mapping[str, float], metric_definition_sha256: str) -> PerturbationValidation:
        if metric_definition_sha256 != self.metric_definition_sha256:
            raise ValueError("physical norm/constraint definition changed")
        if {"solution_relative_change", "source_relative_change"} & set(residuals):
            raise ValueError("residuals cannot overwrite physical distance metrics")
        measured = {"solution_relative_change": solution_change, "source_relative_change": source_change, **dict(residuals)}
        for name, value in measured.items():
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                return PerturbationValidation(False, f"invalid physical metric: {name}")
        missing = set(self.maximum_residuals) - set(residuals)
        if missing:
            return PerturbationValidation(False, "missing physical checks: " + ", ".join(sorted(missing)))
        if not self.solution_change.contains(solution_change):
            return PerturbationValidation(False, "physical solution change outside requested range", measured)
        if source_change > self.maximum_source_change:
            return PerturbationValidation(False, "manufactured source change exceeds limit", measured)
        failed = [name for name, limit in self.maximum_residuals.items() if residuals[name] > limit]
        return PerturbationValidation(not failed, "physical constraint exceeded: " + ", ".join(failed) if failed else "", measured)


@dataclass(frozen=True)
class ScalarJet:
    """Scalar value and physical-coordinate first/second derivatives."""

    value: float
    gradient: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        n = len(self.gradient)
        if not n or len(self.hessian) != n or any(len(row) != n for row in self.hessian):
            raise ValueError("jet derivatives must have matching dimensions")
        if not all(isfinite(value) for value in (self.value, *self.gradient, *(x for row in self.hessian for x in row))):
            raise ValueError("jet values and derivatives must be finite")


@dataclass(frozen=True)
class PositiveKineticMultiplier:
    """Continuous positive child ``f_child = f_parent * exp(alpha * delta)``.

    This kinetic alternative to the epsilon-shifted signed-field default
    preserves nonnegativity without clipping. It preserves spatial periodicity
    when ``delta`` uses the matching integer Fourier modes. It does NOT
    automatically preserve mass or satisfy Poisson: the plugin must derive
    child moments/field and validate them before accepting the manufactured
    question. A bounded multiplier preserves an integrable parent's tails.
    """

    parent: Callable[[Sequence[float]], ScalarJet]
    field: SmoothRandomFourierField
    alpha: float
    maximum_log_change: float = 8.0

    def __post_init__(self):
        if not callable(self.parent):
            raise TypeError("parent must return a ScalarJet")
        if not isinstance(self.field, SmoothRandomFourierField):
            raise TypeError("field must be a smooth Fourier field")
        if isinstance(self.alpha, bool) or not isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        _positive(self.maximum_log_change, "maximum log change")
        if self.alpha * sum(abs(weight) for weight in self.field.weights) > self.maximum_log_change:
            raise ValueError("global multiplier bound exceeds maximum log change")

    def __call__(self, point: Sequence[float]) -> ScalarJet:
        parent = self.parent(point)
        n = len(self.field.length_scales)
        if not isinstance(parent, ScalarJet) or len(parent.gradient) != n:
            raise TypeError("parent jet coordinate dimension differs")
        if parent.value < 0:
            raise ValueError("kinetic parent density cannot be negative")
        value = self.field(point)
        gradient = []
        hessian = []
        for i in range(n):
            order = [0] * n
            order[i] = 1
            gradient.append(self.field.derivative(point, tuple(order)))
            row = []
            for j in range(n):
                second = list(order)
                second[j] += 1
                row.append(self.field.derivative(point, tuple(second)))
            hessian.append(row)
        factor = exp(self.alpha * value)
        return ScalarJet(
            parent.value * factor,
            tuple(factor * (parent.gradient[i] + parent.value * self.alpha * gradient[i]) for i in range(n)),
            tuple(tuple(factor * (parent.hessian[i][j] + self.alpha * (parent.gradient[i] * gradient[j] + parent.gradient[j] * gradient[i] + parent.value * hessian[i][j]) + parent.value * self.alpha ** 2 * gradient[i] * gradient[j]) for j in range(n)) for i in range(n)),
        )


def manufacture_kinetic_source_1d1v(jet: ScalarJet, *, velocity: float, electric_field: float, collision_frequency: float = 0.0, thermal_variance: float = 1.0) -> float:
    """MMS source for f_t + v f_x + E f_v = nu d_v(v f + T f_v) + s.

    The jet coordinate order is ``(t, x, v)``. ``electric_field`` MUST be
    derived from the CHILD density with the experiment's Poisson sign and
    neutrality convention; reusing a changed parent's field is not valid.
    This function is the continuous source equation, not a residual test.
    Qualification must independently check Poisson and source derivatives.
    """
    if len(jet.gradient) != 3:
        raise ValueError("kinetic source needs the (t, x, v) jet")
    if not all(isfinite(value) for value in (velocity, electric_field, collision_frequency, thermal_variance)) or collision_frequency < 0 or thermal_variance <= 0:
        raise ValueError("kinetic equation parameters must be finite and admissible")
    return jet.gradient[0] + velocity * jet.gradient[1] + electric_field * jet.gradient[2] - collision_frequency * (jet.value + velocity * jet.gradient[2] + thermal_variance * jet.hessian[2][2])


def sample_positive_kinetic_mms(
    parent: Callable[[Sequence[float]], ScalarJet],
    *,
    geometry: PerturbationGeometry,
    settings: MultiplicativePerturbationSettings,
    seed: int,
    measure_relative_change,
    manufacture_problem,
    validate,
    maximum_log_change: float = 8.0,
):
    """Use shared alpha/field rejection with a positivity-preserving chart.

    Exponential composition is nonlinear; the shared sampler remeasures the
    physical norm and adjusts alpha after every proposal. The initial probe
    uses a global bound, so every attempted multiplier stays integrable.
    ``epsilon_fraction`` belongs to the signed-field default and is unused
    by this positive exponential chart. Identity, source and policy audit
    must report ``positive_exponential_mms`` when adopting this method.
    """
    from .sampling import RecoverableSamplingRejection

    _positive(maximum_log_change, "maximum log change")

    def draw(field_seed):
        field = geometry.draw(field_seed)
        # Scale the direction, not alpha, to bound every alpha the caller
        # allows. Accepted amplitude is still chosen by physical distance.
        bound = sum(abs(weight) for weight in field.weights)
        reduction = min(1.0, maximum_log_change * (1.0 - 1.0e-14) / (settings.maximum_alpha * bound))
        return SmoothRandomFourierField(field.length_scales, field.frequencies, field.phases, tuple(weight * reduction for weight in field.weights), field.seed)

    def compose(parent, field, alpha, _epsilon):
        return PositiveKineticMultiplier(parent, field, alpha, maximum_log_change)

    try:
        return sample_multiplicative_mms_perturbation(parent, settings=settings, seed=seed, draw_random_field=draw, compose_candidate=compose, measure_relative_change=measure_relative_change, manufacture_problem=manufacture_problem, validate=validate)
    except RecoverableSamplingRejection as error:
        audit = dict(error.audit)
        audit["method"] = "positive_exponential_mms"
        raise RecoverableSamplingRejection(str(error), audit=audit) from error


__all__ = ["PerturbationGeometry", "PhysicalPerturbationPolicy", "PositiveKineticMultiplier", "ScalarJet", "manufacture_kinetic_source_1d1v", "sample_positive_kinetic_mms", "weighted_relative_field_change"]
