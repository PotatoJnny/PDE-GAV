"""Common building blocks for continuous-solution perturbations.

The shared framework does not know how a particular PDE stores a continuous
solution.  A Generator plugin therefore supplies the small equation-specific
callbacks in :func:`sample_multiplicative_mms_perturbation`.  This module owns
the part that is the same for every PDE:

1. draw a smooth, dimensionless random field;
2. apply an epsilon-shifted multiplicative change;
3. choose ``alpha`` so the physical solution change is inside the range chosen
   by the user;
4. try another ``alpha`` and then another random field when a proposal fails;
5. run the plugin's Method of Manufactured Solutions (MMS) callback only for a
   proposal with the requested size.

There is deliberately no dependency on a basis, latent representation, grid,
or one particular equation family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, isfinite, pi, sqrt
from random import Random
from typing import Callable, Generic, Mapping, Sequence, TypeVar

from .sampling import RecoverableSamplingRejection


SolutionT = TypeVar("SolutionT")
RandomFieldT = TypeVar("RandomFieldT")
ManufacturedProblemT = TypeVar("ManufacturedProblemT")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class RelativeChangeRange:
    """The accepted dimensionless change between a parent and its child.

    The PDE plugin defines the actual physical norm.  For example, an NS
    plugin may use a quadrature-weighted relative L2 change of velocity.  The
    shared helper only enforces that the value returned by that metric lies in
    this range.
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum, "minimum relative change")
        maximum = _finite_float(self.maximum, "maximum relative change")
        if minimum <= 0.0:
            raise ValueError("minimum relative change must be positive")
        if maximum <= minimum:
            raise ValueError(
                "maximum relative change must be greater than the minimum"
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(self, value: float) -> bool:
        """Return whether one measured change is accepted."""

        measured = _finite_float(value, "measured relative change")
        return self.minimum <= measured <= self.maximum


@dataclass(frozen=True)
class MultiplicativePerturbationSettings:
    """User-facing choices for the common multiplicative MMS method.

    ``maximum_alpha`` is a final safety cap. A direction that would need a
    larger multiplier is discarded and replaced instead of producing an
    extreme manufactured problem.
    """

    target_relative_change: RelativeChangeRange
    epsilon_fraction: float = 1.0e-3
    max_alpha_attempts_per_random_field: int = 8
    max_random_field_attempts: int = 8
    maximum_alpha: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.target_relative_change, RelativeChangeRange):
            raise TypeError(
                "target_relative_change must be a RelativeChangeRange"
            )
        epsilon_fraction = _finite_float(
            self.epsilon_fraction,
            "epsilon fraction",
        )
        if epsilon_fraction <= 0.0:
            raise ValueError("epsilon fraction must be positive")
        object.__setattr__(self, "epsilon_fraction", epsilon_fraction)
        maximum_alpha = _finite_float(self.maximum_alpha, "maximum alpha")
        if maximum_alpha <= 0.0:
            raise ValueError("maximum alpha must be positive")
        object.__setattr__(self, "maximum_alpha", maximum_alpha)
        for value, name in (
            (self.max_alpha_attempts_per_random_field, "alpha attempts per random field"),
            (self.max_random_field_attempts, "random field attempts"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SmoothRandomFieldSettings:
    """Shape of the SDK's default smooth random field.

    ``length_scales`` uses the same ordered coordinates and physical units as
    the PDE solution, for example ``(L_x, L_y, L_t)`` for ``(x, y, t)``. A
    larger value produces a more slowly varying field along that coordinate.
    ``maximum_frequency_factor`` truncates rare spectral outliers that would
    make MMS source derivatives extreme. This default assumes an ordinary
    Euclidean coordinate chart; periodic, manifold, or topology-aware plugins
    should supply their own ``draw_random_field`` callback.
    """

    length_scales: tuple[float, ...]
    term_count: int = 32
    maximum_frequency_factor: float = 4.0

    def __post_init__(self) -> None:
        length_scales = tuple(
            _finite_float(value, "random-field length scale")
            for value in self.length_scales
        )
        if not length_scales:
            raise ValueError("a smooth random field needs at least one coordinate")
        if any(value <= 0.0 for value in length_scales):
            raise ValueError("random-field length scales must be positive")
        if isinstance(self.term_count, bool) or not isinstance(self.term_count, int):
            raise TypeError("random-field term count must be an integer")
        if self.term_count < 1:
            raise ValueError("random-field term count must be positive")
        maximum_frequency_factor = _finite_float(
            self.maximum_frequency_factor,
            "maximum random-field frequency factor",
        )
        if maximum_frequency_factor <= 0.0:
            raise ValueError("maximum random-field frequency factor must be positive")
        object.__setattr__(
            self, "maximum_frequency_factor", maximum_frequency_factor
        )
        object.__setattr__(self, "length_scales", length_scales)


@dataclass(frozen=True)
class SmoothRandomFourierField:
    """A continuous scalar random field with analytic derivatives.

    It uses a bounded Gaussian-like Fourier spectrum, so every draw is smooth
    and rare frequency outliers cannot create unbounded MMS derivatives. The
    field is dimensionless and has approximately unit RMS amplitude; the
    accepted physical change is controlled by ``alpha``, not by this
    approximate normalization.
    """

    length_scales: tuple[float, ...]
    frequencies: tuple[tuple[float, ...], ...]
    phases: tuple[float, ...]
    weights: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        length_scales = tuple(
            _finite_float(value, "random-field length scale")
            for value in self.length_scales
        )
        if not length_scales or any(value <= 0.0 for value in length_scales):
            raise ValueError("field length scales must be nonempty and positive")
        frequencies = tuple(
            tuple(
                _finite_float(value, "random-field frequency")
                for value in frequency
            )
            for frequency in self.frequencies
        )
        phases = tuple(
            _finite_float(value, "random-field phase") for value in self.phases
        )
        weights = tuple(
            _finite_float(value, "random-field weight") for value in self.weights
        )
        if not frequencies:
            raise ValueError("a smooth random field needs at least one term")
        if len(frequencies) != len(phases) or len(frequencies) != len(weights):
            raise ValueError("random-field frequencies, phases, and weights must align")
        if any(len(frequency) != len(length_scales) for frequency in frequencies):
            raise ValueError("every random-field frequency needs one value per coordinate")
        if not any(weight != 0.0 for weight in weights):
            raise ValueError("a smooth random field cannot have all-zero weights")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("random-field seed must be a nonnegative integer")
        object.__setattr__(self, "length_scales", length_scales)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def sample(
        cls,
        settings: SmoothRandomFieldSettings,
        *,
        seed: int,
    ) -> "SmoothRandomFourierField":
        """Draw one deterministic field from ``seed``."""

        if not isinstance(settings, SmoothRandomFieldSettings):
            raise TypeError("settings must be SmoothRandomFieldSettings")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("random-field seed must be a nonnegative integer")

        random = Random(seed)
        frequencies: list[tuple[float, ...]] = []
        phases: list[float] = []
        raw_weights: list[float] = []

        def bounded_frequency(scale: float) -> float:
            standard_frequency = random.gauss(0.0, 1.0)
            limit = settings.maximum_frequency_factor
            return max(-limit, min(limit, standard_frequency)) / scale

        for _index in range(settings.term_count):
            frequencies.append(
                tuple(
                    bounded_frequency(scale)
                    for scale in settings.length_scales
                )
            )
            phases.append(random.uniform(0.0, 2.0 * pi))
            raw_weights.append(random.gauss(0.0, 1.0))

        # For unrelated phases, spatial mean(delta**2) is approximately
        # sum(weight**2) / 2.  This normalization therefore gives unit RMS.
        rms = sqrt(sum(value * value for value in raw_weights) / 2.0)
        if rms == 0.0:  # Practically impossible, but keep the public API total.
            raw_weights[0] = sqrt(2.0)
            rms = 1.0
        weights = tuple(value / rms for value in raw_weights)
        return cls(
            length_scales=settings.length_scales,
            frequencies=tuple(frequencies),
            phases=tuple(phases),
            weights=weights,
            seed=seed,
        )

    @property
    def dimension(self) -> int:
        return len(self.length_scales)

    def __call__(self, coordinates: Sequence[float]) -> float:
        """Evaluate ``delta U`` at an ordered coordinate tuple."""

        return self.derivative(coordinates, (0,) * self.dimension)

    def derivative(
        self,
        coordinates: Sequence[float],
        orders: Sequence[int],
    ) -> float:
        """Evaluate any mixed partial derivative.

        ``orders=(2, 0, 1)`` means two derivatives in the first coordinate and
        one in the third.  These derivatives let a Generator derive MMS source
        terms without replacing the continuous field by a grid.
        """

        point = tuple(
            _finite_float(value, "random-field coordinate")
            for value in coordinates
        )
        derivative_orders = tuple(orders)
        if len(point) != self.dimension or len(derivative_orders) != self.dimension:
            raise ValueError(
                f"random field expects {self.dimension} coordinates and orders"
            )
        if any(
            isinstance(order, bool) or not isinstance(order, int) or order < 0
            for order in derivative_orders
        ):
            raise ValueError("derivative orders must be nonnegative integers")

        total_order = sum(derivative_orders)
        result = 0.0
        for frequency, phase, weight in zip(
            self.frequencies,
            self.phases,
            self.weights,
        ):
            theta = phase + sum(
                component * coordinate
                for component, coordinate in zip(frequency, point)
            )
            factor = weight
            for component, order in zip(frequency, derivative_orders):
                factor *= component**order
            result += factor * cos(theta + total_order * pi / 2.0)
        return result


def draw_smooth_random_field(
    settings: SmoothRandomFieldSettings,
    *,
    seed: int,
) -> SmoothRandomFourierField:
    """Draw the SDK's basis-independent default ``delta U``."""

    return SmoothRandomFourierField.sample(settings, seed=seed)


def epsilon_shifted_multiplicative_value(
    original: float,
    random_multiplier: float,
    *,
    alpha: float,
    epsilon_fraction: float,
    reference_scale: float,
) -> float:
    """Apply the agreed pointwise perturbation to one physical field value.

    The implemented formula is

    ``(U + epsilon * scale) * (1 + alpha * delta_U) - epsilon * scale``.

    ``delta_U`` is dimensionless.  Multiplying epsilon by ``reference_scale``
    gives the offset the same units as ``U``.  Subtracting that offset at the
    end makes ``alpha=0`` exactly recover the parent solution.
    """

    value = _finite_float(original, "original field value")
    multiplier = _finite_float(random_multiplier, "random multiplier")
    amplitude = _finite_float(alpha, "perturbation alpha")
    epsilon = _finite_float(epsilon_fraction, "epsilon fraction")
    scale = _finite_float(reference_scale, "field reference scale")
    if amplitude < 0.0:
        raise ValueError("perturbation alpha cannot be negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon fraction must be positive")
    if scale <= 0.0:
        raise ValueError("field reference scale must be positive")
    if amplitude == 0.0:
        return value
    offset = epsilon * scale
    # This expanded form avoids cancellation of the offset while remaining
    # algebraically identical to the formula in the docstring.
    return value + amplitude * (value + offset) * multiplier


@dataclass(frozen=True)
class PerturbationValidation:
    """A PDE plugin's final decision after it has manufactured the new Case."""

    accepted: bool
    reason: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("perturbation validation accepted must be a boolean")
        if not isinstance(self.reason, str):
            raise TypeError("perturbation validation reason must be text")
        if not self.accepted and not self.reason.strip():
            raise ValueError("a rejected perturbation needs a reason")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("perturbation validation metric names cannot be empty")
            metrics[name] = _finite_float(
                value,
                f"perturbation validation metric {name}",
            )
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class MultiplicativeMMSPerturbationResult(
    Generic[SolutionT, ManufacturedProblemT]
):
    """The accepted child solution and the problem manufactured from it."""

    solution: SolutionT
    manufactured_problem: ManufacturedProblemT
    alpha: float
    relative_change: float
    random_field_seed: int
    random_field_attempt: int
    alpha_attempt: int
    validation: PerturbationValidation


def sample_multiplicative_mms_perturbation(
    parent_solution: SolutionT,
    *,
    settings: MultiplicativePerturbationSettings,
    seed: int,
    draw_random_field: Callable[[int], RandomFieldT],
    compose_candidate: Callable[
        [SolutionT, RandomFieldT, float, float],
        SolutionT,
    ],
    measure_relative_change: Callable[[SolutionT, SolutionT], float],
    manufacture_problem: Callable[[SolutionT], ManufacturedProblemT],
    validate: Callable[
        [SolutionT, SolutionT, ManufacturedProblemT],
        PerturbationValidation,
    ],
    make_admissible: Callable[[SolutionT], SolutionT] | None = None,
) -> MultiplicativeMMSPerturbationResult[SolutionT, ManufacturedProblemT]:
    """Create one perturbed child using the common multiplicative MMS method.

    ``compose_candidate`` applies the pointwise formula (normally by calling
    :func:`epsilon_shifted_multiplicative_value`) in the plugin's own
    continuous-solution representation.  ``make_admissible`` can then enforce
    a PDE-specific hard constraint such as incompressibility or positivity.

    The helper first measures the change at a bounded probe alpha. Because the
    agreed formula is linear in alpha, this gives a useful alpha interval
    immediately. Every actual proposal is still remeasured after any PDE
    projection and alpha is capped by ``settings.maximum_alpha``. A proposal
    outside the user range, or one rejected by the required final ``validate``
    callback, tries another alpha. That validator sees the manufactured
    forcing/source and must enforce the PDE's boundary, initial, structural,
    and question-change limits. A scientific callback that cannot construct
    the current proposal may raise ``RecoverableSamplingRejection``; the
    helper records it and tries another alpha on the same field. Exhausting
    the alpha attempts draws a new
    random field; exhausting all fields raises the recoverable rejection
    already understood by the shared paired Case sampler.
    """

    if not isinstance(settings, MultiplicativePerturbationSettings):
        raise TypeError("settings must be MultiplicativePerturbationSettings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("perturbation seed must be a nonnegative integer")
    if not callable(draw_random_field):
        raise TypeError("draw_random_field must be callable")
    if not callable(compose_candidate):
        raise TypeError("compose_candidate must be callable")
    if not callable(measure_relative_change):
        raise TypeError("measure_relative_change must be callable")
    if not callable(manufacture_problem):
        raise TypeError("manufacture_problem must be callable")
    if make_admissible is not None and not callable(make_admissible):
        raise TypeError("make_admissible must be callable")
    if not callable(validate):
        raise TypeError("validate must be callable")

    admit = make_admissible or (lambda solution: solution)
    target = settings.target_relative_change
    random = Random(seed)
    rejection_reasons: list[dict[str, object]] = []

    def record_rejection(**row: object) -> None:
        rejection_reasons.append(dict(row))
        # Failure details are useful, but one unlucky root should not create an
        # unbounded result file.
        if len(rejection_reasons) > 12:
            del rejection_reasons[0]

    def measured_change(candidate: SolutionT) -> float | None:
        raw_value = measure_relative_change(parent_solution, candidate)
        if isinstance(raw_value, bool):
            raise TypeError("measured relative change must be a real number")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError("measured relative change must be a real number") from exc
        # A non-finite or negative value normally means this scientific draw is
        # unusable, not that the whole Run is malformed. Let rejection sampling
        # try another alpha/direction. A non-numeric callback result above
        # remains a programming error and fails immediately.
        if not isfinite(value) or value < 0.0:
            return None
        return value

    def allowed_alpha_interval(slope: float) -> tuple[float, float] | None:
        if not isfinite(slope) or slope <= 0.0:
            return None
        lower = target.minimum / slope
        upper = min(target.maximum / slope, settings.maximum_alpha)
        if (
            not isfinite(lower)
            or not isfinite(upper)
            or lower > upper
        ):
            return None
        return lower, upper

    for field_attempt in range(1, settings.max_random_field_attempts + 1):
        field_seed = random.randrange(0, 2**63)
        random_field = draw_random_field(field_seed)

        probe_alpha = min(1.0, target.maximum, settings.maximum_alpha)
        probe = admit(
            compose_candidate(
                parent_solution,
                random_field,
                probe_alpha,
                settings.epsilon_fraction,
            )
        )
        probe_change = measured_change(probe)
        if probe_change is None or probe_change <= 0.0:
            record_rejection(
                random_field_attempt=field_attempt,
                reason=(
                    "random field produced an invalid change"
                    if probe_change is None
                    else "random field produced no measurable change"
                ),
            )
            continue

        alpha_interval = allowed_alpha_interval(probe_change / probe_alpha)
        if alpha_interval is None:
            record_rejection(
                random_field_attempt=field_attempt,
                reason="random field requires an invalid or excessive alpha",
            )
            continue
        alpha_low, alpha_high = alpha_interval

        for alpha_attempt in range(1, settings.max_alpha_attempts_per_random_field + 1):
            alpha = random.uniform(alpha_low, alpha_high)
            candidate = admit(
                compose_candidate(
                    parent_solution,
                    random_field,
                    alpha,
                    settings.epsilon_fraction,
                )
            )
            relative_change = measured_change(candidate)
            if relative_change is None:
                record_rejection(
                    random_field_attempt=field_attempt,
                    alpha_attempt=alpha_attempt,
                    reason="candidate produced an invalid relative change",
                )
                continue
            if not target.contains(relative_change):
                record_rejection(
                    random_field_attempt=field_attempt,
                    alpha_attempt=alpha_attempt,
                    reason="measured change was outside the requested range",
                    measured_relative_change=relative_change,
                )
                # A nonlinear projection may change the alpha-to-distance
                # relationship.  Re-estimate it from the latest proposal before
                # the next rejection-sampling attempt.
                if relative_change > 0.0 and alpha > 0.0:
                    local_slope = relative_change / alpha
                    alpha_interval = allowed_alpha_interval(local_slope)
                    if alpha_interval is None:
                        record_rejection(
                            random_field_attempt=field_attempt,
                            alpha_attempt=alpha_attempt,
                            reason="updated alpha interval is invalid or excessive",
                        )
                        break
                    alpha_low, alpha_high = alpha_interval
                continue

            try:
                manufactured_problem = manufacture_problem(candidate)
                validation = validate(
                    parent_solution,
                    candidate,
                    manufactured_problem,
                )
            except RecoverableSamplingRejection as error:
                record_rejection(
                    random_field_attempt=field_attempt,
                    alpha_attempt=alpha_attempt,
                    reason=str(error),
                    measured_relative_change=relative_change,
                    callback_rejection_audit=error.audit,
                )
                continue
            if not isinstance(validation, PerturbationValidation):
                raise TypeError("validate must return PerturbationValidation")
            if not validation.accepted:
                record_rejection(
                    random_field_attempt=field_attempt,
                    alpha_attempt=alpha_attempt,
                    reason=validation.reason,
                    measured_relative_change=relative_change,
                )
                continue

            return MultiplicativeMMSPerturbationResult(
                solution=candidate,
                manufactured_problem=manufactured_problem,
                alpha=alpha,
                relative_change=relative_change,
                random_field_seed=field_seed,
                random_field_attempt=field_attempt,
                alpha_attempt=alpha_attempt,
                validation=validation,
            )

    raise RecoverableSamplingRejection(
        "could not create a valid multiplicative perturbation in the requested range",
        audit={
            "method": "epsilon_shifted_multiplicative_mms",
            "target_relative_change": {
                "minimum": target.minimum,
                "maximum": target.maximum,
            },
            "epsilon_fraction": settings.epsilon_fraction,
            "maximum_alpha": settings.maximum_alpha,
            "random_fields_tried": settings.max_random_field_attempts,
            "alpha_attempts_per_random_field": settings.max_alpha_attempts_per_random_field,
            "recent_rejections": rejection_reasons,
        },
    )


__all__ = [
    "MultiplicativeMMSPerturbationResult",
    "MultiplicativePerturbationSettings",
    "PerturbationValidation",
    "RelativeChangeRange",
    "SmoothRandomFieldSettings",
    "SmoothRandomFourierField",
    "draw_smooth_random_field",
    "epsilon_shifted_multiplicative_value",
    "sample_multiplicative_mms_perturbation",
]
