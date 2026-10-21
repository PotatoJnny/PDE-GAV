"""Focused tests for the SDK's default multiplicative MMS perturbation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable

import pytest

from pde_agent_verifier_solver.engine.case_sampler import RoundCaseSampler
from pde_agent_verifier_solver.sdk import (
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    GenerationKind,
    GenerationPurpose,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    GeneratorState,
    MultiplicativePerturbationSettings,
    PairedSampleRequest,
    PerturbationValidation,
    PreparedDataset,
    RecoverableSamplingRejection,
    RelativeChangeRange,
    SampleSource,
    SamplingStrategy,
    SmoothRandomFieldSettings,
    SmoothRandomFourierField,
    draw_smooth_random_field,
    epsilon_shifted_multiplicative_value,
    sample_multiplicative_mms_perturbation,
)


@dataclass(frozen=True)
class _ScalarCandidate:
    value: float
    alpha: float
    direction: float


def _scalar_compose(
    parent: float,
    direction: float,
    alpha: float,
    epsilon_fraction: float,
) -> _ScalarCandidate:
    return _ScalarCandidate(
        value=epsilon_shifted_multiplicative_value(
            parent,
            direction,
            alpha=alpha,
            epsilon_fraction=epsilon_fraction,
            reference_scale=1.0,
        ),
        alpha=alpha,
        direction=direction,
    )


def _scalar_relative_change(parent: float, candidate: _ScalarCandidate) -> float:
    return abs(candidate.value - parent) / max(abs(parent), 1.0e-3)


def _accept(_parent, _candidate, _problem) -> PerturbationValidation:
    return PerturbationValidation(True, metrics={"mms_checked": 1.0})


def _settings(
    *, alpha_attempts: int = 3, field_attempts: int = 3
) -> MultiplicativePerturbationSettings:
    return MultiplicativePerturbationSettings(
        target_relative_change=RelativeChangeRange(0.05, 0.10),
        epsilon_fraction=1.0e-3,
        max_alpha_attempts_per_random_field=alpha_attempts,
        max_random_field_attempts=field_attempts,
    )


def test_final_mms_validator_is_required() -> None:
    with pytest.raises(TypeError, match="validate must be callable"):
        sample_multiplicative_mms_perturbation(
            2.0,
            settings=_settings(),
            seed=4,
            draw_random_field=lambda _seed: 1.0,
            compose_candidate=_scalar_compose,
            measure_relative_change=_scalar_relative_change,
            manufacture_problem=lambda candidate: candidate.value,
            validate=None,  # type: ignore[arg-type]
        )


def test_pointwise_formula_is_identity_at_zero_alpha() -> None:
    original = -3.25
    assert epsilon_shifted_multiplicative_value(
        original,
        18.0,
        alpha=0.0,
        epsilon_fraction=0.2,
        reference_scale=7.0,
    ) == original


def test_epsilon_offset_lets_a_zero_parent_value_move() -> None:
    changed = epsilon_shifted_multiplicative_value(
        0.0,
        2.0,
        alpha=0.5,
        epsilon_fraction=0.1,
        reference_scale=3.0,
    )
    assert changed == pytest.approx(0.3)


def test_public_random_fourier_field_constructor_validates_structure() -> None:
    field = SmoothRandomFourierField(
        length_scales=(0.5,),
        frequencies=((2.0,),),
        phases=(0.1,),
        weights=(1.0,),
        seed=3,
    )
    assert isfinite(field((0.2,)))

    with pytest.raises(ValueError):
        SmoothRandomFourierField(
            length_scales=(0.5,),
            frequencies=((2.0,),),
            phases=(),
            weights=(1.0,),
            seed=3,
        )
    with pytest.raises(ValueError):
        SmoothRandomFourierField(
            length_scales=(0.5, 0.7),
            frequencies=((2.0,),),
            phases=(0.1,),
            weights=(1.0,),
            seed=3,
        )
    with pytest.raises(ValueError):
        SmoothRandomFourierField(
            length_scales=(0.5,),
            frequencies=((2.0,),),
            phases=(0.1,),
            weights=(float("nan"),),
            seed=3,
        )


def test_smooth_random_field_is_seeded_and_has_analytic_derivatives() -> None:
    settings = SmoothRandomFieldSettings(length_scales=(0.4, 0.9), term_count=24)
    first = draw_smooth_random_field(settings, seed=71)
    replay = draw_smooth_random_field(settings, seed=71)
    other = draw_smooth_random_field(settings, seed=72)
    point = (0.31, -0.27)

    assert first == replay
    assert first.frequencies != other.frequencies
    assert isfinite(first(point))
    for frequency in first.frequencies:
        for component, scale in zip(frequency, settings.length_scales):
            assert abs(component * scale) <= settings.maximum_frequency_factor


    step = 1.0e-6
    finite_difference = (
        first((point[0] + step, point[1]))
        - first((point[0] - step, point[1]))
    ) / (2.0 * step)
    assert first.derivative(point, (1, 0)) == pytest.approx(
        finite_difference, rel=2.0e-5, abs=2.0e-6
    )

    second_step = 1.0e-4
    finite_second = (
        first((point[0], point[1] + second_step))
        - 2.0 * first(point)
        + first((point[0], point[1] - second_step))
    ) / second_step**2
    assert first.derivative(point, (0, 2)) == pytest.approx(
        finite_second, rel=2.0e-4, abs=2.0e-4
    )


@pytest.mark.parametrize("invalid_measurement", [float("nan"), float("inf"), -0.2])
def test_invalid_candidate_measurement_retries_alpha(
    invalid_measurement: float,
) -> None:
    measurement_calls = 0
    manufacture_calls = 0

    def measure(parent: float, candidate: _ScalarCandidate) -> float:
        nonlocal measurement_calls
        measurement_calls += 1
        if measurement_calls == 2:
            return invalid_measurement
        return _scalar_relative_change(parent, candidate)

    def manufacture(candidate: _ScalarCandidate) -> str:
        nonlocal manufacture_calls
        manufacture_calls += 1
        return f"forcing-for-{candidate.alpha}"

    result = sample_multiplicative_mms_perturbation(
        2.0,
        settings=_settings(alpha_attempts=2, field_attempts=1),
        seed=9,
        draw_random_field=lambda _seed: 1.0,
        compose_candidate=_scalar_compose,
        measure_relative_change=measure,
        manufacture_problem=manufacture,
        validate=_accept,
    )

    assert result.alpha_attempt == 2
    assert result.random_field_attempt == 1
    assert manufacture_calls == 1


def test_validation_rejection_retries_alpha_before_redrawing_field() -> None:
    validation_calls = 0

    def validate(_parent, _candidate, _problem) -> PerturbationValidation:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return PerturbationValidation(False, "first alpha failed physics")
        return PerturbationValidation(True, metrics={"physics_residual": 0.0})

    result = sample_multiplicative_mms_perturbation(
        2.0,
        settings=_settings(alpha_attempts=2, field_attempts=1),
        seed=12,
        draw_random_field=lambda _seed: 1.0,
        compose_candidate=_scalar_compose,
        measure_relative_change=_scalar_relative_change,
        manufacture_problem=lambda candidate: candidate.value,
        validate=validate,
    )

    assert result.alpha_attempt == 2
    assert result.random_field_attempt == 1
    assert validation_calls == 2


def test_recoverable_mms_rejection_retries_alpha_before_field_redraw() -> None:
    manufacture_calls = 0

    def manufacture(candidate: _ScalarCandidate) -> float:
        nonlocal manufacture_calls
        manufacture_calls += 1
        if manufacture_calls == 1:
            raise RecoverableSamplingRejection(
                "current MMS proposal is not admissible",
                audit={"stage": "manufacture_problem", "finite": True},
            )
        return candidate.value

    result = sample_multiplicative_mms_perturbation(
        2.0,
        settings=_settings(alpha_attempts=2, field_attempts=1),
        seed=18,
        draw_random_field=lambda _seed: 1.0,
        compose_candidate=_scalar_compose,
        measure_relative_change=_scalar_relative_change,
        manufacture_problem=manufacture,
        validate=_accept,
    )

    assert result.alpha_attempt == 2
    assert result.random_field_attempt == 1
    assert manufacture_calls == 2


def test_zero_effect_direction_redraws_the_random_field() -> None:
    drawn_seeds: list[int] = []

    def draw(field_seed: int) -> float:
        drawn_seeds.append(field_seed)
        return 0.0 if len(drawn_seeds) == 1 else 1.0

    result = sample_multiplicative_mms_perturbation(
        2.0,
        settings=_settings(alpha_attempts=1, field_attempts=2),
        seed=21,
        draw_random_field=draw,
        compose_candidate=_scalar_compose,
        measure_relative_change=_scalar_relative_change,
        manufacture_problem=lambda candidate: candidate.value,
        validate=_accept,
    )

    assert result.random_field_attempt == 2
    assert len(drawn_seeds) == 2
    assert drawn_seeds[0] != drawn_seeds[1]


def test_excessive_alpha_redraws_then_rejects_without_running_mms() -> None:
    settings = MultiplicativePerturbationSettings(
        target_relative_change=RelativeChangeRange(0.05, 0.10),
        epsilon_fraction=1.0e-3,
        max_alpha_attempts_per_random_field=2,
        max_random_field_attempts=2,
        maximum_alpha=0.01,
    )
    with pytest.raises(RecoverableSamplingRejection) as caught:
        sample_multiplicative_mms_perturbation(
            2.0,
            settings=settings,
            seed=27,
            draw_random_field=lambda _seed: 1.0,
            compose_candidate=_scalar_compose,
            measure_relative_change=_scalar_relative_change,
            manufacture_problem=lambda _candidate: pytest.fail(
                "MMS must not run for an excessive alpha"
            ),
            validate=_accept,
        )

    assert caught.value.audit["random_fields_tried"] == 2


@pytest.mark.parametrize("invalid_probe", [float("nan"), float("-inf"), -1.0])
def test_invalid_probe_exhaustion_is_recoverable(invalid_probe: float) -> None:
    with pytest.raises(RecoverableSamplingRejection) as caught:
        sample_multiplicative_mms_perturbation(
            2.0,
            settings=_settings(alpha_attempts=2, field_attempts=2),
            seed=34,
            draw_random_field=lambda field_seed: field_seed,
            compose_candidate=lambda parent, direction, alpha, epsilon: (
                _ScalarCandidate(parent, alpha, float(direction))
            ),
            measure_relative_change=lambda _parent, _candidate: invalid_probe,
            manufacture_problem=lambda _candidate: pytest.fail(
                "MMS must not run for an invalid measured change"
            ),
            validate=_accept,
        )

    audit = caught.value.audit
    assert audit["random_fields_tried"] == 2
    assert audit["alpha_attempts_per_random_field"] == 2
    assert len(audit["recent_rejections"]) == 2


def test_callbacks_run_in_scientific_order() -> None:
    events: list[tuple[str, float | None]] = []

    def draw(_seed: int) -> float:
        events.append(("draw", None))
        return 1.0

    def compose(parent, direction, alpha, epsilon):
        events.append(("compose", alpha))
        return _scalar_compose(parent, direction, alpha, epsilon)

    def make_admissible(candidate):
        events.append(("make_admissible", candidate.alpha))
        return candidate

    def measure(parent, candidate):
        events.append(("measure", candidate.alpha))
        return _scalar_relative_change(parent, candidate)

    def manufacture(candidate):
        events.append(("manufacture", candidate.alpha))
        return candidate.value

    def validate(_parent, candidate, _problem):
        events.append(("validate", candidate.alpha))
        return PerturbationValidation(True)

    result = sample_multiplicative_mms_perturbation(
        2.0,
        settings=_settings(alpha_attempts=1, field_attempts=1),
        seed=55,
        draw_random_field=draw,
        compose_candidate=compose,
        measure_relative_change=measure,
        manufacture_problem=manufacture,
        validate=validate,
        make_admissible=make_admissible,
    )

    assert [name for name, _value in events] == [
        "draw",
        "compose",
        "make_admissible",
        "measure",
        "compose",
        "make_admissible",
        "measure",
        "manufacture",
        "validate",
    ]
    assert events[1][1] == _settings().target_relative_change.maximum
    assert events[-1][1] == result.alpha


class _PoissonParent:
    @staticmethod
    def value(x: float) -> float:
        return x * (1.0 - x)


@dataclass(frozen=True)
class _PoissonChild:
    field: SmoothRandomFourierField
    alpha: float
    epsilon_fraction: float

    @staticmethod
    def _basis(x: float) -> float:
        return x * (1.0 - x)

    def value(self, x: float) -> float:
        basis = self._basis(x)
        multiplier = basis * self.field((x,))
        return epsilon_shifted_multiplicative_value(
            basis,
            multiplier,
            alpha=self.alpha,
            epsilon_fraction=self.epsilon_fraction,
            reference_scale=1.0,
        )

    def second_derivative(self, x: float) -> float:
        basis = self._basis(x)
        basis_prime = 1.0 - 2.0 * x
        basis_second = -2.0
        epsilon = self.epsilon_fraction
        q = (basis + epsilon) * basis
        q_prime = (2.0 * basis + epsilon) * basis_prime
        q_second = 2.0 * basis_prime**2 + (2.0 * basis + epsilon) * basis_second
        field = self.field((x,))
        field_prime = self.field.derivative((x,), (1,))
        field_second = self.field.derivative((x,), (2,))
        return basis_second + self.alpha * (
            q_second * field + 2.0 * q_prime * field_prime + q * field_second
        )


@dataclass(frozen=True)
class _PoissonProblem:
    left_boundary: float
    right_boundary: float
    forcing: Callable[[float], float]


def _poisson_relative_change(
    parent: _PoissonParent,
    candidate: _PoissonChild,
) -> float:
    points = tuple(index / 256.0 for index in range(257))
    numerator = sqrt(
        sum((candidate.value(x) - parent.value(x)) ** 2 for x in points)
    )
    denominator = sqrt(sum(parent.value(x) ** 2 for x in points))
    return numerator / denominator


def test_continuous_poisson_boundary_envelope_and_mms_end_to_end() -> None:
    parent = _PoissonParent()
    field_settings = SmoothRandomFieldSettings(
        length_scales=(0.55,), term_count=20
    )

    def compose(parent_solution, field, alpha, epsilon):
        del parent_solution
        return _PoissonChild(field, alpha, epsilon)

    def manufacture(candidate: _PoissonChild) -> _PoissonProblem:
        return _PoissonProblem(
            left_boundary=candidate.value(0.0),
            right_boundary=candidate.value(1.0),
            forcing=lambda x: -candidate.second_derivative(x),
        )

    def validate(_parent, candidate, problem) -> PerturbationValidation:
        probe_points = (0.2, 0.5, 0.8)
        boundary_error = max(
            abs(problem.left_boundary), abs(problem.right_boundary)
        )
        forcing_change = max(
            abs(problem.forcing(x) - 2.0) for x in probe_points
        )
        accepted = (
            boundary_error < 1.0e-14
            and forcing_change > 1.0e-8
            and all(isfinite(problem.forcing(x)) for x in probe_points)
        )
        return PerturbationValidation(
            accepted,
            "MMS forcing or boundary data failed the final check"
            if not accepted else "",
            metrics={
                "boundary_change": boundary_error,
                "forcing_change": forcing_change,
            },
        )

    result = sample_multiplicative_mms_perturbation(
        parent,
        settings=MultiplicativePerturbationSettings(
            target_relative_change=RelativeChangeRange(0.03, 0.06),
            epsilon_fraction=0.01,
            max_alpha_attempts_per_random_field=2,
            max_random_field_attempts=2,
        ),
        seed=88,
        draw_random_field=lambda field_seed: draw_smooth_random_field(
            field_settings, seed=field_seed
        ),
        compose_candidate=compose,
        measure_relative_change=_poisson_relative_change,
        manufacture_problem=manufacture,
        validate=validate,
    )

    assert 0.03 <= result.relative_change <= 0.06
    assert result.solution.value(0.0) == pytest.approx(0.0, abs=1.0e-14)
    assert result.solution.value(1.0) == pytest.approx(0.0, abs=1.0e-14)
    assert result.validation.metrics["forcing_change"] > 0.0

    step = 2.0e-4
    for x in (0.2, 0.5, 0.8):
        numerical_second = (
            result.solution.value(x + step)
            - 2.0 * result.solution.value(x)
            + result.solution.value(x - step)
        ) / step**2
        assert result.manufactured_problem.forcing(x) == pytest.approx(
            -numerical_second, rel=3.0e-4, abs=2.0e-3
        )


class _PipelineTruth:
    """Controller-private placeholder; sampling never materializes it."""


class _PipelineGenerator:
    """Small PDE plugin proving the helper crosses the canonical sampler ABI."""

    capabilities = GeneratorCapabilities(
        offline=False,
        online=True,
        perturbations=True,
        trainable=False,
    )

    def __init__(self) -> None:
        self.values: dict[str, float | _ScalarCandidate] = {}
        self.requests = []
        self.relative_changes: dict[str, float] = {}
        self.released: list[str] = []

    def sample(self, state, request, progress=None):
        del state, progress
        self.requests.append(request)
        if request.kind is GenerationKind.BASE:
            records = []
            for index in range(request.count):
                case_id = f"base-{request.seed}-{index}"
                self.values[case_id] = 2.0
                records.append(
                    CaseRecord(
                        case=Case(
                            case_id=case_id,
                            source=CaseSource.GENERATED,
                            visibility=request.visibility,
                            public_data={"equation": "scalar test problem"},
                            round_index=request.round_index,
                        ),
                        truth=_PipelineTruth(),
                    )
                )
            return tuple(records)

        if request.kind is not GenerationKind.PERTURBED:
            raise AssertionError(f"unexpected generation kind {request.kind}")
        if request.count != 1 or len(request.anchors) != 1:
            raise AssertionError("canonical perturbation requests one anchored child")

        anchor = request.anchors[0]
        parent = self.values[anchor.case_id]
        if not isinstance(parent, float):
            raise AssertionError("perturbation parent must be a clean scalar solution")
        result = sample_multiplicative_mms_perturbation(
            parent,
            settings=MultiplicativePerturbationSettings(
                target_relative_change=RelativeChangeRange(
                    float(request.settings["perturbation_relative_change_min"]),
                    float(request.settings["perturbation_relative_change_max"]),
                ),
                epsilon_fraction=float(
                    request.settings["perturbation_epsilon_fraction"]
                ),
                maximum_alpha=float(
                    request.settings["perturbation_maximum_alpha"]
                ),
            ),
            seed=request.seed,
            draw_random_field=lambda _field_seed: 1.0,
            compose_candidate=_scalar_compose,
            measure_relative_change=_scalar_relative_change,
            manufacture_problem=lambda candidate: {"forcing": candidate.value},
            validate=lambda _parent, _candidate, _problem: PerturbationValidation(
                True,
                metrics={"question_change": 1.0},
            ),
        )
        case_id = f"child-{request.seed}"
        self.values[case_id] = result.solution
        self.relative_changes[case_id] = result.relative_change
        return (
            CaseRecord(
                case=Case(
                    case_id=case_id,
                    source=CaseSource.PERTURBED,
                    visibility=request.visibility,
                    public_data={"equation": "manufactured scalar test problem"},
                    round_index=request.round_index,
                    parent_case_id=anchor.case_id,
                ),
                truth=_PipelineTruth(),
            ),
        )

    def release(self, state, case_ids):
        del state
        self.released.extend(case_ids)
        for case_id in case_ids:
            self.values.pop(case_id, None)


@pytest.mark.parametrize(
    ("visibility", "purpose"),
    [
        (CaseVisibility.DEVELOPMENT, GenerationPurpose.DEVELOPMENT),
        (CaseVisibility.HIDDEN, GenerationPurpose.PROMOTION),
    ],
)
def test_helper_is_compatible_with_canonical_paired_sampling(
    visibility: CaseVisibility,
    purpose: GenerationPurpose,
) -> None:
    generator = _PipelineGenerator()
    state = GeneratorState(
        checkpoint=GeneratorCheckpoint("pipeline-generator", payload={}),
        base={},
    )
    generation_settings = {
        "perturbation_relative_change_min": 0.05,
        "perturbation_relative_change_max": 0.10,
        "perturbation_epsilon_fraction": 0.001,
        "perturbation_maximum_alpha": 10.0,
    }
    sampler = RoundCaseSampler(
        agent_id="pipeline-compatibility-test",
        round_index=2,
        max_cases=2,
        strategy=SamplingStrategy.FIXED_RECIPE,
        dataset=PreparedDataset(
            dataset=DatasetInput("pipeline-dataset", DatasetKind.NUMERICAL),
            development_cases=(),
            hidden_cases=(),
        ),
        generator=generator,
        generator_state=state,
        generation_settings=generation_settings,
        visibility=visibility,
        purpose=purpose,
        seed=123,
    )

    batch = sampler.sample_paired_cases(
        PairedSampleRequest(SampleSource.BASE, count=1)
    )

    root = batch.roots[0]
    child = batch.children[0]
    assert child.parent_case_id == root.case_id
    assert child.source is CaseSource.PERTURBED
    assert root.visibility is visibility
    assert child.visibility is visibility
    assert batch.remaining == 0
    assert batch.attempts[-1].accepted
    assert generator.requests[-1].kind is GenerationKind.PERTURBED
    assert generator.requests[-1].purpose is purpose
    assert generator.requests[-1].settings == generation_settings
    assert generator.requests[-1].anchors == (root,)
    assert 0.05 <= generator.relative_changes[child.case_id] <= 0.10
    assert set(sampler.generated_case_ids) == {root.case_id, child.case_id}
