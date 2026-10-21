"""Physical perturbation geometry, derivatives and independent MMS checks."""

from math import exp, pi, sqrt

import numpy as np
import pytest

from pde_agent_verifier_solver.sdk.physical_perturbation import (
    PerturbationGeometry, PhysicalPerturbationPolicy, PositiveKineticMultiplier,
    ScalarJet, manufacture_kinetic_source_1d1v, sample_positive_kinetic_mms,
    weighted_relative_field_change,
)
from pde_agent_verifier_solver.sdk.perturbation import (
    MultiplicativePerturbationSettings, PerturbationValidation, RelativeChangeRange,
    SmoothRandomFourierField,
)


def _maxwellian(point):
    _t, _x, velocity = point
    value = exp(-0.5 * velocity ** 2) / sqrt(2 * pi)
    return ScalarJet(value, (0.0, 0.0, -velocity * value), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, (velocity ** 2 - 1) * value)))


def test_periodic_geometry_values_and_all_derivatives_match():
    geometry = PerturbationGeometry(("t", "x", "v"), (0.4, 0.2, 1.0), (None, 2 * pi, None))
    field = geometry.draw(42)
    point = (0.13, 0.7, -0.8)
    periodic = (point[0], point[1] + 2 * pi, point[2])
    assert field(point) == pytest.approx(field(periodic), abs=1e-12)
    for order in ((1, 0, 0), (0, 1, 0), (0, 2, 0), (0, 1, 1)):
        assert field.derivative(point, order) == pytest.approx(field.derivative(periodic, order), abs=1e-10)
    assert geometry.draw(42) == field
    assert geometry.draw(43) != field


def test_field_norm_respects_units_quadrature_and_components():
    parent = {"rho": [[1.0], [2.0]], "momentum": [[2.0, 3.0], [4.0, 5.0]]}
    child = {name: np.asarray(values) * 1.08 for name, values in parent.items()}
    settings = {"quadrature_weights": (0.2, 0.8), "field_scales": {"rho": 2.0, "momentum": 4.0}}
    assert weighted_relative_field_change(parent, child, **settings) == pytest.approx(0.08)
    converted_parent = {name: np.asarray(values) * 1000 for name, values in parent.items()}
    converted_child = {name: values * 1000 for name, values in child.items()}
    assert weighted_relative_field_change(converted_parent, converted_child, quadrature_weights=(2, 8), field_scales={"rho": 2000, "momentum": 4000}) == pytest.approx(0.08)
    with pytest.raises(ValueError, match="identical"):
        weighted_relative_field_change(parent, child, quadrature_weights=(1, 1), field_scales={"rho": 2})
    with pytest.raises(ValueError, match="positive"):
        weighted_relative_field_change(parent, child, quadrature_weights=(0, 1), field_scales=settings["field_scales"])


def test_physical_policy_rejects_missing_physics_and_source_blowup():
    policy = PhysicalPerturbationPolicy(RelativeChangeRange(0.05, 0.1), 0.3, {"mms_residual": 1e-5, "boundary_residual": 1e-8, "poisson_residual": 1e-5, "positivity_deficit": 0.0}, "a" * 64)
    audit = {"mms_residual": 1e-7, "boundary_residual": 0.0, "poisson_residual": 1e-7, "positivity_deficit": 0.0}
    def validate(**changes):
        values = {"solution_change": 0.07, "source_change": 0.1, "residuals": audit, "metric_definition_sha256": "a" * 64}
        return policy.validate(**(values | changes))
    assert validate().accepted
    assert len(policy.sha256) == 64
    assert not validate(source_change=0.31).accepted
    assert not validate(residuals={"mms_residual": 0.0}).accepted
    assert not validate(residuals=audit | {"positivity_deficit": 1e-10}).accepted
    assert not validate(solution_change=float("nan")).accepted
    with pytest.raises(ValueError, match="definition changed"):
        validate(metric_definition_sha256="b" * 64)


def test_positive_multiplier_analytic_jet_matches_finite_differences():
    field = PerturbationGeometry(("t", "x", "v"), (0.5, 0.5, 1.0), (None, 2 * pi, None), term_count=5).draw(12)
    child = PositiveKineticMultiplier(_maxwellian, field, 0.15)
    point = np.asarray((0.3, 0.7, -0.4))
    jet = child(point)
    assert jet.value > 0
    step = 2e-4
    basis = np.eye(3) * step
    for i in range(3):
        first = (child(point + basis[i]).value - child(point - basis[i]).value) / (2 * step)
        second = (child(point + basis[i]).value - 2 * jet.value + child(point - basis[i]).value) / step ** 2
        assert jet.gradient[i] == pytest.approx(first, abs=2e-7)
        assert jet.hessian[i][i] == pytest.approx(second, abs=3e-6)
        for j in range(i + 1, 3):
            mixed = (child(point + basis[i] + basis[j]).value - child(point + basis[i] - basis[j]).value - child(point - basis[i] + basis[j]).value + child(point - basis[i] - basis[j]).value) / (4 * step ** 2)
            assert jet.hessian[i][j] == pytest.approx(mixed, abs=3e-6)
    periodic = child(point + np.asarray((0, 2 * pi, 0)))
    assert periodic.value == pytest.approx(jet.value)
    assert periodic.gradient == pytest.approx(jet.gradient)


def test_kinetic_mms_source_matches_independent_finite_difference_operator():
    field = SmoothRandomFourierField((1.0, 1.0, 1.0), ((1.2, 2.0, 0.3),), (0.2,), (1.0,), 2)
    child = PositiveKineticMultiplier(_maxwellian, field, 0.1)
    point = np.asarray((0.2, 0.7, -0.4))
    # Prescribed E verifies the source operator only, not self-consistent Poisson.
    electric_field, collision = 0.13, 0.4
    source = manufacture_kinetic_source_1d1v(child(point), velocity=point[2], electric_field=electric_field, collision_frequency=collision)
    step = 2e-4
    basis = np.eye(3) * step
    value = child(point).value
    derivatives = [(child(point + shift).value - child(point - shift).value) / (2 * step) for shift in basis]
    dvv = (child(point + basis[2]).value - 2 * value + child(point - basis[2]).value) / step ** 2
    independently_computed = derivatives[0] + point[2] * derivatives[1] + electric_field * derivatives[2] - collision * (value + point[2] * derivatives[2] + dvv)
    assert source == pytest.approx(independently_computed, abs=1e-7)
    assert manufacture_kinetic_source_1d1v(_maxwellian(point), velocity=point[2], electric_field=0.0, collision_frequency=collision) == pytest.approx(0, abs=1e-15)


def test_positive_kinetic_mms_calibrates_physical_amplitude_and_calls_validator():
    points = [(0.2, float(x), float(v)) for x in np.linspace(0, 2 * pi, 16, endpoint=False) for v in np.linspace(-4, 4, 17)]
    parent_values = {"f": [_maxwellian(point).value for point in points]}
    calls = []
    def metric(_parent, child):
        return weighted_relative_field_change(parent_values, {"f": [child(point).value for point in points]}, quadrature_weights=[1.0] * len(points), field_scales={"f": 1.0})
    def manufacture(child):
        calls.append("manufacture")
        return {"child": child}
    def validate(parent, child, problem):
        calls.append("validate")
        assert problem["child"] is child
        return PerturbationValidation(True, metrics={"test_callback": 1.0})
    result = sample_positive_kinetic_mms(_maxwellian, geometry=PerturbationGeometry(("t", "x", "v"), (1.0, 0.7, 1.0), (None, 2 * pi, None), term_count=3), settings=MultiplicativePerturbationSettings(RelativeChangeRange(0.05, 0.1)), seed=8, measure_relative_change=metric, manufacture_problem=manufacture, validate=validate)
    assert 0.05 <= result.relative_change <= 0.1
    assert calls == ["manufacture", "validate"]
    assert all(result.solution(point).value > 0 for point in points)


def test_multiplier_rejects_unbounded_amplitude_and_negative_density():
    field = SmoothRandomFourierField((1.0, 1.0, 1.0), ((1.0, 1.0, 1.0),), (0.0,), (1.0,), 0)
    with pytest.raises(ValueError, match="global multiplier bound"):
        PositiveKineticMultiplier(_maxwellian, field, 9.0)
    invalid = lambda _: ScalarJet(-1.0, (0.0, 0.0, 0.0), ((0.0,) * 3,) * 3)
    with pytest.raises(ValueError, match="negative"):
        PositiveKineticMultiplier(invalid, field, 0.1)((0, 0, 0))
