"""Independent unified matching for stable GKW macroscopic quantities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean
from typing import Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    MacroMatchResult,
    MacroTarget,
    PanelCard,
    ProgramInfo,
    QueryKind,
)

from .contract import (
    GKW_COMPLEX_COMPONENT_AXIS,
    GKW_MACRO_FIELDS,
    GKW_MACRO_MATCH_CONTRACT,
    gkw_field_quadrature_weights,
    gkw_point_query_coordinates,
    validate_gkw_macro_query,
)


@dataclass(frozen=True)
class GKWMacroMatcher:
    """Match four complex GKW fields with the stable quadrature metric."""

    maximum_relative_l2: float | None = None

    info = ProgramInfo(
        "gkw-macro-matcher",
        "GKW macroscopic quantity match",
        "Match particle density, parallel momentum, kinetic energy and phi "
        "against controller-private contractions of the continuous ion state.",
    )
    contract_id = GKW_MACRO_MATCH_CONTRACT
    fields = GKW_MACRO_FIELDS

    def __post_init__(self) -> None:
        if self.maximum_relative_l2 is not None and (
            not isfinite(float(self.maximum_relative_l2))
            or float(self.maximum_relative_l2) < 0.0
        ):
            raise ValueError("GKW macro-match tolerance must be nonnegative")

    def prepare_target(
        self,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> MacroTarget:
        if set(query.fields) != set(self.fields):
            raise ValueError("GKW macro matching requires all registered fields")
        weights = self._sample_weights(case, query)
        target = MacroTarget(
            case_id=case.case_id,
            contract_id=self.contract_id,
            query=query,
            values=truth,
            sample_weights=weights,
        )
        self._validate_target_coordinates(case, target)
        return target

    def match(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> MacroMatchResult:
        self._validate_prediction(case, prediction, target)
        field_errors = {
            name: weighted_relative_l2(
                prediction.fields[name],
                target.values.fields[name],
                target.sample_weights,
            )
            for name in self.fields
        }
        primary = fmean(field_errors.values())
        passed = (
            None
            if self.maximum_relative_l2 is None
            else primary <= float(self.maximum_relative_l2)
        )
        failures = (
            ()
            if passed is not False
            else ("aggregate GKW macro relative L2 exceeds tolerance",)
        )
        metrics = {
            "macro_match_relative_l2": primary,
            "macro_relative_l2": primary,
            "gkw_macro_match_v1_relative_l2": primary,
            "legacy_fusion_gkw_aggregate_relative_l2": primary,
            **{
                f"{name}_relative_l2": value
                for name, value in field_errors.items()
            },
        }
        if self.maximum_relative_l2 is not None:
            metrics["macro_match_tolerance"] = float(
                self.maximum_relative_l2
            )
        return MacroMatchResult(
            case_id=case.case_id,
            contract_id=self.contract_id,
            primary_error=primary,
            metrics=metrics,
            passed=passed,
            failures=failures,
        )

    def result_card(self, result: MacroMatchResult) -> PanelCard:
        if result.contract_id != self.contract_id:
            raise ValueError("cannot render another GKW macro contract")
        return PanelCard(
            title="GKW macro match",
            summary=(
                "Candidate macros matched against controller-private "
                "velocity contractions; data observations are not targets."
            ),
            values={
                "gkw_macro_match_v1_relative_l2": result.primary_error,
                "particle_density_relative_l2": result.metrics[
                    "particle_density_relative_l2"
                ],
                "parallel_momentum_relative_l2": result.metrics[
                    "parallel_momentum_relative_l2"
                ],
                "kinetic_energy_relative_l2": result.metrics[
                    "kinetic_energy_relative_l2"
                ],
                "phi_relative_l2": result.metrics["phi_relative_l2"],
                "gate": (
                    "not configured" if result.passed is None else result.passed
                ),
            },
        )

    def _sample_weights(
        self,
        case: Case,
        query: AnswerQuery,
    ) -> tuple[float, ...]:
        import numpy as np

        if query.kind is QueryKind.GRID:
            times, indices = validate_gkw_macro_query(case, query)
            field = gkw_field_quadrature_weights(case, indices)
            repeated = np.broadcast_to(
                field[None, :, None],
                (len(times), len(indices), 2),
            )
        else:
            _times, indices = gkw_point_query_coordinates(
                case, query, require_native_panel=False
            )
            field = gkw_field_quadrature_weights(case, indices)
            repeated = np.broadcast_to(field[:, None], (len(indices), 2))
        return tuple(float(value) for value in repeated.reshape(-1))

    def _validate_target_coordinates(
        self,
        case: Case,
        target: MacroTarget,
    ) -> None:
        if set(target.values.fields) != set(self.fields):
            raise ValueError("GKW macro target fields changed")
        if target.query.kind is QueryKind.POINTS:
            gkw_point_query_coordinates(
                case, target.query, require_native_panel=False
            )
            if target.values.points != target.query.points:
                raise ValueError("GKW point target must echo query points")
            if target.values.shape != (len(target.query.points), 2):
                raise ValueError("GKW point target shape must be (point, complex)")
            return
        times, indices = validate_gkw_macro_query(case, target.query)
        expected_axes = {
            "t": tuple(float(value) for value in times),
            "field_row": tuple(float(value) for value in range(len(indices))),
            "complex_component": GKW_COMPLEX_COMPONENT_AXIS,
        }
        if dict(target.values.axes) != expected_axes:
            raise ValueError("GKW grid target axes differ from the public panel")
        if target.values.shape != (len(times), len(indices), 2):
            raise ValueError("GKW grid target shape differs from the public panel")

    def _validate_prediction(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> None:
        if case.case_id != target.case_id:
            raise ValueError("GKW macro target belongs to another case")
        if target.contract_id != self.contract_id:
            raise ValueError("GKW macro target uses another contract")
        self._validate_target_coordinates(case, target)
        if prediction.kind is not target.values.kind:
            raise ValueError("GKW macro prediction uses the wrong query kind")
        if set(prediction.fields) != set(self.fields):
            raise ValueError("GKW macro prediction fields changed")
        if prediction.shape != target.values.shape:
            raise ValueError("GKW macro prediction and target shapes differ")
        if target.query.kind is QueryKind.GRID:
            if dict(prediction.axes) != dict(target.values.axes):
                raise ValueError("GKW macro prediction axes differ from target")
        elif prediction.points != target.values.points:
            raise ValueError("GKW macro prediction points differ from target")
        for name in self.fields:
            values = prediction.fields[name]
            if len(values) != len(target.values.fields[name]):
                raise ValueError(f"GKW macro sample count changed for {name}")
            if any(not isfinite(float(value)) for value in values):
                raise ValueError(f"GKW macro prediction {name} is non-finite")


def weighted_relative_l2(
    actual: Sequence[float],
    expected: Sequence[float],
    weights: Sequence[float],
) -> float:
    if len(actual) != len(expected) or len(actual) != len(weights):
        raise ValueError("GKW values and quadrature weights must align")
    numerator = sum(
        float(weight) * (float(observed) - float(truth)) ** 2
        for observed, truth, weight in zip(actual, expected, weights)
    )
    denominator = sum(
        float(weight) * float(truth) ** 2
        for truth, weight in zip(expected, weights)
    )
    return sqrt(numerator / max(denominator, 1.0e-30))
