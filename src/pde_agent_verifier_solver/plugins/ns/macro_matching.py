"""Velocity-field matching for the shared continuous-solution loop."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean

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


NS_VELOCITY_FIELDS = ("velocity_x", "velocity_y")
NS_VELOCITY_MATCH_CONTRACT = "ns_velocity_field_match_v1"


@dataclass(frozen=True)
class NSVelocityMatcher:
    """Match the two physical velocity components at controller-owned points."""

    maximum_relative_l2: float | None = None

    info = ProgramInfo(
        "ns-velocity-field-matcher",
        "Navier--Stokes velocity-field match",
        "Match both velocity components against the controller-private "
        "continuous manufactured answer.",
    )
    contract_id = NS_VELOCITY_MATCH_CONTRACT
    fields = NS_VELOCITY_FIELDS

    def __post_init__(self) -> None:
        if self.maximum_relative_l2 is not None and (
            not isfinite(float(self.maximum_relative_l2))
            or float(self.maximum_relative_l2) < 0.0
        ):
            raise ValueError("NS velocity-match tolerance must be nonnegative")

    def prepare_target(
        self,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> MacroTarget:
        self._validate_view(truth, query, label="target")
        return MacroTarget(
            case_id=case.case_id,
            contract_id=self.contract_id,
            query=query,
            values=truth,
        )

    def match(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> MacroMatchResult:
        if target.case_id != case.case_id:
            raise ValueError("NS velocity target belongs to another case")
        if target.contract_id != self.contract_id:
            raise ValueError("NS velocity target uses another contract")
        self._validate_view(target.values, target.query, label="target")
        self._validate_view(prediction, target.query, label="prediction")
        if prediction.kind is QueryKind.POINTS:
            if prediction.points != target.values.points:
                raise ValueError("NS velocity prediction points differ from target")
        elif dict(prediction.axes) != dict(target.values.axes):
            raise ValueError("NS velocity prediction axes differ from target")
        if prediction.shape != target.values.shape:
            raise ValueError("NS velocity prediction shape differs from target")

        component_errors = {
            field: _relative_l2(
                prediction.fields[field], target.values.fields[field]
            )
            for field in self.fields
        }
        primary = fmean(component_errors.values())
        passed = (
            None
            if self.maximum_relative_l2 is None
            else primary <= float(self.maximum_relative_l2)
        )
        metrics = {
            "macro_match_relative_l2": primary,
            "macro_relative_l2": primary,
            "ns_velocity_field_match_v1_relative_l2": primary,
            **{
                f"{name}_relative_l2": value
                for name, value in component_errors.items()
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
            failures=(
                ()
                if passed is not False
                else ("aggregate NS velocity relative L2 exceeds tolerance",)
            ),
        )

    def result_card(self, result: MacroMatchResult) -> PanelCard:
        if result.contract_id != self.contract_id:
            raise ValueError("cannot render another NS velocity contract")
        return PanelCard(
            title="Navier--Stokes velocity match",
            summary=(
                "Candidate velocity matched against the controller-private "
                "continuous manufactured answer."
            ),
            values={
                "velocity_relative_l2": result.primary_error,
                "velocity_x_relative_l2": result.metrics[
                    "velocity_x_relative_l2"
                ],
                "velocity_y_relative_l2": result.metrics[
                    "velocity_y_relative_l2"
                ],
                "gate": (
                    "not configured" if result.passed is None else result.passed
                ),
            },
        )

    def _validate_view(
        self,
        view: AnswerView,
        query: AnswerQuery,
        *,
        label: str,
    ) -> None:
        if view.kind is not query.kind:
            raise ValueError(f"NS velocity {label} uses the wrong query kind")
        if set(query.fields) != set(self.fields):
            raise ValueError("NS velocity matching requires both components")
        if set(view.fields) != set(self.fields):
            raise ValueError(f"NS velocity {label} fields changed")
        if query.kind is QueryKind.POINTS:
            if view.points != query.points:
                raise ValueError(f"NS velocity {label} must echo query points")
            expected_shape = (len(query.points),)
        else:
            expected_shape = tuple(
                int(query.resolution[name]) for name in ("t", "x", "y")
            )
        if view.shape != expected_shape:
            raise ValueError(f"NS velocity {label} shape differs from query")
        sample_count = 1
        for extent in expected_shape:
            sample_count *= extent
        for field in self.fields:
            values = view.fields[field]
            if len(values) != sample_count:
                raise ValueError(f"NS velocity {label} sample count changed")
            if any(not isfinite(float(value)) for value in values):
                raise ValueError(f"NS velocity {label} is non-finite")


def _relative_l2(actual, expected) -> float:
    if len(actual) != len(expected):
        raise ValueError("NS velocity values must align")
    numerator = sum(
        (float(observed) - float(truth)) ** 2
        for observed, truth in zip(actual, expected)
    )
    denominator = sum(float(truth) ** 2 for truth in expected)
    return sqrt(numerator / max(denominator, 1.0e-30))


__all__ = [
    "NS_VELOCITY_FIELDS",
    "NS_VELOCITY_MATCH_CONTRACT",
    "NSVelocityMatcher",
]
