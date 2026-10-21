"""Method-neutral matching for private 1D1V VP macroscopic truth."""

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

from .problem import VP1D_MACRO_FIELDS
from pde_agent_verifier_solver.plugins.plasma_transport import PlasmaTransportMacroMatcher


class VP1DTransportMacroMatcher(PlasmaTransportMacroMatcher):
    """Convert each own raw M0--M3 answer to central transport quantities."""

    dimension = 1
    fields = VP1D_MACRO_FIELDS
    info = ProgramInfo(
        "vp-1d1v-transport-macro-matcher-v2",
        "1D1V VP macro transport v2",
        "Score density, field, particle flux, central pressure and central heat "
        "flux with the same frozen objective as both VPFP regimes.",
    )


@dataclass(frozen=True)
class VP1DMacroMatcher:
    """Compare ``M0..M3,E`` without inspecting a candidate's method."""

    maximum_relative_l2: float | None = None

    info = ProgramInfo(
        "vp-1d1v-macro-matcher",
        "1D1V VP macroscopic match",
        "Compare four velocity moments and the self-consistent electric field "
        "against controller-private continuous truth.",
    )
    contract_id = "vp_1d1v_m0_m3_e_match_v1"
    fields = VP1D_MACRO_FIELDS

    def __post_init__(self) -> None:
        if self.maximum_relative_l2 is not None and (
            not isfinite(float(self.maximum_relative_l2))
            or float(self.maximum_relative_l2) < 0.0
        ):
            raise ValueError("VP1D macro tolerance must be finite and nonnegative")

    def prepare_target(
        self,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> MacroTarget:
        if set(query.fields) != set(self.fields):
            raise ValueError("VP1D macro matching requires M0, M1, M2, M3, and E")
        target = MacroTarget(
            case_id=case.case_id,
            contract_id=self.contract_id,
            query=query,
            values=truth,
        )
        self._validate_coordinates(target.values, query)
        return target

    def match(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> MacroMatchResult:
        if case.case_id != target.case_id:
            raise ValueError("VP1D macro target belongs to another case")
        if target.contract_id != self.contract_id:
            raise ValueError("VP1D macro target uses another contract")
        self._validate_view(prediction, target)
        component_errors = {
            name: relative_l2(prediction.fields[name], target.values.fields[name])
            for name in self.fields
        }
        primary = fmean(component_errors.values())
        passed = (
            None
            if self.maximum_relative_l2 is None
            else primary <= float(self.maximum_relative_l2)
        )
        failures = (
            ()
            if passed is not False
            else ("aggregate VP1D macro relative L2 exceeds the tolerance",)
        )
        metrics = {
            "macro_match_relative_l2": primary,
            "macro_relative_l2": primary,
            "vp_1d1v_macro_match_v1_relative_l2": primary,
            **{
                f"{name}_relative_l2": error
                for name, error in component_errors.items()
            },
        }
        if self.maximum_relative_l2 is not None:
            metrics["macro_match_tolerance"] = float(self.maximum_relative_l2)
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
            raise ValueError("cannot render a result from another macro contract")
        return PanelCard(
            title="1D1V VP macro match",
            summary=(
                "Candidate M0--M3 and E matched against controller-private "
                "continuous truth; the candidate method is unrestricted."
            ),
            values={
                "relative_l2": result.primary_error,
                **{
                    name: result.metrics[f"{name}_relative_l2"]
                    for name in self.fields
                },
                "gate": "not configured" if result.passed is None else result.passed,
            },
        )

    def _validate_view(self, prediction: AnswerView, target: MacroTarget) -> None:
        if set(target.values.fields) != set(self.fields):
            raise ValueError("VP1D target fields changed")
        self._validate_coordinates(target.values, target.query)
        if prediction.kind is not target.values.kind:
            raise ValueError("VP1D prediction uses the wrong query kind")
        if set(prediction.fields) != set(self.fields):
            raise ValueError("VP1D prediction fields changed")
        if prediction.shape != target.values.shape:
            raise ValueError("VP1D prediction shape differs from truth")
        self._validate_coordinates(prediction, target.query)
        for name in self.fields:
            values = prediction.fields[name]
            if len(values) != len(target.values.fields[name]):
                raise ValueError(f"VP1D prediction sample count changed for {name}")
            if any(not isfinite(float(value)) for value in values):
                raise ValueError(f"VP1D prediction {name} is non-finite")

    @staticmethod
    def _validate_coordinates(view: AnswerView, query: AnswerQuery) -> None:
        if view.kind is not query.kind:
            raise ValueError("VP1D answer uses the wrong query kind")
        if query.kind is QueryKind.POINTS:
            if view.points != query.points or view.shape != (len(query.points),):
                raise ValueError("VP1D point coordinates differ from the query")
            return
        if set(view.axes) != set(query.resolution):
            raise ValueError("VP1D grid axes differ from the query")
        for name, count in query.resolution.items():
            axis = tuple(view.axes[name])
            if len(axis) != int(count):
                raise ValueError(f"VP1D grid axis {name!r} has the wrong size")
            if any(not isfinite(float(value)) for value in axis):
                raise ValueError(f"VP1D grid axis {name!r} is non-finite")
            if any(float(right) <= float(left) for left, right in zip(axis, axis[1:])):
                raise ValueError(f"VP1D grid axis {name!r} is not ordered")
        expected_shape = tuple(len(view.axes[name]) for name in view.axes)
        if view.shape != expected_shape:
            raise ValueError("VP1D grid shape differs from its axes")


def relative_l2(actual: Sequence[float], expected: Sequence[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError("prediction and truth have different sample counts")
    squared_error = sum(
        (float(observed) - float(reference)) ** 2
        for observed, reference in zip(actual, expected)
    )
    squared_reference = sum(float(value) ** 2 for value in expected)
    return sqrt(squared_error / max(squared_reference, 1.0e-30))


__all__ = ["VP1DMacroMatcher", "VP1DTransportMacroMatcher", "relative_l2"]
