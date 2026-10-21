"""VPFP matching against exact full-velocity macroscopic targets."""

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
from pde_agent_verifier_solver.plugins.plasma_transport import (
    PlasmaTransportMacroMatcher,
    VPFP_TRANSPORT_FIELDS,
)


class VPFPTransportMacroMatcher(PlasmaTransportMacroMatcher):
    """Shared v2 transport score with the 2D2V centered-moment ABI."""

    dimension = 2
    fields = VPFP_TRANSPORT_FIELDS
    info = ProgramInfo(
        "vpfp-transport-macro-matcher-v2",
        "VPFP macro transport v2",
        "Score density, self-consistent field, particle flux, full central "
        "pressure and central heat flux using the shared frozen Plasma calibration.",
    )


VPFP_MACRO_FIELDS = (
    "rho",
    "momentum_x",
    "momentum_y",
    "kinetic_energy",
)


@dataclass(frozen=True)
class VPFPMacroMatcher:
    """Compare candidate moments with hidden full-velocity moment truth.

    maximum_relative_l2=None records a score without imposing a gate.
    Generator admission or a Task selection policy may opt into a threshold
    explicitly; the shared Engine never invents one.
    The v1 primary score weights rho, vector momentum, and kinetic energy
    equally.  The former four-scalar average remains a diagnostic metric.
    """

    maximum_relative_l2: float | None = None

    info = ProgramInfo(
        "vpfp-macro-matcher",
        "VPFP macroscopic quantity match",
        "Match density, vector momentum and kinetic energy against hidden "
        "full-velocity moments, independently of training observations.",
    )
    contract_id = "vpfp_full_velocity_macro_match_v1"
    fields = VPFP_MACRO_FIELDS

    def __post_init__(self) -> None:
        if self.maximum_relative_l2 is not None and (
            not isfinite(float(self.maximum_relative_l2))
            or self.maximum_relative_l2 < 0.0
        ):
            raise ValueError(
                "VPFP macro-match tolerance must be finite and nonnegative"
            )

    def prepare_target(
        self,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> MacroTarget:
        if set(query.fields) != set(self.fields):
            raise ValueError(
                "VPFP macro matching requires all registered macro fields"
            )
        target = MacroTarget(
            case_id=case.case_id,
            contract_id=self.contract_id,
            query=query,
            values=truth,
        )
        self._validate_target_coordinates(target)
        return target

    def match(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> MacroMatchResult:
        self._validate_prediction(case, prediction, target)
        expected = target.values.fields
        observed = prediction.fields
        component_errors = {
            name: relative_l2(observed[name], expected[name])
            for name in self.fields
        }
        momentum_error = relative_l2(
            tuple(observed["momentum_x"]) + tuple(observed["momentum_y"]),
            tuple(expected["momentum_x"]) + tuple(expected["momentum_y"]),
        )
        physical_errors = {
            "rho": component_errors["rho"],
            "momentum": momentum_error,
            "kinetic_energy": component_errors["kinetic_energy"],
        }
        primary = fmean(physical_errors.values())
        legacy_four_scalar = fmean(component_errors.values())
        passed = (
            None
            if self.maximum_relative_l2 is None
            else primary <= self.maximum_relative_l2
        )
        failures = (
            ()
            if passed is not False
            else (
                "aggregate macro relative L2 exceeds the configured tolerance",
            )
        )
        metrics = {
            "macro_match_relative_l2": primary,
            "macro_relative_l2": primary,
            "vpfp_macro_match_v1_relative_l2": primary,
            "legacy_four_scalar_macro_relative_l2": legacy_four_scalar,
            "rho_relative_l2": physical_errors["rho"],
            "momentum_relative_l2": physical_errors["momentum"],
            "momentum_x_relative_l2": component_errors["momentum_x"],
            "momentum_y_relative_l2": component_errors["momentum_y"],
            "kinetic_energy_relative_l2": physical_errors["kinetic_energy"],
        }
        if self.maximum_relative_l2 is not None:
            metrics["macro_match_tolerance"] = self.maximum_relative_l2
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
            title="VPFP macro match",
            summary=(
                "Candidate moments matched against controller-private "
                "full-velocity truth; data observations are not targets."
            ),
            values={
                "vpfp_macro_match_v1_relative_l2": result.primary_error,
                "rho_relative_l2": result.metrics["rho_relative_l2"],
                "momentum_relative_l2": result.metrics[
                    "momentum_relative_l2"
                ],
                "kinetic_energy_relative_l2": result.metrics[
                    "kinetic_energy_relative_l2"
                ],
                "gate": (
                    "not configured" if result.passed is None else result.passed
                ),
            },
        )

    def _validate_target_coordinates(self, target: MacroTarget) -> None:
        if target.query.kind is QueryKind.POINTS:
            if target.values.points != target.query.points:
                raise ValueError(
                    "VPFP macro target points must match the query exactly"
                )
            if target.values.shape != (len(target.query.points),):
                raise ValueError("VPFP point target shape does not match the query")
            return

        axes = target.values.axes
        if not axes:
            raise ValueError("VPFP grid macro targets must declare their axes")
        for name, count in target.query.resolution.items():
            if name not in axes or len(axes[name]) != int(count):
                raise ValueError(
                    f"VPFP macro target axis {name!r} does not match the query"
                )
        expected_shape = tuple(len(values) for values in axes.values())
        if target.values.shape != expected_shape:
            raise ValueError("VPFP grid target shape does not match its axes")
        if any(
            not isfinite(float(coordinate))
            for values in axes.values()
            for coordinate in values
        ):
            raise ValueError("VPFP macro target axes must be finite")

    def _validate_prediction(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> None:
        if case.case_id != target.case_id:
            raise ValueError("VPFP macro target belongs to a different case")
        if target.contract_id != self.contract_id:
            raise ValueError("VPFP macro target uses a different contract")
        if set(target.values.fields) != set(self.fields):
            raise ValueError(
                "VPFP macro target fields must match the contract"
            )
        self._validate_target_coordinates(target)
        if prediction.kind is not target.values.kind:
            raise ValueError("VPFP macro prediction uses the wrong query kind")
        if set(prediction.fields) != set(self.fields):
            raise ValueError(
                "VPFP macro prediction fields must match the contract"
            )
        if prediction.shape != target.values.shape:
            raise ValueError("VPFP macro prediction and target shapes differ")
        if target.query.kind is QueryKind.GRID:
            if set(prediction.axes) != set(target.values.axes) or any(
                tuple(prediction.axes[name])
                != tuple(target.values.axes[name])
                for name in target.values.axes
            ):
                raise ValueError(
                    "VPFP macro prediction axes differ from the target"
                )
        elif prediction.points != target.values.points:
            raise ValueError(
                "VPFP macro prediction points differ from the target"
            )
        for name in self.fields:
            values = prediction.fields[name]
            if len(values) != len(target.values.fields[name]):
                raise ValueError(f"VPFP macro sample count changed for {name}")
            if any(not isfinite(float(value)) for value in values):
                raise ValueError(f"VPFP macro prediction {name} is non-finite")


def relative_l2(actual: Sequence[float], expected: Sequence[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            "prediction and truth contain different sample counts"
        )
    squared_error = sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(actual, expected)
    )
    squared_truth = sum(float(value) ** 2 for value in expected)
    return sqrt(squared_error / max(squared_truth, 1.0e-30))
