"""Fail-closed audit of unified macro matches in formal VPFP evidence.

The established VPFP evaluator remains responsible for running candidate
subprocesses and selecting a winner.  During a unified campaign, every
accepted solver call additionally records a controller-side MacroMatcher
result.  This module validates those additive results without changing or
reinterpreting the legacy ranking metric.
"""

from __future__ import annotations

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean
import tempfile
from typing import Any, Mapping

from .legacy_macro_bridge import (
    LEGACY_MACRO_BRIDGE_SCHEMA,
    LEGACY_MACRO_BRIDGE_VERSION,
    VPFP_MACRO_AXIS_ORDER,
)
from .macro_matching import VPFPMacroMatcher


AUDIT_SCHEMA = "vpfp_unified_macro_contract_audit_v1"
FORMAL_SUMMARY_SCHEMA = "vpfp_long_time_robust_formal_summary_v2"
FORMAL_CONTROLLER_SCHEMA = "vpfp_long_time_robust_formal_private_controller_v2"
PRIVATE_STAGE_SCHEMA = "vpfp_private_paired_development_stage_v2"
BRIDGE_RESULT_KEY = "unified_macro_match"
COMPUTE_BUDGET_SCHEMA = "vpfp_solver_compute_budget_v1"
PRIMARY_METRIC = "vpfp_macro_match_v1_relative_l2"
METRIC_RELATIVE_TOLERANCE = 1.0e-12
METRIC_ABSOLUTE_TOLERANCE = 1.0e-15
REQUIRED_MACRO_METRICS = frozenset(
    {
        "macro_match_relative_l2",
        "macro_relative_l2",
        "vpfp_macro_match_v1_relative_l2",
        "legacy_four_scalar_macro_relative_l2",
        "rho_relative_l2",
        "momentum_relative_l2",
        "momentum_x_relative_l2",
        "momentum_y_relative_l2",
        "kinetic_energy_relative_l2",
    }
)
PAIR_NAMES = ("agent_pair", "champion_pair")
STAGE_NAMES = ("broad", "frontier")
VARIANT_NAMES = {"base", "perturbed"}
SOLVER_NAMES = {"candidate", "comparison"}
BUDGET_KEYS = frozenset(
    {
        "schema",
        "tier",
        "wall_time_seconds",
        "cpus",
        "gpus",
        "memory_bytes",
    }
)


class VPFPMacroContractAuditError(ValueError):
    """Formal evidence is incomplete or violates the unified contract."""


def validate_vpfp_formal_macro_contract(
    formal_root: Path | str,
    *,
    contract_id: str = VPFPMacroMatcher.contract_id,
) -> dict[str, object]:
    """Validate every accepted formal call and return a passing sidecar.

    This function raises :class:`VPFPMacroContractAuditError` on the first
    inconsistency.  It never treats the optional ``MacroMatchResult.passed``
    value as a scientific gate: the audit establishes that the configured
    matcher ran and produced coherent finite evidence, while the historical
    formal evaluator retains winner-selection authority.
    """

    root = Path(formal_root).resolve()
    if not root.is_dir():
        raise VPFPMacroContractAuditError(
            f"formal VPFP root does not exist: {root}"
        )
    expected_contract = str(contract_id).strip()
    if not expected_contract:
        raise VPFPMacroContractAuditError("macro contract id cannot be empty")

    summary_path = root / "formal_summary.json"
    summary = _read_object(summary_path)
    if (
        summary.get("schema") != FORMAL_SUMMARY_SCHEMA
        or summary.get("passed") is not True
    ):
        raise VPFPMacroContractAuditError(
            "formal VPFP summary is absent, failed, or has the wrong schema"
        )

    controller_path = root / "private" / "formal_controller_report.json"
    controller = _read_object(controller_path)
    if controller.get("schema") != FORMAL_CONTROLLER_SCHEMA:
        raise VPFPMacroContractAuditError(
            "formal VPFP controller report has the wrong schema"
        )
    public_summary = _required_path(controller, "public_summary")
    if public_summary != summary_path.resolve():
        raise VPFPMacroContractAuditError(
            "formal controller report is bound to a different public summary"
        )

    private_root = (root / "private").resolve()
    report_paths: set[Path] = set()
    stage_rows: list[dict[str, object]] = []
    all_primary_errors: list[float] = []
    total_calls = 0
    accepted_calls = 0
    unaccepted_calls = 0
    stage_case_bindings: dict[str, tuple[object, ...]] = {}
    stage_budget_schedules: dict[
        str, tuple[Mapping[str, object], ...]
    ] = {}

    for pair_name in PAIR_NAMES:
        pair = _mapping(controller.get(pair_name), f"{pair_name} controller row")
        for stage_name in STAGE_NAMES:
            key = f"{stage_name}_report"
            report_path = _required_path(pair, key)
            if not report_path.is_relative_to(private_root):
                raise VPFPMacroContractAuditError(
                    f"{pair_name}.{key} escapes the formal private root"
                )
            if report_path in report_paths:
                raise VPFPMacroContractAuditError(
                    "formal controller report reuses a private stage report"
                )
            report_paths.add(report_path)
            report = _read_object(report_path)
            expected_samples = _positive_integer(
                summary.get(f"{stage_name}_root_case_count"),
                f"formal summary {stage_name} root case count",
            )
            result = _validate_stage_report(
                report,
                pair_name=pair_name,
                stage_name=stage_name,
                expected_samples=expected_samples,
                contract_id=expected_contract,
            )
            bindings = tuple(result["case_bindings"])
            prior_bindings = stage_case_bindings.get(stage_name)
            if prior_bindings is None:
                stage_case_bindings[stage_name] = bindings
            elif prior_bindings != bindings:
                raise VPFPMacroContractAuditError(
                    f"{stage_name} macro bindings changed between formal pairs"
                )
            budget_schedule = tuple(result["budget_schedule"])
            prior_schedule = stage_budget_schedules.get(stage_name)
            if prior_schedule is None:
                stage_budget_schedules[stage_name] = budget_schedule
            elif not _budget_schedules_equal(
                prior_schedule,
                budget_schedule,
            ):
                raise VPFPMacroContractAuditError(
                    f"{stage_name} budget schedule changed between formal pairs"
                )
            total_calls += result["total_calls"]
            accepted_calls += result["accepted_calls"]
            unaccepted_calls += result["unaccepted_calls"]
            all_primary_errors.extend(result["primary_errors"])
            stage_rows.append(
                {
                    "pair": pair_name,
                    "stage": stage_name,
                    "root_case_count": expected_samples,
                    "budget_count": result["budget_count"],
                    "call_count": result["total_calls"],
                    "accepted_call_count": result["accepted_calls"],
                    "unaccepted_call_count": result["unaccepted_calls"],
                }
            )

    if len(report_paths) != len(PAIR_NAMES) * len(STAGE_NAMES):
        raise VPFPMacroContractAuditError(
            "formal macro audit did not resolve four distinct stage reports"
        )
    if total_calls <= 0 or accepted_calls <= 0:
        raise VPFPMacroContractAuditError(
            "formal macro audit found no accepted solver calls"
        )
    if accepted_calls != len(all_primary_errors):
        raise VPFPMacroContractAuditError(
            "formal macro audit retained only partial accepted-call evidence"
        )

    return {
        "schema": AUDIT_SCHEMA,
        "passed": True,
        "contract_id": expected_contract,
        "formal_root": str(root),
        "formal_summary": str(summary_path.resolve()),
        "stage_report_count": len(report_paths),
        "total_call_count": total_calls,
        "accepted_call_count": accepted_calls,
        "unaccepted_call_count": unaccepted_calls,
        "audited_macro_match_count": len(all_primary_errors),
        "macro_primary_error_mean": fmean(all_primary_errors),
        "macro_primary_error_maximum": max(all_primary_errors),
        "stages": stage_rows,
        "legacy_ranking_replaced": False,
        "used_for_winner_selection": False,
        "private_case_ids_included": False,
    }


def write_vpfp_formal_macro_contract_audit(
    formal_root: Path | str,
    output: Path | str | None = None,
    *,
    contract_id: str = VPFPMacroMatcher.contract_id,
) -> dict[str, object]:
    """Write passing evidence atomically; record failures in a sibling sidecar."""

    root = Path(formal_root).resolve()
    destination = (
        Path(output).resolve()
        if output is not None
        else root / "macro_contract_validation.json"
    )
    try:
        report = validate_vpfp_formal_macro_contract(
            root,
            contract_id=contract_id,
        )
    except Exception as exc:
        report = {
            "schema": AUDIT_SCHEMA,
            "passed": False,
            "contract_id": str(contract_id),
            "formal_root": str(root),
            "failure": f"{type(exc).__name__}: {exc}",
            "legacy_ranking_replaced": False,
            "used_for_winner_selection": False,
            "private_case_ids_included": False,
        }
        failure_destination = destination.with_name(
            f"{destination.stem}.failed{destination.suffix or '.json'}"
        )
        _atomic_json_write(failure_destination, report)
        destination.unlink(missing_ok=True)
        if isinstance(exc, VPFPMacroContractAuditError):
            raise
        raise VPFPMacroContractAuditError(str(exc)) from exc
    _atomic_json_write(destination, report)
    return report


def _validate_stage_report(
    report: Mapping[str, Any],
    *,
    pair_name: str,
    stage_name: str,
    expected_samples: int,
    contract_id: str,
) -> dict[str, Any]:
    location = f"{pair_name}.{stage_name}"
    if report.get("schema") != PRIVATE_STAGE_SCHEMA:
        raise VPFPMacroContractAuditError(
            f"{location} private stage has the wrong schema"
        )
    if report.get("stage_label") != f"formal_{stage_name}":
        raise VPFPMacroContractAuditError(
            f"{location} private stage has the wrong label"
        )
    sample_count = _positive_integer(
        report.get("sample_count"),
        f"{location} sample count",
    )
    if sample_count != expected_samples:
        raise VPFPMacroContractAuditError(
            f"{location} sample count differs from the formal summary"
        )
    budgets = _validate_budget_schedule(report.get("budgets"), location)
    roots = report.get("root_case_rows")
    if not isinstance(roots, list) or len(roots) != sample_count:
        raise VPFPMacroContractAuditError(
            f"{location} contains partial root-case evidence"
        )

    total_calls = 0
    accepted_calls = 0
    unaccepted_calls = 0
    primary_errors: list[float] = []
    case_ids: set[str] = set()
    case_bindings: list[tuple[str, tuple[int, ...], str]] = []
    for root_index, root_row in enumerate(roots):
        root_mapping = _mapping(root_row, f"{location} root {root_index}")
        variants = _mapping(
            root_mapping.get("variants"),
            f"{location} root {root_index} variants",
        )
        if set(variants) != VARIANT_NAMES:
            raise VPFPMacroContractAuditError(
                f"{location} root {root_index} contains partial variants"
            )
        for variant_name in sorted(VARIANT_NAMES):
            variant = _mapping(
                variants[variant_name],
                f"{location} root {root_index} {variant_name}",
            )
            solvers = _mapping(
                variant.get("solvers"),
                f"{location} root {root_index} {variant_name} solvers",
            )
            if set(solvers) != SOLVER_NAMES:
                raise VPFPMacroContractAuditError(
                    f"{location} root {root_index} {variant_name} "
                    "contains partial solver evidence"
                )
            variant_binding: tuple[str, tuple[int, ...], str] | None = None
            for solver_name in sorted(SOLVER_NAMES):
                solver = _mapping(
                    solvers[solver_name],
                    f"{location} root {root_index} {variant_name} {solver_name}",
                )
                calls = solver.get("budgets")
                if not isinstance(calls, list) or len(calls) != len(budgets):
                    raise VPFPMacroContractAuditError(
                        f"{location} root {root_index} {variant_name} "
                        f"{solver_name} contains partial budget evidence"
                    )
                for budget_index, raw_call in enumerate(calls):
                    call_location = (
                        f"{location} root {root_index} {variant_name} "
                        f"{solver_name} budget {budget_index}"
                    )
                    call = _mapping(raw_call, call_location)
                    call_budget = _validate_budget_identity(
                        call.get("budget"),
                        f"{call_location} budget identity",
                    )
                    if not _budget_identities_equal(
                        call_budget,
                        budgets[budget_index],
                    ):
                        raise VPFPMacroContractAuditError(
                            f"{call_location} does not match its stage budget"
                        )
                    accepted = call.get("accepted")
                    if not isinstance(accepted, bool):
                        raise VPFPMacroContractAuditError(
                            f"{call_location} has no boolean accepted state"
                        )
                    total_calls += 1
                    if not accepted:
                        unaccepted_calls += 1
                        if BRIDGE_RESULT_KEY in call:
                            raise VPFPMacroContractAuditError(
                                f"{call_location} attaches a macro match to an "
                                "unaccepted call"
                            )
                        continue
                    accepted_calls += 1
                    bridge = call.get(BRIDGE_RESULT_KEY)
                    if not isinstance(bridge, Mapping):
                        raise VPFPMacroContractAuditError(
                            f"{call_location} is missing {BRIDGE_RESULT_KEY}"
                        )
                    primary_error, binding = _validate_bridge_result(
                        bridge,
                        contract_id=contract_id,
                        location=call_location,
                    )
                    if variant_binding is None:
                        variant_binding = binding
                    elif variant_binding != binding:
                        raise VPFPMacroContractAuditError(
                            f"{call_location} changed the case or coordinate binding"
                        )
                    primary_errors.append(primary_error)
            if variant_binding is None:
                raise VPFPMacroContractAuditError(
                    f"{location} root {root_index} {variant_name} "
                    "has no accepted macro-bound call"
                )
            case_id = variant_binding[0]
            if case_id in case_ids:
                raise VPFPMacroContractAuditError(
                    f"{location} reuses macro case id {case_id!r}"
                )
            case_ids.add(case_id)
            case_bindings.append(variant_binding)

    expected_calls = sample_count * len(VARIANT_NAMES) * len(SOLVER_NAMES) * len(
        budgets
    )
    if total_calls != expected_calls:
        raise VPFPMacroContractAuditError(
            f"{location} contains partial formal call evidence"
        )
    return {
        "budget_count": len(budgets),
        "budget_schedule": budgets,
        "total_calls": total_calls,
        "accepted_calls": accepted_calls,
        "unaccepted_calls": unaccepted_calls,
        "primary_errors": primary_errors,
        "case_bindings": case_bindings,
    }


def _validate_bridge_result(
    bridge: Mapping[str, Any],
    *,
    contract_id: str,
    location: str,
) -> tuple[float, tuple[str, tuple[int, ...], str]]:
    if (
        bridge.get("schema") != LEGACY_MACRO_BRIDGE_SCHEMA
        or bridge.get("version") != LEGACY_MACRO_BRIDGE_VERSION
    ):
        raise VPFPMacroContractAuditError(
            f"{location} uses the wrong macro bridge schema or version"
        )
    if bridge.get("contract") != contract_id:
        raise VPFPMacroContractAuditError(
            f"{location} uses the wrong macro contract id"
        )
    case_id = bridge.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise VPFPMacroContractAuditError(
            f"{location} macro match has no case id"
        )
    if bridge.get("axis_order") != list(VPFP_MACRO_AXIS_ORDER):
        raise VPFPMacroContractAuditError(
            f"{location} macro match uses the wrong axis order"
        )
    shape = bridge.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != len(VPFP_MACRO_AXIS_ORDER)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
    ):
        raise VPFPMacroContractAuditError(
            f"{location} macro match has an invalid grid shape"
        )
    coordinate_digest = bridge.get("coordinate_digest")
    if not (
        isinstance(coordinate_digest, str)
        and len(coordinate_digest) == len("sha256:") + 64
        and coordinate_digest.startswith("sha256:")
        and all(
            character in "0123456789abcdef"
            for character in coordinate_digest[len("sha256:") :]
        )
    ):
        raise VPFPMacroContractAuditError(
            f"{location} macro match has an invalid coordinate digest"
        )
    if bridge.get("used_for_winner_selection") is not False:
        raise VPFPMacroContractAuditError(
            f"{location} macro bridge must remain diagnostic-only"
        )
    primary = _finite_nonnegative(
        bridge.get("primary_error"),
        f"{location} macro primary error",
    )
    metrics = _mapping(bridge.get("metrics"), f"{location} macro metrics")
    if not metrics:
        raise VPFPMacroContractAuditError(
            f"{location} macro metrics are empty"
        )
    normalized_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise VPFPMacroContractAuditError(
                f"{location} macro metric names must be nonempty strings"
            )
        normalized_metrics[name] = _finite_nonnegative(
            value,
            f"{location} macro metric {name}",
        )
    missing_metrics = REQUIRED_MACRO_METRICS - normalized_metrics.keys()
    if missing_metrics:
        raise VPFPMacroContractAuditError(
            f"{location} macro result is missing required metrics: "
            f"{sorted(missing_metrics)}"
        )
    if PRIMARY_METRIC not in normalized_metrics:
        raise VPFPMacroContractAuditError(
            f"{location} macro result is missing {PRIMARY_METRIC}"
        )
    if not isclose(
        normalized_metrics[PRIMARY_METRIC],
        primary,
        rel_tol=METRIC_RELATIVE_TOLERANCE,
        abs_tol=METRIC_ABSOLUTE_TOLERANCE,
    ):
        raise VPFPMacroContractAuditError(
            f"{location} macro primary aliases disagree"
        )
    for alias_name in (
        "macro_match_relative_l2",
        "macro_relative_l2",
    ):
        if not isclose(
            normalized_metrics[alias_name],
            primary,
            rel_tol=METRIC_RELATIVE_TOLERANCE,
            abs_tol=METRIC_ABSOLUTE_TOLERANCE,
        ):
            raise VPFPMacroContractAuditError(
                f"{location} macro primary aliases disagree"
            )

    legacy_expected = fmean(
        normalized_metrics[name]
        for name in (
            "rho_relative_l2",
            "momentum_x_relative_l2",
            "momentum_y_relative_l2",
            "kinetic_energy_relative_l2",
        )
    )
    if not isclose(
        normalized_metrics["legacy_four_scalar_macro_relative_l2"],
        legacy_expected,
        rel_tol=METRIC_RELATIVE_TOLERANCE,
        abs_tol=METRIC_ABSOLUTE_TOLERANCE,
    ):
        raise VPFPMacroContractAuditError(
            f"{location} legacy four-scalar metric disagrees with component mean"
        )

    physical_expected = fmean(
        normalized_metrics[name]
        for name in (
            "rho_relative_l2",
            "momentum_relative_l2",
            "kinetic_energy_relative_l2",
        )
    )
    if not isclose(
        primary,
        physical_expected,
        rel_tol=METRIC_RELATIVE_TOLERANCE,
        abs_tol=METRIC_ABSOLUTE_TOLERANCE,
    ):
        raise VPFPMacroContractAuditError(
            f"{location} macro primary disagrees with physical-field mean"
        )

    passed = bridge.get("passed")
    if passed is not None and not isinstance(passed, bool):
        raise VPFPMacroContractAuditError(
            f"{location} macro passed state must be bool or null"
        )
    failures = bridge.get("failures", [])
    if not isinstance(failures, list) or not all(
        isinstance(value, str) for value in failures
    ):
        raise VPFPMacroContractAuditError(
            f"{location} macro failures must be a string list"
        )
    if passed is True and failures:
        raise VPFPMacroContractAuditError(
            f"{location} passing macro match contains failures"
        )
    return primary, (case_id, tuple(shape), coordinate_digest)


def _validate_budget_schedule(
    value: object,
    location: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise VPFPMacroContractAuditError(
            f"{location} has no formal budget schedule"
        )
    schedule: list[dict[str, object]] = []
    for budget_index, raw_budget in enumerate(value):
        budget = _validate_budget_identity(
            raw_budget,
            f"{location} stage budget {budget_index}",
        )
        if any(
            _budget_identities_equal(budget, prior)
            for prior in schedule
        ):
            raise VPFPMacroContractAuditError(
                f"{location} contains a duplicate stage budget identity"
            )
        if schedule and int(budget["tier"]) <= int(schedule[-1]["tier"]):
            raise VPFPMacroContractAuditError(
                f"{location} budget tiers are duplicated or out of order"
            )
        if schedule and float(budget["wall_time_seconds"]) <= float(
            schedule[-1]["wall_time_seconds"]
        ):
            raise VPFPMacroContractAuditError(
                f"{location} budget wall times are duplicated or out of order"
            )
        schedule.append(budget)
    return tuple(schedule)


def _validate_budget_identity(
    value: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise VPFPMacroContractAuditError(
            f"{label} is missing or is not an object"
        )
    payload = dict(value)
    if set(payload) != BUDGET_KEYS:
        raise VPFPMacroContractAuditError(
            f"{label} fields must be exactly {sorted(BUDGET_KEYS)}"
        )
    if payload["schema"] != COMPUTE_BUDGET_SCHEMA:
        raise VPFPMacroContractAuditError(
            f"{label} uses the wrong compute-budget schema"
        )
    tier = payload["tier"]
    if isinstance(tier, bool) or not isinstance(tier, int) or tier < 0:
        raise VPFPMacroContractAuditError(
            f"{label} tier must be a nonnegative integer"
        )
    wall_time = payload["wall_time_seconds"]
    if (
        isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not isfinite(float(wall_time))
        or float(wall_time) <= 0.0
    ):
        raise VPFPMacroContractAuditError(
            f"{label} wall time must be finite and positive"
        )
    for name, minimum in (("cpus", 1), ("gpus", 0), ("memory_bytes", 1)):
        item = payload[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < minimum
        ):
            requirement = "positive" if minimum else "nonnegative"
            raise VPFPMacroContractAuditError(
                f"{label} {name} must be a {requirement} integer"
            )
    return payload


def _budget_identities_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        type(left[key]) is type(right[key])
        and left[key] == right[key]
        for key in BUDGET_KEYS
    )


def _budget_schedules_equal(
    left: tuple[Mapping[str, object], ...],
    right: tuple[Mapping[str, object], ...],
) -> bool:
    return (
        len(left) == len(right)
        and all(
            _budget_identities_equal(left_row, right_row)
            for left_row, right_row in zip(left, right, strict=True)
        )
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VPFPMacroContractAuditError(f"{label} must be an object")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VPFPMacroContractAuditError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VPFPMacroContractAuditError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise VPFPMacroContractAuditError(
            f"{label} must be finite and nonnegative"
        )
    return result


def _required_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VPFPMacroContractAuditError(f"formal report path {key!r} is missing")
    return Path(value).resolve()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise VPFPMacroContractAuditError(f"formal evidence file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VPFPMacroContractAuditError(
            f"cannot read formal evidence object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise VPFPMacroContractAuditError(
            f"formal evidence must be a JSON object: {path}"
        )
    return value


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    try:
        report = write_vpfp_formal_macro_contract_audit(
            args.formal_root,
            args.out or None,
        )
    except VPFPMacroContractAuditError as exc:
        raise SystemExit(f"VPFP formal macro-contract audit failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
