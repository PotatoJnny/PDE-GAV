#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from public_vpfp_runtime import (
    MACRO_OUTPUT_CONTRACT,
    decode_basis_coeffs,
    load_public_case,
    load_submission,
    poisson_electric_field,
    relative_l2,
    residual,
    run_submission,
    velocity_basis,
)


EPS = 1.0e-12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a VPFP solver on intentionally released training answers."
    )
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--bank", default="released_truth")
    parser.add_argument("--out", default="released_truth_feedback.json")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    bank = Path(args.bank)
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    module = load_submission(args.submission)
    rows = []
    for row in manifest.get("cases", []):
        case_dir = bank / str(row["case_dir"])
        try:
            case = load_public_case(case_dir)
            raw_output = run_submission(module, case_dir)
            with np.load(case_dir / "released_truth.npz", allow_pickle=False) as payload:
                truth = {name: np.asarray(payload[name], dtype=float) for name in payload.files}
            if _is_macro_case(case):
                rho, momentum, kinetic = _coerce_macro_output(case, raw_output)
                rows.append(
                    _macro_truth_metrics(case, truth, rho, momentum, kinetic)
                )
            else:
                rho, g = _coerce_output(case, raw_output)
                rows.append(_truth_metrics(case, truth, rho, g))
        except Exception as exc:  # noqa: BLE001 - training diagnostics retain failures.
            rows.append(
                {
                    "case_id": case_dir.name,
                    "status": "exception",
                    "score_smaller_is_better": float("inf"),
                    "exception": repr(exc),
                }
            )
    valid = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "schema": "vpfp_released_truth_feedback_v1",
        "case_count": len(rows),
        "valid_case_count": len(valid),
        "all_valid": bool(rows) and len(valid) == len(rows),
        "mean_score": _mean(valid, "score_smaller_is_better"),
        "mean_rho_rel_l2": _mean(valid, "rho_rel_l2"),
        "mean_momentum_density_rel_l2": _mean(valid, "momentum_density_rel_l2"),
        "mean_kinetic_energy_density_rel_l2": _mean(valid, "kinetic_energy_density_rel_l2"),
        "mean_g_rel_l2": _mean(valid, "g_rel_l2"),
        "mean_f_rel_l2": _mean(valid, "f_rel_l2"),
        "mean_electric_field_rel_l2": _mean(valid, "electric_field_rel_l2"),
        "mean_vpfp_residual_excess_rel": _mean(valid, "vpfp_residual_excess_rel"),
    }
    output = {"summary": summary, "cases": rows}
    Path(args.out).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_invalid and not summary["all_valid"]:
        raise SystemExit("one or more released training cases failed")


def _coerce_output(
    case: dict[str, Any],
    output: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(output, dict):
        raise TypeError("solver output must be a dict")
    rho_value = output.get("rho", output.get("rho_pred"))
    g_value = output.get("g", output.get("g_pred"))
    coeff_value = output.get("basis_coeffs", output.get("coeffs_pred"))
    if rho_value is None or (g_value is None and coeff_value is None):
        raise ValueError("solver output must contain rho and g or basis_coeffs")
    rho = np.asarray(rho_value, dtype=float)
    expected_rho = tuple(int(value) for value in case["question"]["target_shapes"]["rho"])
    expected_g = tuple(int(value) for value in case["question"]["target_shapes"]["g"])
    if tuple(rho.shape) != expected_rho:
        raise ValueError(f"bad rho shape: {rho.shape} != {expected_rho}")
    if g_value is not None:
        g = np.asarray(g_value, dtype=float)
    else:
        atoms = velocity_basis(
            case["grid"],
            (case["question"].get("basis") or {}).get("config"),
        )
        g = decode_basis_coeffs(np.asarray(coeff_value, dtype=float), atoms)
    if tuple(g.shape) != expected_g:
        raise ValueError(f"bad g shape: {g.shape} != {expected_g}")
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(g)):
        raise ValueError("solver output contains non-finite values")
    return rho, g


def _is_macro_case(case: dict[str, Any]) -> bool:
    return str(
        case["question"].get("solver_output_contract_version") or ""
    ) == MACRO_OUTPUT_CONTRACT


def _coerce_macro_output(
    case: dict[str, Any],
    output: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(output, dict):
        raise TypeError("solver output must be a dict")
    rho_value = output.get("rho", output.get("rho_pred"))
    momentum_value = output.get("momentum", output.get("j"))
    kinetic_value = output.get("kinetic_energy", output.get("K"))
    if rho_value is None or momentum_value is None or kinetic_value is None:
        raise ValueError(
            "macro solver output must contain rho, momentum, and kinetic_energy"
        )
    outputs = (
        np.asarray(rho_value, dtype=float),
        np.asarray(momentum_value, dtype=float),
        np.asarray(kinetic_value, dtype=float),
    )
    shapes = case["question"]["target_shapes"]
    expected = (
        tuple(int(value) for value in shapes["rho"]),
        tuple(int(value) for value in shapes["momentum"]),
        tuple(int(value) for value in shapes["kinetic_energy"]),
    )
    for name, value, expected_shape in zip(
        ("rho", "momentum", "kinetic_energy"), outputs, expected, strict=True
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"bad {name} shape: {value.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    return outputs


def _macro_truth_metrics(
    case: dict[str, Any],
    truth: dict[str, np.ndarray],
    rho: np.ndarray,
    momentum: np.ndarray,
    kinetic: np.ndarray,
) -> dict[str, Any]:
    required = {"rho_true", "momentum_true", "kinetic_energy_true", "E_true"}
    missing = required - set(truth)
    if missing:
        raise ValueError(f"released macro truth is missing {sorted(missing)}")
    rho_true = truth["rho_true"]
    momentum_true = truth["momentum_true"]
    kinetic_true = truth["kinetic_energy_true"]
    electric = poisson_electric_field(rho, case["grid"])
    rho_l2 = relative_l2(rho - rho_true, rho_true)
    momentum_l2 = relative_l2(momentum - momentum_true, momentum_true)
    kinetic_l2 = relative_l2(kinetic - kinetic_true, kinetic_true)
    field_l2 = relative_l2(electric - truth["E_true"], truth["E_true"])
    rho_neg = float(
        np.linalg.norm(np.minimum(rho, 0.0)) / (np.linalg.norm(rho) + EPS)
    )
    kinetic_neg = float(
        np.linalg.norm(np.minimum(kinetic, 0.0))
        / (np.linalg.norm(kinetic) + EPS)
    )
    score = (
        (rho_l2 + momentum_l2 + kinetic_l2) / 3.0
        + rho_neg
        + kinetic_neg
    )
    return {
        "case_id": case["case_id"],
        "status": "ok",
        "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
        "rho_rel_l2": float(rho_l2),
        "momentum_density_rel_l2": float(momentum_l2),
        "kinetic_energy_density_rel_l2": float(kinetic_l2),
        "electric_field_rel_l2": float(field_l2),
        "rho_positivity_violation": rho_neg,
        "kinetic_energy_positivity_violation": kinetic_neg,
        "score_smaller_is_better": float(score),
    }


def _truth_metrics(
    case: dict[str, Any],
    truth: dict[str, np.ndarray],
    rho: np.ndarray,
    g: np.ndarray,
) -> dict[str, Any]:
    rho_true = truth["rho_true"]
    g_true = truth["g_true"]
    f = rho[..., None, None] * g
    f_true = rho_true[..., None, None] * g_true
    electric = poisson_electric_field(rho, case["grid"])
    electric_true = truth["E_true"]
    source = np.asarray(case["arrays"]["source"], dtype=float)
    prediction_residual = residual(
        rho,
        g,
        source,
        grid=case["grid"],
        time=case["time"],
        collision_nu=case["collision_nu"],
    )
    truth_residual = residual(
        rho_true,
        g_true,
        source,
        grid=case["grid"],
        time=case["time"],
        collision_nu=case["collision_nu"],
    )
    rho_l2 = relative_l2(rho - rho_true, rho_true)
    g_l2 = relative_l2(g - g_true, g_true)
    f_l2 = relative_l2(f - f_true, f_true)
    field_l2 = relative_l2(electric - electric_true, electric_true)
    residual_excess = relative_l2(prediction_residual - truth_residual, source)
    weights = np.asarray(case["arrays"]["velocity_weights"], dtype=float)
    g_norm = float(np.max(np.abs(np.sum(g * weights, axis=(-2, -1)) - 1.0)))
    rho_neg = float(np.linalg.norm(np.minimum(rho, 0.0)) / (np.linalg.norm(rho) + EPS))
    g_neg = float(np.linalg.norm(np.minimum(g, 0.0)) / (np.linalg.norm(g) + EPS))
    score = (
        f_l2
        + 0.25 * rho_l2
        + 0.10 * g_l2
        + 0.10 * field_l2
        + 0.10 * residual_excess
        + g_norm
        + rho_neg
        + g_neg
    )
    return {
        "case_id": case["case_id"],
        "status": "ok",
        "rho_rel_l2": float(rho_l2),
        "g_rel_l2": float(g_l2),
        "f_rel_l2": float(f_l2),
        "electric_field_rel_l2": float(field_l2),
        "vpfp_residual_excess_rel": float(residual_excess),
        "g_normalization_max_abs_error": g_norm,
        "score_smaller_is_better": float(score),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if key in row and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
