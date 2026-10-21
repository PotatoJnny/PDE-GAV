from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.basis import decode_basis_coeffs, velocity_basis
from agent_verifier.vpfp.continuous_cases import (
    evaluate_continuous_prediction,
    is_standard_continuous_question,
    verify_continuous_case_dir,
)
from agent_verifier.vpfp.io import load_case
from agent_verifier.vpfp.operators import (
    EPS,
    f_from_rho_g,
    poisson_electric_field,
    make_axes,
    relative_l2,
    residual,
)

EXACT_DISCRETE_RESIDUAL_SOURCE_ORIGINS = {
    "manufactured_solution_residual",
    "generated_solution_residual",
}


def verify_case_dir(case_dir: str | Path, *, residual_tol: float = 1.0e-8, normalization_tol: float = 5.0e-6, require_residual: bool | None = None) -> dict[str, Any]:
    case = load_case(case_dir)
    if is_standard_continuous_question(case["public_question"]):
        return verify_continuous_case_dir(
            case_dir,
            residual_tol=residual_tol,
            normalization_tol=normalization_tol,
            require_residual=require_residual,
        )
    public = case["public_arrays"]
    hidden = case["hidden_arrays"]
    grid = case["grid"]
    time = case["time"]
    source = np.asarray(public["source"], dtype=float)
    rho = np.asarray(hidden["rho_true"], dtype=float)
    g = np.asarray(hidden["g_true"], dtype=float)
    res = residual(rho, g, source, grid=grid, time=time, collision_nu=case["collision_nu"])
    source_origin = str(case["public_question"].get("source_origin") or case["case_spec"].get("source_origin") or "manufactured_solution_residual")
    gate_residual = (
        source_origin in EXACT_DISCRETE_RESIDUAL_SOURCE_ORIGINS
        if require_residual is None
        else bool(require_residual)
    )
    g_mass = np.sum(g, axis=(-2, -1)) * grid.velocity_cell_area
    metrics = {
        "truth_vpfp_residual_rel": relative_l2(res, source) if np.linalg.norm(source) > EPS else float(np.linalg.norm(res)),
        "g_normalization_max_abs_error": float(np.max(np.abs(g_mass - 1.0))),
        "rho_min": float(np.min(rho)),
        "g_min": float(np.min(g)),
        "source_rms": float(np.sqrt(np.mean(source**2))),
        "repeat_initial_difficulty": float(case.get("diagnostics", {}).get("repeat_initial_difficulty", 0.0)),
        "source_origin": source_origin,
        "finite_difference_residual_is_hard_gate": bool(gate_residual),
    }
    failures = []
    if gate_residual and metrics["truth_vpfp_residual_rel"] > float(residual_tol):
        failures.append("truth_vpfp_residual_rel")
    if metrics["g_normalization_max_abs_error"] > float(normalization_tol):
        failures.append("g_normalization_max_abs_error")
    if metrics["rho_min"] < -1.0e-10:
        failures.append("rho_negative")
    if metrics["g_min"] < -1.0e-10:
        failures.append("g_negative")
    return {"accepted": not failures, "metrics": metrics, "failures": failures, "case_id": case["case_id"]}


def evaluate_prediction(case_dir: str | Path, output: dict[str, Any]) -> dict[str, Any]:
    case = load_case(case_dir)
    if is_standard_continuous_question(case["public_question"]):
        return evaluate_continuous_prediction(case_dir, output)
    status, rho_pred, g_pred = _coerce_output(case, output)
    if status != "ok" or rho_pred is None or g_pred is None:
        return _failed(case["case_id"], status)

    grid = case["grid"]
    time = case["time"]
    hidden = case["hidden_arrays"]
    public = case["public_arrays"]
    rho_true = np.asarray(hidden["rho_true"], dtype=float)
    g_true = np.asarray(hidden["g_true"], dtype=float)
    f_true = f_from_rho_g(rho_true, g_true)
    f_pred = f_from_rho_g(rho_pred, g_pred)
    phi_pred, electric_pred = poisson_electric_field(rho_pred, grid)
    _phi_true = np.asarray(hidden["phi_true"], dtype=float)
    electric_true = np.asarray(hidden["E_true"], dtype=float)
    source = np.asarray(public["source"], dtype=float)
    res = residual(
        rho_pred,
        g_pred,
        source,
        grid=grid,
        time=time,
        collision_nu=case["collision_nu"],
    )
    truth_res = residual(
        rho_true,
        g_true,
        source,
        grid=grid,
        time=time,
        collision_nu=case["collision_nu"],
    )
    g_mass = np.sum(g_pred, axis=(-2, -1)) * grid.velocity_cell_area
    rho_l2 = relative_l2(rho_pred - rho_true, rho_true)
    g_l2 = relative_l2(g_pred - g_true, g_true)
    f_l2 = relative_l2(f_pred - f_true, f_true)
    field_l2 = relative_l2(electric_pred - electric_true, electric_true)
    residual_rel = relative_l2(res, source)
    reference_residual_rel = relative_l2(truth_res, source)
    residual_excess_rel = relative_l2(res - truth_res, source)
    g_norm = float(np.max(np.abs(g_mass - 1.0)))
    rho_neg = float(np.linalg.norm(np.minimum(rho_pred, 0.0)) / (np.linalg.norm(rho_pred) + EPS))
    g_neg = float(np.linalg.norm(np.minimum(g_pred, 0.0)) / (np.linalg.norm(g_pred) + EPS))
    score = f_l2 + 0.25 * rho_l2 + 0.10 * g_l2 + 0.10 * field_l2 + 0.10 * residual_excess_rel + g_norm + rho_neg + g_neg
    kinetic_errors = _kinetic_error_metrics(
        f_pred, f_true, grid=grid
    )
    return {
        "case_id": case["case_id"],
        "agent_status": "ok",
        "rho_rel_l2": float(rho_l2),
        "g_rel_l2": float(g_l2),
        "f_rel_l2": float(f_l2),
        "electric_field_rel_l2": float(field_l2),
        "vpfp_residual_rel": float(residual_rel),
        "reference_vpfp_residual_rel": float(reference_residual_rel),
        "vpfp_residual_excess_rel": float(residual_excess_rel),
        "g_normalization_max_abs_error": g_norm,
        "rho_positivity_violation": rho_neg,
        "g_positivity_violation": g_neg,
        "score_smaller_is_better": float(score),
        "adversarial_score_larger_is_harder": float(score),
        **kinetic_errors,
    }


def write_feedback(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = [row for row in rows if row.get("agent_status") == "ok"]
    summary = {
        "n_cases": len(rows),
        "n_ok": len(ok),
        "mean_score": _mean(ok, "score_smaller_is_better"),
        "mean_rho_rel_l2": _mean(ok, "rho_rel_l2"),
        "mean_f_rel_l2": _mean(ok, "f_rel_l2"),
        "mean_momentum_density_rel_l2": _mean(ok, "momentum_density_rel_l2"),
        "mean_kinetic_energy_density_rel_l2": _mean(ok, "kinetic_energy_density_rel_l2"),
        "mean_pressure_tensor_rel_l2": _mean(ok, "pressure_tensor_rel_l2"),
        "max_vpfp_residual_rel": _max(ok, "vpfp_residual_rel"),
    }
    for key, value in dict(summary_extra or {}).items():
        if key in summary:
            raise ValueError(f"duplicate VPFP feedback summary key: {key}")
        summary[str(key)] = value
    payload = {"summary": summary, "cases": rows}
    (out / "feedback.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# VPFP Verifier Feedback", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return payload


def _coerce_output(case: dict[str, Any], output: dict[str, Any]) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    if not isinstance(output, dict):
        return "failed", None, None
    rho_val = output.get("rho", output.get("rho_pred"))
    g_val = output.get("g", output.get("g_pred"))
    coeff_val = output.get("basis_coeffs", output.get("coeffs_pred"))
    if rho_val is None or (g_val is None and coeff_val is None):
        return "missing_rho_or_g", None, None
    rho = np.asarray(rho_val, dtype=float)
    expected_rho = tuple(case["hidden_arrays"]["rho_true"].shape)
    expected_g = tuple(case["hidden_arrays"]["g_true"].shape)
    if tuple(rho.shape) != expected_rho:
        return "bad_shape", None, None
    if g_val is not None:
        g = np.asarray(g_val, dtype=float)
    else:
        basis_meta = case["public_question"].get("basis") or {}
        atoms, _metadata = velocity_basis(case["grid"], basis_meta.get("config", basis_meta))
        coeffs = np.asarray(coeff_val, dtype=float)
        expected_coeff = expected_g[:-2] + (atoms.shape[0],)
        if tuple(coeffs.shape) != expected_coeff:
            return "bad_shape", None, None
        g = decode_basis_coeffs(coeffs, atoms)
    if tuple(g.shape) != expected_g:
        return "bad_shape", None, None
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(g)):
        return "nan", None, None
    return "ok", rho, g


def _kinetic_error_metrics(
    f_pred: np.ndarray,
    f_true: np.ndarray,
    *,
    grid,
) -> dict[str, float]:
    axes = make_axes(grid)
    vx = np.asarray(axes["vx"], dtype=float)[:, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, :]
    area = float(grid.velocity_cell_area)
    radius_squared = vx**2 + vy**2
    vmax_x = max(abs(grid.v_bounds[0]), abs(grid.v_bounds[1]))
    vmax_y = max(abs(grid.v_bounds[2]), abs(grid.v_bounds[3]))
    tail_mask = (np.abs(vx) >= 0.80 * vmax_x) | (np.abs(vy) >= 0.80 * vmax_y)

    def fields(f_value: np.ndarray) -> dict[str, np.ndarray]:
        value = np.asarray(f_value, dtype=float)
        density = np.sum(value, axis=(-2, -1)) * area
        momentum_x = np.sum(value * vx, axis=(-2, -1)) * area
        momentum_y = np.sum(value * vy, axis=(-2, -1)) * area
        safe_density = np.maximum(density, EPS)
        raw_xx = np.sum(value * vx**2, axis=(-2, -1)) * area
        raw_xy = np.sum(value * vx * vy, axis=(-2, -1)) * area
        raw_yy = np.sum(value * vy**2, axis=(-2, -1)) * area
        momentum = np.stack(
            (
                momentum_x,
                momentum_y,
            ),
            axis=1,
        )
        pressure = np.stack(
            (
                raw_xx - momentum_x**2 / safe_density,
                raw_xy - momentum_x * momentum_y / safe_density,
                raw_yy - momentum_y**2 / safe_density,
            ),
            axis=1,
        )
        return {
            "momentum": momentum,
            "kinetic_energy": (
                0.5 * np.sum(value * radius_squared, axis=(-2, -1)) * area
            ),
            "pressure": pressure,
            "entropy": (
                np.sum(
                    np.maximum(value, EPS) * np.log(np.maximum(value, EPS)),
                    axis=(-2, -1),
                )
                * area
            ),
            "tail_mass": np.sum(value * tail_mask, axis=(-2, -1)) * area,
        }

    predicted = fields(f_pred)
    reference = fields(f_true)
    return {
        "momentum_density_rel_l2": relative_l2(
            predicted["momentum"] - reference["momentum"],
            reference["momentum"],
        ),
        "kinetic_energy_density_rel_l2": relative_l2(
            predicted["kinetic_energy"] - reference["kinetic_energy"],
            reference["kinetic_energy"],
        ),
        "pressure_tensor_rel_l2": relative_l2(
            predicted["pressure"] - reference["pressure"],
            reference["pressure"],
        ),
        "entropy_density_rel_l2": relative_l2(
            predicted["entropy"] - reference["entropy"],
            reference["entropy"],
        ),
        "velocity_tail_mass_rel_l2": relative_l2(
            predicted["tail_mass"] - reference["tail_mass"],
            reference["tail_mass"],
        ),
    }


def _failed(case_id: str, status: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "agent_status": status,
        "rho_rel_l2": float("inf"),
        "g_rel_l2": float("inf"),
        "f_rel_l2": float("inf"),
        "electric_field_rel_l2": float("inf"),
        "vpfp_residual_rel": float("inf"),
        "reference_vpfp_residual_rel": float("inf"),
        "vpfp_residual_excess_rel": float("inf"),
        "momentum_density_rel_l2": float("inf"),
        "kinetic_energy_density_rel_l2": float("inf"),
        "pressure_tensor_rel_l2": float("inf"),
        "entropy_density_rel_l2": float("inf"),
        "velocity_tail_mass_rel_l2": float("inf"),
        "score_smaller_is_better": float("inf"),
        "adversarial_score_larger_is_harder": float("inf"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("inf")


def _max(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.max(values)) if values else float("inf")
