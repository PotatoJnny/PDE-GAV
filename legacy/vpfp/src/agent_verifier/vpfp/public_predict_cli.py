#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from public_vpfp_runtime import (
    decode_basis_coeffs,
    evaluate_public_output,
    load_public_case,
    load_submission,
    run_submission,
    velocity_basis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VPFP solver on public cases and save prediction arrays.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--cases", default="cases")
    parser.add_argument("--out", default="predictions")
    args = parser.parse_args()

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    module = load_submission(args.submission)
    rows = []
    for case_dir in sorted(path for path in Path(args.cases).iterdir() if path.is_dir()):
        case_id = case_dir.name
        try:
            case = load_public_case(case_dir)
            raw = run_submission(module, case_dir)
            public_metrics = evaluate_public_output(case, raw)
            if public_metrics.get("status") != "ok":
                rows.append({"case_id": case_id, "status": str(public_metrics.get("status")), "metrics": public_metrics})
                continue
            arrays = _prediction_arrays(case, raw)
            prediction_path = output_root / f"{case_id}.npz"
            np.savez(prediction_path, **arrays)
            rows.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "prediction_file": prediction_path.name,
                    "array_keys": sorted(arrays),
                    "metrics": public_metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001 - retained as public execution diagnostics.
            rows.append({"case_id": case_id, "status": "exception", "exception": repr(exc)})
    summary = {
        "case_count": len(rows),
        "ok_count": sum(row.get("status") == "ok" for row in rows),
        "all_ok": bool(rows) and all(row.get("status") == "ok" for row in rows),
    }
    payload = {"summary": summary, "cases": rows}
    (output_root / "prediction_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _prediction_arrays(case: dict[str, Any], output: dict[str, Any]) -> dict[str, np.ndarray]:
    rho_value = output.get("rho", output.get("rho_pred"))
    g_value = output.get("g", output.get("g_pred"))
    coeff_value = output.get("basis_coeffs", output.get("coeffs_pred"))
    rho = np.asarray(rho_value, dtype=float)
    expected_rho = tuple(int(value) for value in case["question"]["target_shapes"]["rho"])
    expected_g = tuple(int(value) for value in case["question"]["target_shapes"]["g"])
    if tuple(rho.shape) != expected_rho:
        raise ValueError(f"bad rho shape: {rho.shape} != {expected_rho}")
    arrays = {"rho": rho}
    if g_value is not None:
        g = np.asarray(g_value, dtype=float)
        if tuple(g.shape) != expected_g:
            raise ValueError(f"bad g shape: {g.shape} != {expected_g}")
        arrays["g"] = g
    elif coeff_value is not None:
        coeffs = np.asarray(coeff_value, dtype=float)
        atoms = velocity_basis(case["grid"], (case["question"].get("basis") or {}).get("config"))
        expected_coeffs = expected_g[:-2] + (atoms.shape[0],)
        if tuple(coeffs.shape) != expected_coeffs:
            raise ValueError(f"bad basis coefficient shape: {coeffs.shape} != {expected_coeffs}")
        decoded = decode_basis_coeffs(coeffs, atoms)
        if tuple(decoded.shape) != expected_g:
            raise ValueError(f"decoded g shape mismatch: {decoded.shape} != {expected_g}")
        arrays["basis_coeffs"] = coeffs
    else:
        raise ValueError("submission omitted g and basis_coeffs")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("prediction contains non-finite values")
    return arrays


if __name__ == "__main__":
    main()
