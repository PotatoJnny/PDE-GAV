from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.core.io import read_json, write_json
from agent_verifier.core.spectral import make_physical_grid
from agent_verifier.core.types import GridSpec, PDEParams, PublicCaseSpec, TimeSpec
from agent_verifier.manufacture.source_export import (
    GridFieldEvaluator,
    initial_condition_fn_from_evaluator,
    source_fn_from_evaluator,
)

FORBIDDEN_FILENAMES = {
    "hidden_truth.npz",
    "validity.json",
    "feedback.json",
}

FORBIDDEN_NPZ_KEYS = {
    "u_true",
    "du_dt_true",
    "hidden_truth",
    "truth",
    "validity",
}

FORBIDDEN_JSON_KEYS = {
    "u_true",
    "du_dt_true",
    "hidden_truth",
    "validity",
    "true_projected_residual_rel",
    "true_divergence_rel",
}


def _public_case_dict(case: PublicCaseSpec) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "grid": {
            "shape": list(case.grid.shape),
            "domain": list(case.grid.domain),
            "dtype": case.grid.dtype,
            "dealias": case.grid.dealias,
        },
        "time": {"t0": case.time.t0, "t1": case.time.t1, "nt": case.time.nt},
        "params": {
            "nu": case.params.nu,
            "forcing_kmax": case.params.forcing_kmax,
            "equation": case.params.equation,
        },
        "public_params": {
            **case.public_params,
            "boundary_condition": case.public_params.get("boundary_condition", "periodic"),
        },
    }


def save_public_case_bundle(hidden_case, out_dir: str | Path) -> None:
    """Write a public-only challenge bundle for one hidden case.

    The bundle includes public metadata, public forcing snapshots, and the
    public initial condition. It deliberately excludes hidden truth arrays,
    validity diagnostics, generator coefficients, and verifier feedback.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = hidden_case.public
    write_json(out_dir / "public_case.json", _public_case_dict(case))
    times = np.linspace(case.time.t0, case.time.t1, case.time.nt)
    x = make_physical_grid(case.grid)
    u0 = np.moveaxis(case.initial_condition_fn(x, case.public_params), -1, 0)
    np.savez_compressed(out_dir / "forcing_grid.npz", f_grid=hidden_case.f_grid, times=times)
    np.savez_compressed(out_dir / "initial_condition.npz", u0=u0)
    (out_dir / "README.md").write_text(
        "# Public PDE Challenge\n\n"
        "Implement `def solve(case): ...` and return `{'u': u_pred}`.\n\n"
        "Available public files: `public_case.json`, `forcing_grid.npz`, "
        "and `initial_condition.npz`. The velocity output shape must be "
        "`(case.time.nt, 3, *case.grid.shape)`.\n"
    )
    assert_no_hidden_truth_leak(out_dir)


def load_public_case_bundle(bundle_dir: str | Path) -> PublicCaseSpec:
    bundle_dir = Path(bundle_dir)
    public = read_json(bundle_dir / "public_case.json")
    forcing = np.load(bundle_dir / "forcing_grid.npz")
    initial = np.load(bundle_dir / "initial_condition.npz")
    grid = GridSpec(
        shape=tuple(public["grid"]["shape"]),
        domain=tuple(public["grid"]["domain"]),
        dtype=public["grid"].get("dtype", "float64"),
        dealias=bool(public["grid"].get("dealias", True)),
    )
    time = TimeSpec(**public["time"])
    params = PDEParams(**public["params"])
    times = forcing["times"]
    source_eval = GridFieldEvaluator(grid, times, forcing["f_grid"])
    init_eval = GridFieldEvaluator(grid, np.asarray([time.t0]), initial["u0"][None, ...])
    return PublicCaseSpec(
        case_id=public["case_id"],
        grid=grid,
        time=time,
        params=params,
        source_fn=source_fn_from_evaluator(source_eval),
        initial_condition_fn=initial_condition_fn_from_evaluator(init_eval),
        public_params=public.get("public_params", {}),
    )


def assert_no_hidden_truth_leak(path: str | Path) -> None:
    path = Path(path)
    leaks: list[str] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if item.name in FORBIDDEN_FILENAMES:
            leaks.append(f"forbidden filename: {item}")
        if item.suffix == ".npz":
            try:
                data = np.load(item)
                bad = FORBIDDEN_NPZ_KEYS.intersection(data.files)
                if bad:
                    leaks.append(f"forbidden npz keys {sorted(bad)} in {item}")
            except Exception as exc:  # noqa: BLE001
                leaks.append(f"cannot inspect npz {item}: {exc!r}")
        if item.suffix == ".json":
            try:
                data = json.loads(item.read_text())
            except Exception:
                continue
            bad = _find_forbidden_json_keys(data)
            if bad:
                leaks.append(f"forbidden json keys {sorted(bad)} in {item}")
    if leaks:
        raise AssertionError("Hidden truth leak detected:\n" + "\n".join(leaks))


def _find_forbidden_json_keys(data: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key in FORBIDDEN_JSON_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_json_keys(value))
    elif isinstance(data, list):
        for value in data:
            found.update(_find_forbidden_json_keys(value))
    return found
