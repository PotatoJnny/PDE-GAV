from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.core.types import GridSpec, HiddenTruthCase, PDEParams, PublicCaseSpec, TimeSpec
from agent_verifier.manufacture.source_export import (
    GridFieldEvaluator,
    initial_condition_fn_from_evaluator,
    source_fn_from_evaluator,
)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_hidden_case(case: HiddenTruthCase, case_dir: str | Path) -> None:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    public = case.public
    write_json(
        case_dir / "public_case.json",
        {
            "case_id": public.case_id,
            "grid": {
                "shape": list(public.grid.shape),
                "domain": list(public.grid.domain),
                "dtype": public.grid.dtype,
                "dealias": public.grid.dealias,
            },
            "time": {"t0": public.time.t0, "t1": public.time.t1, "nt": public.time.nt},
            "params": {
                "nu": public.params.nu,
                "forcing_kmax": public.params.forcing_kmax,
                "equation": public.params.equation,
            },
            "public_params": public.public_params,
        },
    )
    times = np.linspace(public.time.t0, public.time.t1, public.time.nt)
    np.savez_compressed(case_dir / "hidden_truth.npz", u_true=case.u_true, f_grid=case.f_grid, times=times)
    write_json(case_dir / "validity.json", case.validity)
    metadata = {k: v for k, v in case.metadata.items() if k != "du_dt_true"}
    write_json(case_dir / "source_params.json", metadata)


def load_hidden_case(case_dir: str | Path) -> HiddenTruthCase:
    case_dir = Path(case_dir)
    public_data = read_json(case_dir / "public_case.json")
    truth = np.load(case_dir / "hidden_truth.npz")
    u_true = truth["u_true"]
    f_grid = truth["f_grid"]
    times = truth["times"]
    grid = GridSpec(
        shape=tuple(public_data["grid"]["shape"]),
        domain=tuple(public_data["grid"]["domain"]),
        dtype=public_data["grid"].get("dtype", "float64"),
        dealias=bool(public_data["grid"].get("dealias", True)),
    )
    time = TimeSpec(**public_data["time"])
    params = PDEParams(**public_data["params"])
    source_eval = GridFieldEvaluator(grid, times, f_grid)
    init_eval = GridFieldEvaluator(grid, np.asarray([time.t0]), u_true[:1])
    public = PublicCaseSpec(
        case_id=public_data["case_id"],
        grid=grid,
        time=time,
        params=params,
        source_fn=source_fn_from_evaluator(source_eval),
        initial_condition_fn=initial_condition_fn_from_evaluator(init_eval),
        public_params=public_data.get("public_params", {}),
    )
    metadata = read_json(case_dir / "source_params.json") if (case_dir / "source_params.json").exists() else {}
    validity = read_json(case_dir / "validity.json")
    return HiddenTruthCase(public=public, u_true=u_true, f_grid=f_grid, metadata=metadata, validity=validity)


def save_cases(cases: list[HiddenTruthCase], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, case in enumerate(cases):
        case_dir = out_dir / case.public.case_id
        save_hidden_case(case, case_dir)
        manifest.append({"case_id": case.public.case_id, "path": case.public.case_id, "index": idx})
    write_json(out_dir / "manifest.json", {"cases": manifest})


def load_cases(cases_dir: str | Path) -> list[HiddenTruthCase]:
    cases_dir = Path(cases_dir)
    manifest_path = cases_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)["cases"]
        return [load_hidden_case(cases_dir / item["path"]) for item in manifest]
    return [load_hidden_case(path) for path in sorted(cases_dir.iterdir()) if path.is_dir()]
