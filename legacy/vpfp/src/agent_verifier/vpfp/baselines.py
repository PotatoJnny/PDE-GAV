from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_macros import is_continuous_macro_question
from agent_verifier.vpfp.io import load_case, public_case_view


def truth_solution(case_dir: str | Path) -> dict[str, Any]:
    case = load_case(case_dir)
    if is_continuous_macro_question(case["public_question"]):
        return {
            "rho": np.asarray(case["hidden_arrays"]["rho_true"]).copy(),
            "momentum": np.asarray(
                case["hidden_arrays"]["momentum_true"]
            ).copy(),
            "kinetic_energy": np.asarray(
                case["hidden_arrays"]["kinetic_energy_true"]
            ).copy(),
            "metadata": {"solver": "vpfp_macro_truth"},
        }
    return {
        "rho": np.asarray(case["hidden_arrays"]["rho_true"]).copy(),
        "g": np.asarray(case["hidden_arrays"]["g_true"]).copy(),
        "metadata": {"solver": "vpfp_truth"},
    }


def hold_initial_solution(case_dir: str | Path) -> dict[str, Any]:
    case = load_case(case_dir)
    public = case["public_arrays"]
    nt = int(case["time"].nt)
    rho0 = np.asarray(public["rho0"], dtype=float)
    if is_continuous_macro_question(case["public_question"]):
        momentum0 = np.asarray(public["momentum0"], dtype=float)
        kinetic0 = np.asarray(public["kinetic_energy0"], dtype=float)
        return {
            "rho": np.repeat(rho0[None, ...], nt, axis=0),
            "momentum": np.repeat(momentum0[None, ...], nt, axis=0),
            "kinetic_energy": np.repeat(kinetic0[None, ...], nt, axis=0),
            "metadata": {"solver": "vpfp_macro_hold_initial"},
        }
    g0 = np.asarray(public["g0"], dtype=float)
    return {
        "rho": np.repeat(rho0[None, ...], nt, axis=0),
        "g": np.repeat(g0[None, ...], nt, axis=0),
        "metadata": {"solver": "vpfp_hold_initial"},
    }


def load_solver(path_or_name: str):
    if path_or_name == "truth":
        return lambda case_dir: truth_solution(case_dir)
    if path_or_name in {"hold", "hold_initial"}:
        return lambda case_dir: hold_initial_solution(case_dir)
    path = Path(path_or_name)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    module_dir = str(path.parent.resolve())
    inserted = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
    if hasattr(module, "solve_case"):
        return lambda case_dir: module.solve_case(str(case_dir))
    if hasattr(module, "solve"):
        return lambda case_dir: module.solve(public_case_view(case_dir))
    raise AttributeError(f"{path} must define solve_case(case_dir) or solve(public_case)")
