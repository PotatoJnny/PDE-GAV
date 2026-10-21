from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.operators import (
    VPFPGrid,
    VPFPTime,
    make_axes,
    poisson_electric_field,
    velocity_weights,
)

Array = np.ndarray

FORBIDDEN_PUBLIC_FILENAMES = {
    "continuous_private_state.npz",
    "hidden_truth_cache.npz",
    "hidden_truth_manifest.json",
}
FORBIDDEN_PUBLIC_NPZ_KEYS = {
    "rho_true",
    "g_true",
    "f_true",
    "phi_true",
    "E_true",
    "hidden_answer",
    "private_answer",
    "basis_coeffs_true",
    "momentum_true",
    "kinetic_energy_true",
    "raw_second_moment_true",
}
FORBIDDEN_PUBLIC_JSON_KEYS = {
    "rho_true",
    "g_true",
    "f_true",
    "phi_true",
    "E_true",
    "hidden_answer",
    "private_answer",
    "latent",
    "micro_latent",
    "decoder_checkpoint",
    "decoder_checkpoint_sha256",
    "selection_score",
    "momentum_true",
    "kinetic_energy_true",
    "raw_second_moment_true",
}


@dataclass(frozen=True)
class VPFPCase:
    case_id: str
    grid: VPFPGrid
    time: VPFPTime
    collision_nu: float
    rho: Array
    g: Array
    source: Array
    phi: Array
    electric: Array
    diagnostics: dict[str, Any]
    case_spec: dict[str, Any]
    generator_metadata: dict[str, Any]
    basis_coeffs: Array | None = None
    basis_metadata: dict[str, Any] | None = None
    public_metadata: dict[str, Any] | None = None
    source_origin: str = "manufactured_solution_residual"
    macro_truth: dict[str, Array] | None = None


def write_case(case_dir: str | Path, case: VPFPCase) -> None:
    path = Path(case_dir)
    path.mkdir(parents=True, exist_ok=True)
    axes = make_axes(case.grid, case.time)
    public_metadata = dict(case.public_metadata or {})
    basis_metadata = case.basis_metadata or public_metadata.get("basis")
    source_origin = str(
        case.source_origin
        or case.case_spec.get("source_origin", "manufactured_solution_residual")
    )
    macro_truth = _validated_macro_truth(case.macro_truth, case.rho)
    macro_output = macro_truth is not None
    truth_rho = (
        np.asarray(macro_truth["rho"], dtype=np.float64)
        if macro_output
        else np.asarray(case.rho, dtype=np.float64)
    )
    target_shapes = {"rho": list(truth_rho.shape)}
    if macro_output:
        target_shapes.update(
            {
                "momentum": list(np.asarray(macro_truth["momentum"]).shape),
                "kinetic_energy": list(
                    np.asarray(macro_truth["kinetic_energy"]).shape
                ),
            }
        )
    else:
        target_shapes["g"] = list(np.asarray(case.g).shape)
        if case.basis_coeffs is not None:
            target_shapes["basis_coeffs"] = list(
                np.asarray(case.basis_coeffs).shape
            )
    public_question = {
        "case_id": case.case_id,
        "equation": "electrostatic_vlasov_poisson_fokker_planck_2x2v_factorized_v1",
        "task": "predict factorized rho(t,x,y) and g(t,x,y,vx,vy) for a VPFP case",
        "factorization": "f(t,x,v)=rho(t,x)*g(t,v|x)",
        "source_origin": source_origin,
        "target_shapes": target_shapes,
        "grid": case.grid.to_dict(),
        "time": case.time.to_dict(),
        "collision_nu": float(case.collision_nu),
        "basis": basis_metadata,
        "source_family": public_metadata.get("source_family"),
        "poisson": {
            "equation": "-Delta_x phi = rho - mean_x(rho)",
            "electric_field": "E = -grad_x phi",
            "x_boundary": "periodic",
        },
        "velocity_boundary": "truncated_box_homogeneous_neumann_for_collision",
        "arrays": {
            "x": "public_arrays.npz:x",
            "y": "public_arrays.npz:y",
            "vx": "public_arrays.npz:vx",
            "vy": "public_arrays.npz:vy",
            "t": "public_arrays.npz:t",
            "velocity_weights": "public_arrays.npz:velocity_weights",
            "rho0": "public_arrays.npz:rho0",
            "g0": "public_arrays.npz:g0",
            "source": "public_arrays.npz:source",
        },
        "solver_contract": "return {'rho': rho_pred, 'g': g_pred}; optional {'rho': rho_pred, 'basis_coeffs': coeffs} is decoded by the verifier",
        "generator_backend": str(case.generator_metadata.get("backend", "unknown")),
        "generator_representation": str(
            case.generator_metadata.get("representation", "factorized_rho_g")
        ),
    }
    question_overrides = public_metadata.get("question_overrides", {})
    if not isinstance(question_overrides, dict):
        raise TypeError("public question_overrides must be a mapping")
    allowed_question_overrides = {
        "background_density",
        "basis",
        "collision_operator",
        "domain",
        "equation",
        "equation_version",
        "factorization",
        "normalization",
        "poisson",
        "solver_contract",
        "task",
        "temperature",
        "velocity_boundary",
    }
    unexpected = set(question_overrides) - allowed_question_overrides
    if unexpected:
        raise ValueError(
            f"unsupported public question override keys: {sorted(unexpected)}"
        )
    public_question.update(question_overrides)
    if macro_output:
        public_question.update(
            {
                "task": (
                    "predict macroscopic rho, momentum density, and kinetic "
                    "energy density for the full kinetic VPFP equation"
                ),
                "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
                "solver_contract": (
                    "return {'rho': rho_pred, 'momentum': momentum_pred, "
                    "'kinetic_energy': kinetic_energy_pred} on target_shapes"
                ),
                "macroscopic_quantities": {
                    "rho": "integral_R2 f dv",
                    "momentum": "integral_R2 v f dv; final axis is (x,y)",
                    "kinetic_energy": "0.5*integral_R2 |v|^2 f dv",
                },
                "macro_truth": {
                    "schema": (
                        "vpfp_equation_first_refined_numerical_macro_truth_v1"
                    ),
                    "velocity_integration": (
                        "reference-grid quadrature before restriction to the "
                        "public time-space grid"
                    ),
                    "fullspace_exact": False,
                },
            }
        )
    public_arrays = {
        **axes,
        "velocity_weights": velocity_weights(case.grid),
        "rho0": np.asarray(truth_rho[0], dtype=np.float64),
        "g0": np.asarray(case.g[0], dtype=np.float64),
        "source": np.asarray(case.source, dtype=np.float64),
    }
    if macro_output:
        public_arrays.update(
            {
                "momentum0": np.asarray(
                    macro_truth["momentum"][0], dtype=np.float64
                ),
                "kinetic_energy0": np.asarray(
                    macro_truth["kinetic_energy"][0], dtype=np.float64
                ),
            }
        )
        phi_true, electric_true = poisson_electric_field(truth_rho, case.grid)
    else:
        phi_true = np.asarray(case.phi, dtype=np.float64)
        electric_true = np.asarray(case.electric, dtype=np.float64)
    hidden_arrays = {
        "rho_true": truth_rho,
        "g_true": np.asarray(case.g, dtype=np.float64),
        "phi_true": phi_true,
        "E_true": electric_true,
    }
    hidden_manifest_arrays = {
        "rho_true": "hidden_truth_cache.npz:rho_true",
        "g_true": "hidden_truth_cache.npz:g_true",
        "phi_true": "hidden_truth_cache.npz:phi_true",
        "E_true": "hidden_truth_cache.npz:E_true",
    }
    if macro_output:
        for name in (
            "momentum_true",
            "kinetic_energy_true",
            "raw_second_moment_true",
        ):
            source_name = name.removesuffix("_true")
            hidden_arrays[name] = np.asarray(
                macro_truth[source_name], dtype=np.float64
            )
            hidden_manifest_arrays[name] = f"hidden_truth_cache.npz:{name}"
    if case.basis_coeffs is not None:
        hidden_arrays["basis_coeffs_true"] = np.asarray(
            case.basis_coeffs, dtype=np.float64
        )
        hidden_manifest_arrays["basis_coeffs_true"] = (
            "hidden_truth_cache.npz:basis_coeffs_true"
        )
    hidden_manifest = {
        "case_id": case.case_id,
        "truth_type": (
            "equation_first_refined_numerical_macro_answer"
            if macro_output
            else "factorized_vpfp_answer"
        ),
        "source_origin": source_origin,
        "hidden_arrays": hidden_manifest_arrays,
        "f_true_policy": (
            "macro_truth_integrated_on_refined_velocity_grid; "
            "g_true_is_a_separate_public-grid_reference_view"
            if macro_output
            else "derived_from_rho_true_times_g_true_not_primary_parameterization"
        ),
    }
    np.savez_compressed(path / "public_arrays.npz", **public_arrays)
    np.savez_compressed(path / "hidden_truth_cache.npz", **hidden_arrays)
    _write_json(path / "public_question.json", public_question)
    _write_json(path / "hidden_truth_manifest.json", hidden_manifest)
    _write_json(path / "diagnostics.json", case.diagnostics)
    _write_json(path / "case_spec.json", case.case_spec)
    _write_json(path / "generator_metadata.json", case.generator_metadata)
    assert_public_bundle_isolated(path)


def _validated_macro_truth(
    macro_truth: dict[str, Array] | None,
    fallback_rho: Array,
) -> dict[str, Array] | None:
    if macro_truth is None:
        return None
    payload = {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in dict(macro_truth).items()
    }
    if "rho" not in payload:
        payload["rho"] = np.asarray(fallback_rho, dtype=np.float64)
    rho_shape = tuple(payload["rho"].shape)
    expected = {
        "rho": rho_shape,
        "momentum": (*rho_shape, 2),
        "kinetic_energy": rho_shape,
        "raw_second_moment": (*rho_shape, 3),
    }
    for name, shape in expected.items():
        value = payload.get(name)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(
                f"macro truth {name} has shape "
                f"{None if value is None else tuple(value.shape)}, expected {shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"macro truth {name} contains non-finite values")
    return payload


def load_case(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir)
    public_question = json.loads((path / "public_question.json").read_text())
    grid = VPFPGrid(**_tupled_grid(public_question["grid"]))
    time = VPFPTime(**public_question["time"])
    with np.load(path / "public_arrays.npz") as public_npz:
        public_arrays = {name: np.asarray(public_npz[name]) for name in public_npz.files}
    with np.load(path / "hidden_truth_cache.npz") as hidden_npz:
        hidden_arrays = {name: np.asarray(hidden_npz[name]) for name in hidden_npz.files}
    diagnostics = _read_json(path / "diagnostics.json") if (path / "diagnostics.json").exists() else {}
    case_spec = _read_json(path / "case_spec.json") if (path / "case_spec.json").exists() else {}
    generator_metadata = _read_json(path / "generator_metadata.json") if (path / "generator_metadata.json").exists() else {}
    return {
        "case_dir": path,
        "case_id": str(public_question["case_id"]),
        "public_question": public_question,
        "grid": grid,
        "time": time,
        "collision_nu": float(public_question["collision_nu"]),
        "public_arrays": public_arrays,
        "hidden_arrays": hidden_arrays,
        "diagnostics": diagnostics,
        "case_spec": case_spec,
        "generator_metadata": generator_metadata,
    }


def public_case_view(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir)
    public_question = json.loads((path / "public_question.json").read_text())
    grid = VPFPGrid(**_tupled_grid(public_question["grid"]))
    time = VPFPTime(**public_question["time"])
    with np.load(path / "public_arrays.npz") as public_npz:
        public_arrays = {name: np.asarray(public_npz[name]) for name in public_npz.files}
    return {
        "case_id": str(public_question["case_id"]),
        "question": public_question,
        "arrays": public_arrays,
        "grid": grid,
        "time": time,
        "collision_nu": float(public_question["collision_nu"]),
    }


def assert_public_bundle_isolated(case_dir: str | Path) -> None:
    path = Path(case_dir)
    leaks: list[str] = []
    for item in [path / "public_question.json", path / "public_arrays.npz"]:
        if not item.exists():
            leaks.append(f"missing public file: {item}")
            continue
        if item.name in FORBIDDEN_PUBLIC_FILENAMES:
            leaks.append(f"forbidden public filename: {item}")
        if item.suffix == ".npz":
            with np.load(item) as data:
                bad = FORBIDDEN_PUBLIC_NPZ_KEYS.intersection(data.files)
                if bad:
                    leaks.append(f"forbidden public npz keys {sorted(bad)} in {item}")
        if item.suffix == ".json":
            bad = _find_forbidden_json_keys(json.loads(item.read_text()))
            if bad:
                leaks.append(f"forbidden public json keys {sorted(bad)} in {item}")
    if leaks:
        raise AssertionError("VPFP public bundle leak detected:\n" + "\n".join(leaks))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _tupled_grid(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out["x_domain"] = tuple(float(x) for x in out.get("x_domain", (2.0 * np.pi, 2.0 * np.pi)))
    out["v_bounds"] = tuple(float(x) for x in out.get("v_bounds", (-5.0, 5.0, -5.0, 5.0)))
    return out


def _find_forbidden_json_keys(data: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key) in FORBIDDEN_PUBLIC_JSON_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_json_keys(value))
    elif isinstance(data, list):
        for value in data:
            found.update(_find_forbidden_json_keys(value))
    return found
