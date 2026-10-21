"""Dataset-exact public GKW packet contract pinned by stable commit 6947a64."""

from __future__ import annotations

import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contract import GKW_LEGACY_MACRO_OUTPUT_CONTRACT, GKW_MACRO_FIELDS


GKW_DATASET_ID = "pinc_gyroswin_gkw_301_collisionless_es_adiabatic_ion"
GKW_OPERATOR_ID = "gkw_delta_f_es_adiabatic_ion_answer_first_v1"
GKW_GYARADAX_COMMIT = "318e3d11d7326d99e96f2bd5a8998a462261dcd4"
GKW_NEURAL_GK_COMMIT = "568411b33be486262b8c5cfe1dfb3e0fe97f6419"

GKW_GRID = MappingProxyType({
    "nvpar": 32,
    "nmu": 8,
    "ns": 16,
    "nkx": 85,
    "nky": 32,
    "nperiod": 1,
    "vpar_max": 3.0,
})
GKW_STATE_SHAPE = (32, 8, 16, 85, 32)
GKW_FIELD_SHAPE = (16, 85, 32)

GKW_PUBLIC_ARRAY_SPECS = MappingProxyType({
    "vpar": ((32,), "float64"),
    "intvp": ((32,), "float64"),
    "mu": ((8,), "float64"),
    "intmu": ((8,), "float64"),
    "s": ((16,), "float64"),
    "ints": ((16,), "float64"),
    "B": ((16,), "float64"),
    "F": ((16,), "float64"),
    "G": ((16,), "float64"),
    "E_psi_zeta": ((16,), "float64"),
    "R": ((16,), "float64"),
    "B_t_fraction": ((16,), "float64"),
    "kx": ((85,), "float64"),
    "ky": ((32,), "float64"),
    "parseval_ky": ((32,), "float64"),
    "D": ((16, 3), "float64"),
    "metric_perp": ((16, 3), "float64"),
    "mode_label": ((85, 32), "int32"),
    "ixplus": ((85, 32), "int32"),
    "ixminus": ((85, 32), "int32"),
    "pos_parallel_class": ((16, 85, 32), "int8"),
    "s_shift": ((9, 16, 85, 32), "int32"),
    "kx_shift": ((9, 16, 85, 32), "int32"),
    "valid_shift": ((9, 16, 85, 32), "bool"),
})
GKW_PUBLIC_ARRAY_NAMES = frozenset(GKW_PUBLIC_ARRAY_SPECS)
GKW_PUBLIC_SCALAR_NAMES = frozenset(
    {
        "q",
        "shat",
        "eps",
        "rlt_i",
        "rln_i",
        "ikxspace",
        "krhomax",
        "kthnorm",
        "dvp",
        "ds",
        "idisp",
        "dgrid",
        "mass_i",
        "charge_i",
        "temperature_i",
        "density_i",
        "vth_ratio_i",
        "d2X",
        "sign_B",
        "ixzero",
        "iyzero",
        "disp_par",
        "disp_vp",
        "disp_x",
        "disp_y",
    }
)

GKW_ARRAY_CONVENTIONS = MappingProxyType({
    "E_psi_zeta": (
        "signed GKW efun coefficient, equal to -E^{psi,zeta} in the pinned "
        "geometry assembly"
    ),
    "D": "GKW drift tensor columns used directly by drift_x/y",
    "metric_perp": "ordered as (g_yy, g_xy, g_xx) in k_perp^2",
    "ky": "nonnegative real-to-complex half spectrum",
    "parseval_ky": (
        "physical half-spectrum weights: 1 for ky=0 and 2 for every "
        "retained ky>0 mode"
    ),
    "kx": "centered signed spectrum",
})


def canonical_gkw_equation_contract() -> dict[str, object]:
    """Return the exact nested public equation declaration at the pin."""

    active_terms = [
        "I_parallel_streaming",
        "II_magnetic_curvature_gradB_drift",
        "III_nonlinear_gyroaveraged_ExB_bracket",
        "IV_mirror_trapping_vpar_advection",
        "V_density_temperature_gradient_drive",
        "VII_parallel_electrostatic_field_drive",
        "VIII_magnetic_drift_electrostatic_field_drive",
        "parallel_fourth_order_numerical_dissipation",
        "vpar_fourth_order_numerical_dissipation",
        "kx_ky_hyper_dissipation",
    ]
    closed_terms = [
        "collisions",
        "kinetic_electron_evolution",
        "A_parallel",
        "delta_B_parallel",
        "equilibrium_rotation_and_flow_shear",
        "neoclassical_background_correction",
    ]
    return {
        "schema": "fusion_gkw_public_equation_contract_v1",
        "dataset_id": GKW_DATASET_ID,
        "operator_id": GKW_OPERATOR_ID,
        "equation": (
            "d_t delta_f_i = R_GKW[delta_f_i, phi; public_coefficients] "
            "+ S_balance"
        ),
        "field_closure": (
            "GKW electrostatic gyrokinetic quasineutrality with zonal "
            "adiabatic-electron correction"
        ),
        "state_variable": "delta_f_i",
        "api_state_alias": "h",
        "api_alias_definition": "h == delta_f_i; A_parallel is disabled",
        "state_is_complex_half_ky_spectrum": True,
        "continuous_hidden_state_is_primary": True,
        "operator_grid_is_an_oracle_quadrature_view": True,
        "grid": dict(GKW_GRID),
        "semidiscrete_problem": {
            "class": "continuous_time_method_of_lines",
            "state_axis_order": [
                "v_parallel",
                "mu",
                "s",
                "kx",
                "ky_nonnegative_half_spectrum",
            ],
            "state_shape": list(GKW_STATE_SHAPE),
            "field_axis_order": [
                "s",
                "kx",
                "ky_nonnegative_half_spectrum",
            ],
            "phase_operator": (
                "fixed public GKW quadrature, finite-difference, "
                "twist-and-shift, and dealiased Fourier operator"
            ),
            "exactness_statement": (
                "the hidden continuous-time state satisfies this declared "
                "float64 semidiscrete equation with the numerical source "
                "oracle, up to float64 evaluation roundoff"
            ),
            "truth_precision": "float64",
            "float32_status": (
                "diagnostic throughput mode only; not the truth contract"
            ),
        },
        "public_solver_reference": {
            "reference_is_part_of_public_question": True,
            "modules": [
                "gkw_equation_contract.py",
                "gkw_public_packet.py",
                "gkw_torch_precompute.py",
                "gkw_torch_operator.py",
            ],
            "coefficient_rule": (
                "all derived operator coefficients are deterministic "
                "functions of required_public_arrays and "
                "required_public_scalars"
            ),
            "hidden_representation_included": False,
            "generator_parameters_included": False,
        },
        "regime": {
            "dataset_id": GKW_DATASET_ID,
            "operator_id": GKW_OPERATOR_ID,
            "state_variable": "delta_f_i",
            "api_state_alias": "h",
            "geometry_model": "circ",
            "parallel_boundary": "open_twist_and_shift_mode_topology",
            "perpendicular_boundary": "flux_tube_fourier_mode_box",
            "velocity_boundary": "truncated_fast_decay_zero_ghost_stencil",
            "nonlinear": True,
            "electrostatic": True,
            "adiabatic_electrons": True,
            "zonal_adiabatic": True,
            "collisions": False,
            "kinetic_species": 1,
            "beta": 0.0,
            "nlapar": False,
            "nlbpar": False,
            "vcor": 0.0,
            "uprim": 0.0,
            "eps": 0.19,
            "order_of_scheme": "fourth_order",
            "time_method": "EXP_meth2",
            "dtim": 0.01,
            "disp_par": 1.0,
            "disp_vp": 0.2,
            "disp_x": 0.1,
            "disp_y": 0.1,
        },
        "active_terms": active_terms,
        "closed_terms": closed_terms,
        "source_policy": {
            "origin": "generated_solution_residual",
            "definition": "S_balance = d_t delta_f_i - R_GKW",
            "absolute_norm_must_be_small": False,
            "physical_admissibility_required": True,
            "public_access": "metered numerical oracle",
            "neural_parameterization_public": False,
        },
        "array_conventions": dict(GKW_ARRAY_CONVENTIONS),
        "required_public_scalars": [
            "q",
            "shat",
            "eps",
            "rlt_i",
            "rln_i",
            "ikxspace",
            "krhomax",
            "kthnorm",
            "dvp",
            "ds",
            "idisp",
            "dgrid",
            "mass_i",
            "charge_i",
            "temperature_i",
            "density_i",
            "vth_ratio_i",
            "d2X",
            "sign_B",
            "ixzero",
            "iyzero",
            "disp_par",
            "disp_vp",
            "disp_x",
            "disp_y",
        ],
        "required_public_arrays": sorted(GKW_PUBLIC_ARRAY_NAMES),
        "provenance": {
            "neural_gyrokinetics_commit": GKW_NEURAL_GK_COMMIT,
            "gyaradax_audit_commit": GKW_GYARADAX_COMMIT,
            "gkw_reference_doi": "10.1016/j.cpc.2009.07.001",
        },
    }


def canonical_gkw_macro_contract() -> dict[str, object]:
    return {
        "schema": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
        "hidden_primary_object": (
            "continuous delta_f_i(t,vpar,mu,s,kx,ky)"
        ),
        "solver_output_fields": list(GKW_MACRO_FIELDS),
        "field_axis_order": [
            "s",
            "kx",
            "ky_nonnegative_half_spectrum",
        ],
        "complex_encoding": ["real", "imag"],
        "definitions": {
            "particle_density": (
                "integral 2*B*intvp*intmu*h over (vpar,mu)"
            ),
            "parallel_momentum": (
                "integral 2*B*intvp*intmu*vpar*h over (vpar,mu)"
            ),
            "kinetic_energy": (
                "integral 2*B*intvp*intmu*(vpar^2+2*B*mu)*h over "
                "(vpar,mu)"
            ),
            "phi": (
                "public GKW adiabatic-electron quasineutrality applied to h"
            ),
        },
        "score_weights": {name: 0.25 for name in GKW_MACRO_FIELDS},
        "truth_evaluation": (
            "analytic linear contraction of spline time controls; no dense "
            "velocity reconstruction per query"
        ),
        "full_state_submission_required": False,
    }


def canonical_gkw_public_metadata(
    *,
    case_id: str,
    time_nodes: Sequence[float],
    byte_count: int,
    source_origin: str = "generated_solution_residual",
) -> dict[str, object]:
    """Build the exact release/evaluator public metadata envelope."""

    return {
        "schema": "fusion_gkw_public_equation_packet_v1",
        "case_id": case_id,
        "operator_id": GKW_OPERATOR_ID,
        "dataset_id": GKW_DATASET_ID,
        "geometry_reference": "pinned_gyaradax_compute_geometry",
        "gyaradax_commit": GKW_GYARADAX_COMMIT,
        "source_access": "metered_numerical_oracle",
        "hidden_parameterization_public": False,
        "continuous_hidden_state_is_primary": True,
        "state_access": "initial_state_and_completed_released_cases_only",
        "array_conventions": dict(GKW_ARRAY_CONVENTIONS),
        "equation_contract": canonical_gkw_equation_contract(),
        "truth_numeric_precision": "float64",
        "public_solver_reference_included": True,
        "parseval_ky_origin": "physical_nonnegative_ky_half_spectrum",
        "legacy_neural_gk_parseval_is_public": False,
        "byte_count": int(byte_count),
        "time_nodes": [float(value) for value in time_nodes],
        "source_origin": source_origin,
        "case_admissibility_passed": True,
        "solver_output_contract_version": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
        "macro_contract": canonical_gkw_macro_contract(),
    }


def validate_gkw_public_components(
    arrays: Mapping[str, Any],
    scalars: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    expected_case_id: str | None = None,
) -> None:
    """Validate all public arrays plus the full recursive metadata contract."""

    import numpy as np

    if set(arrays) != GKW_PUBLIC_ARRAY_NAMES:
        raise ValueError("GKW public packet array keys are not canonical")
    if set(scalars) != GKW_PUBLIC_SCALAR_NAMES:
        raise ValueError("GKW public packet scalar keys are not canonical")
    byte_count = 0
    for name, (shape, dtype_name) in GKW_PUBLIC_ARRAY_SPECS.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != np.dtype(dtype_name):
            raise TypeError(
                f"GKW public array {name} must be {dtype_name} {shape}"
            )
        if value.dtype.kind != "b" and not np.isfinite(value).all():
            raise TypeError(f"GKW public array {name} must be finite")
        byte_count += int(value.nbytes)
    for name, raw in scalars.items():
        value = np.asarray(raw)
        if (
            value.shape != ()
            or value.dtype != np.dtype(np.float64)
            or not np.isfinite(value).all()
        ):
            raise TypeError(f"GKW public scalar {name} must be scalar float64")

    if np.any(np.asarray(arrays["B"]) <= 0.0):
        raise ValueError("GKW public magnetic field must be positive")
    for name in ("intvp", "intmu", "ints"):
        if np.any(np.asarray(arrays[name]) < 0.0):
            raise ValueError(f"GKW public quadrature {name} must be nonnegative")
    ky = np.asarray(arrays["ky"])
    expected_parseval = np.where(np.abs(ky) < 1.0e-12, 1.0, 2.0)
    if np.any(ky < -1.0e-12) or not np.array_equal(
        np.asarray(arrays["parseval_ky"]), expected_parseval
    ):
        raise ValueError("GKW public half-spectrum weights are invalid")

    fixed_scalars = {
        "eps": 0.19,
        "mass_i": 1.0,
        "charge_i": 1.0,
        "temperature_i": 1.0,
        "density_i": 1.0,
        "vth_ratio_i": 1.0,
        "idisp": 2.0,
        "dgrid": 1.0,
        "disp_par": 1.0,
        "disp_vp": 0.2,
        "disp_x": 0.1,
        "disp_y": 0.1,
    }
    for name, expected in fixed_scalars.items():
        if not np.isclose(
            float(scalars[name]), expected, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(f"GKW public regime scalar {name} changed")
    for name in ("ikxspace", "ixzero", "iyzero"):
        value = float(scalars[name])
        if value != round(value):
            raise ValueError(f"GKW public index scalar {name} is not integral")

    plain = _plain_json(metadata)
    if not isinstance(plain, dict):
        raise TypeError("GKW public metadata must be an object")
    if set(plain) != set(
        canonical_gkw_public_metadata(
            case_id="placeholder",
            time_nodes=(0.0, 1.0),
            byte_count=0,
        )
    ):
        raise ValueError("GKW public metadata keys are not canonical")
    case_id = plain.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("GKW public case id must be nonempty text")
    if expected_case_id is not None and case_id != expected_case_id:
        raise ValueError("GKW public packet case id differs from its binding")
    time_nodes = plain.get("time_nodes")
    if (
        not isinstance(time_nodes, list)
        or len(time_nodes) != 4
        or any(isinstance(value, bool) for value in time_nodes)
    ):
        raise ValueError("GKW public time nodes must contain four controls")
    times = np.asarray(time_nodes, dtype=np.float64)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        raise ValueError("GKW public time nodes must increase and be finite")
    source_origin = plain.get("source_origin")
    if source_origin not in {
        "generated_solution_residual",
        "equation_first_numerical_solve",
    }:
        raise ValueError("GKW public source origin changed")
    expected = canonical_gkw_public_metadata(
        case_id=case_id,
        time_nodes=times,
        byte_count=byte_count,
        source_origin=source_origin,
    )
    if plain.get("geometry_reference") == "gyaradax_compute_geometry":
        expected["geometry_reference"] = "gyaradax_compute_geometry"
    if not gkw_json_values_equal_exact(plain, expected):
        raise ValueError("GKW public metadata recursively changed")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise TypeError("GKW public metadata keys must be text")
        return {name: _plain_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("GKW public metadata must be finite JSON")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_json(item())
    raise TypeError("GKW public metadata is not canonical JSON")


def gkw_json_values_equal_exact(left: Any, right: Any) -> bool:
    """Compare canonical JSON recursively while retaining scalar types."""

    left = _plain_json(left)
    right = _plain_json(right)
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
