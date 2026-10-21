"""Canonical release package matching stable GKW commit 6947a64."""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

from pde_agent_verifier_solver.plugins.gkw.contract import GKW_MACRO_FIELDS
from pde_agent_verifier_solver.plugins.gkw.public_contract import (
    GKW_FIELD_SHAPE,
    GKW_OPERATOR_ID,
    GKW_PUBLIC_ARRAY_SPECS,
    GKW_PUBLIC_SCALAR_NAMES,
    GKW_STATE_SHAPE,
    canonical_gkw_public_metadata,
)
from pde_agent_verifier_solver.plugins.gkw.release_io import (
    GKW_RELEASED_MACRO_ANSWER_SCHEMA,
)


def canonical_gkw_public_components(case_id: str):
    """Return a full, mutable canonical packet for unit and bridge tests."""

    import numpy as np

    arrays = {
        name: np.ones(shape, dtype=dtype)
        for name, (shape, dtype) in GKW_PUBLIC_ARRAY_SPECS.items()
    }
    arrays.update(
        {
            "vpar": np.linspace(-3.0, 3.0, 32, dtype=np.float64),
            "intvp": np.ones(32, dtype=np.float64),
            "mu": np.linspace(0.0, 3.0, 8, dtype=np.float64),
            "intmu": np.ones(8, dtype=np.float64),
            "s": np.linspace(-0.5, 0.5, 16, dtype=np.float64),
            "ints": np.ones(16, dtype=np.float64),
            "B": np.ones(16, dtype=np.float64),
            "kx": np.arange(-42, 43, dtype=np.float64),
            "ky": np.arange(32, dtype=np.float64),
            "parseval_ky": np.concatenate(
                (
                    np.ones(1, dtype=np.float64),
                    np.full(31, 2.0, dtype=np.float64),
                )
            ),
        }
    )
    arrays["B"][:2] = (2.0, 4.0)
    arrays["ints"][:2] = (1.0, 0.5)
    for name in ("mode_label", "ixplus", "ixminus", "s_shift", "kx_shift"):
        arrays[name].fill(0)
    arrays["pos_parallel_class"].fill(0)
    arrays["valid_shift"].fill(False)

    scalars = {name: 1.0 for name in GKW_PUBLIC_SCALAR_NAMES}
    scalars.update(
        {
            "eps": 0.19,
            "ikxspace": 5.0,
            "krhomax": 1.4,
            "dvp": 0.1875,
            "ds": 0.0625,
            "idisp": 2.0,
            "dgrid": 1.0,
            "ixzero": 42.0,
            "iyzero": 0.0,
            "disp_par": 1.0,
            "disp_vp": 0.2,
            "disp_x": 0.1,
            "disp_y": 0.1,
        }
    )
    time_nodes = (0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0)
    metadata = canonical_gkw_public_metadata(
        case_id=case_id,
        time_nodes=time_nodes,
        byte_count=sum(value.nbytes for value in arrays.values()),
    )
    return arrays, scalars, metadata


def structured_truth_construction() -> dict[str, object]:
    controls = 4
    return {
        "schema": "fusion_gkw_structured_macro_truth_v1",
        "method": "linear_functional_contraction_of_spline_time_controls",
        "time_control_count": controls,
        "fused_velocity_contraction_calls_per_case": 1,
        "velocity_control_slices_contracted": controls,
        "velocity_functional": (
            "declared_gkw_semidiscrete_intvp_intmu_weights"
        ),
        "dense_velocity_reconstruction_per_query": False,
        "velocity_requadrature_per_query": False,
        "quasineutrality_applied_to_time_controls_once": True,
        "quasineutrality_method": (
            "direct_phi_weight_contraction_on_truncated_controls"
        ),
        "full_five_dimensional_control_padding_avoided": True,
        "arbitrary_time_evaluation": "natural_cubic_control_interpolation",
        "linearity_identity_used": "L(sum_i c_i h_i)=sum_i c_i L(h_i)",
        "dense_state_elements_per_time_avoided": prod(GKW_STATE_SHAPE),
        "macro_cache_complex_elements": (
            controls * len(GKW_MACRO_FIELDS) * prod(GKW_FIELD_SHAPE)
        ),
        "hidden_state_remains_primary": True,
    }


def write_gkw_release(case_dir: Path, case_id: str, scale: float) -> None:
    import numpy as np

    case_dir.mkdir()
    arrays, scalars, metadata = canonical_gkw_public_components(case_id)
    np.savez(
        case_dir / "case_public.npz",
        **{f"array__{name}": value for name, value in arrays.items()},
        **{
            f"scalar__{name}": np.asarray(value, dtype=np.float64)
            for name, value in scalars.items()
        },
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    oracle = {
        "arbitrary_phase_subsets_supported": True,
        "arbitrary_time_queries_supported": True,
        "case_id": case_id,
        "complex_value_encoding": ["real", "imag"],
        "equation_view": "continuous_time_fixed_phase_grid_method_of_lines",
        "field_axis_order": ["s", "kx", "ky"],
        "field_grid_shape": list(GKW_FIELD_SHAPE),
        "fixed_phase_coordinates_are_in_public_packet": True,
        "generator_parameters_public": False,
        "hidden_parameterization_public": False,
        "limits": {
            "max_rows_per_request": 65536,
            "max_total_rows": 2000000,
            "max_unique_source_times": 256,
            "source_frame_cache_entries": 1,
        },
        "off_grid_phase_values_are_not_part_of_the_semidiscrete_equation": True,
        "phase_axis_order": ["vpar", "mu", "s", "kx", "ky"],
        "phase_grid_shape": list(GKW_STATE_SHAPE),
        "queries": {
            "initial_h": {
                "inputs": {"phase_indices": "int64 [N,5]"},
                "output": "float64 [N,2]",
            },
            "initial_phi": {
                "inputs": {"field_indices": "int64 [N,3]"},
                "output": "float64 [N,2]",
            },
            "source": {
                "definition": "S_balance = d_t h_hidden - R_GKW[h_hidden]",
                "inputs": {
                    "phase_indices": "int64 [N,5]",
                    "times": "float64 scalar or [N]",
                },
                "output": "float64 [N,2]",
            },
        },
        "schema": "fusion_gkw_numerical_oracle_v1",
        "time_interval_closed": [0.5, 1.0],
        "truth_precision": "float64",
    }
    (case_dir / "oracle_protocol.json").write_text(
        json.dumps(oracle, sort_keys=True), encoding="utf-8"
    )

    times = np.asarray([0.5, 1.0], dtype=np.float64)
    indices = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    value = np.zeros((2, 2, 2), dtype=np.float64)
    value[..., 0] = scale
    answer_metadata = {
        "schema": GKW_RELEASED_MACRO_ANSWER_SCHEMA,
        "case_id": case_id,
        "fields": list(GKW_MACRO_FIELDS),
        "complex_value_encoding": ["real", "imag"],
        "field_axis_order": ["s", "kx", "ky_nonnegative_half_spectrum"],
        "sample_seed_public": 7,
        "continuous_hidden_parameters_included": False,
        "macro_truth_construction": structured_truth_construction(),
    }
    np.savez(
        case_dir / "released_answer.npz",
        times=times,
        field_indices=indices,
        **{f"{name}_values": value.copy() for name in GKW_MACRO_FIELDS},
        metadata_json=np.asarray(json.dumps(answer_metadata, sort_keys=True)),
    )
    (case_dir / "README.md").write_text(
        "stable GKW test release\n", encoding="utf-8"
    )

    payload_names = (
        "case_public.npz",
        "oracle_protocol.json",
        "released_answer.npz",
        "README.md",
    )
    files = []
    for name in payload_names:
        data = (case_dir / name).read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "built_from_transient_observation": True,
        "case_id": case_id,
        "completed_before_release": True,
        "continuous_hidden_parameters_included": False,
        "decoder_included": False,
        "field_row_count": len(indices),
        "files": files,
        "full_hidden_state_included": False,
        "generator_latent_included": False,
        "generator_parameters_included": False,
        "hidden_state_artifact_written": False,
        "macro_fields": list(GKW_MACRO_FIELDS),
        "numeric_representation": "sampled_macroscopic_linear_functionals",
        "operator_id": GKW_OPERATOR_ID,
        "phase_row_count": 0,
        "schema": GKW_RELEASED_MACRO_ANSWER_SCHEMA,
        "structure_aware_truth_contraction": True,
        "time_count": len(times),
        "total_bytes": sum(row["bytes"] for row in files),
        "truth_precision": "float64",
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
