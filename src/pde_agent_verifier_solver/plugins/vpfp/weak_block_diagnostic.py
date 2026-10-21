"""Complete density/kinetic block interventions at fixed real macro controls.

This is a few-root development diagnostic, not a generator or physics gate.
Changing rho changes A's rho-weighted centering: fixed m does not imply fixed
local velocity or momentum. The full kinetic carrier and its floor survive.
"""
from __future__ import annotations

from hashlib import sha256
import importlib
import json
import time
import traceback

import numpy as np
import torch

from . import weak_dynamics_diagnostic as previous
from .independent_physics import _Runtime, _ratio, audit_independent_physics
from .moving_frame_backend import manufacture_moving_frame_source
from .physical_controls import carrier_hashes
from .value_control_diffusion import extract_value_controls

SCHEMA = "vpfp_weak_complete_block_diagnostic_v1"
ARMS = ("rho_real_kin_real", "rho_gen_kin_real", "rho_real_kin_gen", "rho_gen_kin_gen")


def block_design(*, control_count=33):
    phase = previous.dynamics_design(control_count=control_count)
    return {"schema": SCHEMA, "phase_design": phase, "time_count": phase["time_counts"][-1],
        "arms": list(ARMS), "fixed_macro": "same_real_m_bitwise",
        "blocks": "complete_density_then_complete_kinetic_native_value_coordinates",
        "deep_selection": {"replicas": [0], "strata": [0, 1], "arms": list(ARMS[1:3]), "carrier_count": 4},
        "baseline_reuse": "SHA_bound_completed5287092_129timeline_not_rerun",
        "clean_selection": "eight_original_roots_once_after_bitwise_real_pair_check_across_seeds",
        "statistical_unit": "eight_development_roots_nested_seed_and_block_interventions",
        "local_macro_moments_fixed_claimed": False, "new_training": False,
        "new_sampling": False, "new_PDE_solves": 0, "scientific_admission": False}


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode()
    return sha256(header+b"\0"+array.tobytes()).hexdigest()


def make_block_carriers(decoder, real_carrier, generated_carrier, condition):
    """No normalization/refitting: concatenate two complete stored z blocks."""
    for carrier in (real_carrier, generated_carrier):
        decoder.carrier_to_payload(carrier)
        _Runtime(decoder, carrier, condition, 16)
    if any(real_carrier[k].shape != generated_carrier[k].shape or
           real_carrier[k].dtype != generated_carrier[k].dtype or
           real_carrier[k].device != generated_carrier[k].device for k in ("m", "z")):
        raise ValueError("same native carrier shape/dtype/device required")
    if len(real_carrier["z"]) != 1 or tensor_sha(real_carrier["m"]) != tensor_sha(generated_carrier["m"]):
        raise ValueError("one original root and exactly the same real m required")
    cut = decoder.value_config.rho_dimension
    sources = (real_carrier, generated_carrier)
    source_hashes = [carrier_hashes(decoder, x, condition.reshape(1, 2))[0] for x in sources]
    outputs, receipts = {}, {}
    for arm, (rho_index, kinetic_index) in zip(ARMS, ((0, 0), (1, 0), (0, 1), (1, 1))):
        density = sources[rho_index]["z"][:, :cut]
        kinetic = sources[kinetic_index]["z"][:, cut:]
        carrier = {"m": real_carrier["m"].clone(), "z": torch.cat((density, kinetic), dim=1)}
        decoder.carrier_to_payload(carrier)
        outputs[arm] = carrier
        receipts[arm] = {"arm": arm, "input_carrier_sha256": source_hashes,
            "output_carrier_sha256": carrier_hashes(decoder, carrier, condition.reshape(1, 2))[0],
            "density_source": rho_index, "kinetic_source": kinetic_index,
            "density_range": [0, cut], "kinetic_range": [cut, carrier["z"].shape[1]],
            "density_block_sha256": tensor_sha(density), "kinetic_block_sha256": tensor_sha(kinetic),
            "fixed_m_sha256": tensor_sha(carrier["m"]), "complete_floor_preserved": True,
            "no_Hermite_reprojection": True, "independent_population_sample": False,
            "rho_changes_weighted_center_and_therefore_local_velocity": True,
            "conditional_generated_blocks_are_counterfactual_not_truth_pairs": True}
    if source_hashes != [carrier_hashes(decoder, x, condition.reshape(1, 2))[0] for x in sources]:
        raise ValueError("source carrier mutated during block construction")
    return outputs, receipts


def _runtime(decoder, carrier, condition, design):
    if design != block_design(control_count=decoder.macro_config.control_count):
        raise ValueError("registered complete-block design required")
    return previous._prepare(decoder, carrier, condition, design["phase_design"])


def _time_rms(values, times):
    values = np.asarray(values, dtype=np.float64).reshape(len(times), -1)
    energy = np.mean(values*values, axis=1)
    integral = np.sum((energy[:-1]+energy[1:])*.5*np.diff(times))
    return float(np.sqrt(integral/(times[-1]-times[0])))


def compare_timelines(reference, candidate):
    """Signed same-root counterfactual differences, not pointwise truth error."""
    if reference["count"] != candidate["count"] or not np.array_equal(
            reference["physical_points_txyvxvy"], candidate["physical_points_txyvxvy"]):
        raise ValueError("same actual physical queries required for block comparisons")
    times = np.asarray(reference["times"], dtype=np.float64)
    if not np.array_equal(times, candidate["times"]) or len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError("same strictly increasing physical times required")
    fields = {}
    for key in ("rho", "u", "C", "U", "dt_u", "dt_C", "f", "dt_f", "S", "transport",
                "dt_plus_transport", "electric_acceleration", "nu_collision"):
        a, b = (np.asarray(row["values"][key], dtype=np.float64) for row in (reference, candidate))
        if a.shape != b.shape or a.shape[0] != len(times) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("finite matching signed term arrays required")
        delta = b-a
        fields[key] = {"signed_candidate_minus_anchor": delta,
            "anchor_time_probe_RMS": _time_rms(a, times), "candidate_time_probe_RMS": _time_rms(b, times),
            "difference_time_probe_RMS": _time_rms(delta, times),
            "relative_difference_RMS": _ratio(_time_rms(delta, times), _time_rms(a, times))}
    return previous._finite({"comparison": "same_root_counterfactual_not_truth_error", "fields": fields,
        "vector_convention": "mean_square_over_all_saved_components_not_Euclidean_vector_norm",
        "global_U_drive_proxy_is_independent_source_integral": False,
        "source_algebraic_closure_is_independent_check": False})


def measure_block_carrier(decoder, carrier, condition, *, output, root_id, provenance,
                          design, deep=False, check_deadline=None):
    report = previous._start(output, root_id, provenance, design)
    report.update(schema=SCHEMA, deep_requested=bool(deep))
    started = time.monotonic()
    try:
        runtime, xy, vv = _runtime(decoder, carrier, condition, design)
        digest = carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0]
        report.update(input_carrier_sha256=digest, decoder_identity=decoder.identity(), condition=runtime.condition)
        previous._write(output, report)
        with runtime.measuring():
            report["stage"] = "actual129_macro_and_signed_source_terms"
            report["timeline"] = previous._phase_grid(runtime, xy, vv, design["time_count"], check_deadline)
            previous._write(output, report)
            if deep:
                previous._check(check_deadline)
                phase = design["phase_design"]
                times = runtime.time_domain[0]+np.asarray(phase["deep_time_fractions"])*(runtime.time_domain[1]-runtime.time_domain[0])
                points = np.column_stack((np.repeat(times, len(xy)), np.tile(xy, (len(times), 1))))
                report["stage"] = "preselected_full_velocity_and_coordinate_FD"
                previous._write(output, report)
                report["independent_physics"] = audit_independent_physics(decoder, runtime.answer, runtime.condition,
                    points_txy=points, spatial_times=times, spatial_grids=tuple(phase["deep_spatial_grids"]),
                    relative_fd_step=phase["relative_fd_step"], chunk_size=runtime.chunk_size, include_l2=True)
        if digest != carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0]:
            raise ValueError("block carrier changed during measurement")
        report.update(execution_passed=report.get("independent_physics", {}).get("execution_passed", True) is True,
            stage="complete", independent_physics_selected=bool(deep))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["actual_wall_seconds"] = time.monotonic()-started
    return previous._write(output, report)


def measure_clean_derivatives(decoder, carrier, condition, *, reference_full, output,
                              root_id, provenance, design, check_deadline=None):
    """Actual A autodiff vs original B analytic jets at identical lab queries.

This complements clean VALUE error; it does not compare to raw-PDE truth.
At control knots each native implementation's own one-sided convention is used.
Node/midpoint/quarter subsets are reported separately, never silently pooled.
"""
    report = previous._start(output, root_id, provenance, design)
    report.update(schema=SCHEMA+"_clean_derivatives", fields={})
    started = time.monotonic()
    try:
        if provenance.get("data_origin") != "physical_reference" or provenance.get("view") != "real_m_real_z":
            raise ValueError("same-root real-m/real-z physical reference provenance required")
        runtime, xy, vv = _runtime(decoder, carrier, condition, design)
        original = _Runtime(decoder.base_decoder, reference_full, condition, runtime.chunk_size)
        if tensor_sha(extract_value_controls(original.answer, decoder.value_config, validate_slopes=True)) != tensor_sha(runtime.answer["z"]):
            raise ValueError("clean z must equal all stored native B value coordinates bitwise")
        from .perturbation_backend import _legacy_api
        _legacy_api()
        analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
        times = np.linspace(*runtime.time_domain, design["time_count"])
        points = np.column_stack((np.repeat(times, len(xy)), np.tile(xy, (len(times), 1)), np.tile(vv, (len(times), 1))))
        report.update(input_carrier_sha256=carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0],
            input_B_full_sha256=tensor_sha(original.answer), decoder_identity=decoder.identity(),
            times=times, physical_points_txyvxvy=points, probe_count=len(xy),
            comparison="same-root A(real,real) minus stored B; not raw PDE truth",
            derivative_order=["t", "x", "y", "vx", "vy"], units="physical_coordinates_no_horizon_rescaling",
            node_convention="each_native_decoder_endpoint_or_knot_convention; no_cross_knot_FD",
            independent_PDE_residual=False)
        collected = {role: {} for role in ("A", "B")}
        with runtime.measuring(), original.measuring(), analytic.ieee_float32_coordinate_matmul(original.decoder):
            for offset in range(0, len(points), runtime.chunk_size):
                previous._check(check_deadline)
                raw = points[offset:offset+runtime.chunk_size]
                query = torch.tensor(raw, dtype=torch.float64, device=runtime.device, requires_grad=True)
                with torch.enable_grad():
                    a = manufacture_moving_frame_source(decoder, query, runtime.answer, runtime.condition)
                b = analytic.manufacture_standard_vpfp_source_analytic(original.decoder, raw,
                    original.answer, equation_condition=original.condition)
                for role, values in (("A", a), ("B", b)):
                    grad = values["f_gradient"]
                    physical = torch.as_tensor(raw, device=grad.device, dtype=grad.dtype)
                    terms = {"f": values["f"], "f_gradient": grad, "S": values["source"],
                        "dt_f": grad[:, 0], "transport": (physical[:, 3:5]*grad[:, 1:3]).sum(-1),
                        "electric_acceleration": (values["electric"]*grad[:, 3:5]).sum(-1),
                        "nu_collision": float(runtime.nu)*values["collision"]}
                    terms.update({name: grad[:, axis] for axis, name in enumerate(
                        ("dt_f", "dx_f", "dy_f", "dvx_f", "dvy_f"))})
                    terms["dt_plus_transport"] = terms["dt_f"]+terms["transport"]
                    for key, value in terms.items():
                        collected[role].setdefault(key, []).append(value.detach().double().cpu().numpy())
                report["completed_phase_query_count"] = min(offset+runtime.chunk_size, len(points))
            regions = {"all": np.ones(len(times), dtype=bool),
                "nodes": np.arange(len(times)) % 4 == 0,
                "midpoints": np.arange(len(times)) % 4 == 2,
                "quarters": np.arange(len(times)) % 2 == 1}
            for key in collected["A"]:
                a, b = (np.concatenate(collected[role][key]) for role in ("A", "B"))
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("nonfinite clean A/B derivative or source")
                delta = a-b; summaries = {}
                # t/x/v derivatives have different dimensions; retain ordered
                # joint values but never report a mixed-unit gradient norm.
                for name, mask in (regions.items() if key != "f_gradient" else []):
                    rows = np.repeat(mask, len(xy)); scale = float(np.sqrt(np.mean(b[rows]**2)))
                    error = float(np.sqrt(np.mean(delta[rows]**2)))
                    summaries[name] = {"phase_query_count": int(rows.sum()), "B_RMS": scale,
                        "RMS_difference": error, "relative_RMS": _ratio(error, scale)}
                report["fields"][key] = {"A": a, "B": b, "signed_A_minus_B": delta, "by_time_kind": summaries,
                    "mixed_coordinate_gradient_norm_claimed": False}
        if tensor_sha(original.answer) != report["input_B_full_sha256"] or carrier_hashes(
                decoder, runtime.answer, runtime.condition[None])[0] != report["input_carrier_sha256"]:
            raise ValueError("clean comparison input changed")
        report.update(execution_passed=True, stage="complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["actual_wall_seconds"] = time.monotonic()-started
    return previous._write(output, report)
