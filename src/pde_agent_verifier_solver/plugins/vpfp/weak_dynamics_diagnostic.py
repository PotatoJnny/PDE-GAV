"""Fixed-carrier weak-regime dynamics measurements, not a training objective.

Researchers compare four m/z cross views at the SAME physical root/condition.
Only real-m/real-z may carry a clean B reconstruction comparison.  Manufacturing
S and its algebraic decomposition is not an independent PDE residual test.
The optional, separately budgeted quadrature/FD check supplies that cross-check.
"""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import time
import traceback

import numpy as np
import torch

from . import physical_controls as controls
from .independent_physics import _Runtime, _finite, _ratio, audit_independent_physics
from .moving_frame import MovingFrameDecoder, source_local_moments
from .moving_frame_backend import manufacture_moving_frame_source
from .value_control_diffusion import extract_value_controls

SCHEMA = "vpfp_fixed_carrier_weak_dynamics_v1"


def dynamics_design(*, control_count=33):
    if type(control_count) is not int or control_count < 3:
        raise ValueError("at least three native control nodes required")
    return {"schema": SCHEMA, "time_counts": [2*(control_count-1)+1, 4*(control_count-1)+1],
        "spatial_probe_fractions": [[.173, .389], [.431, .713], [.719, .227], [.857, .563]],
        "physical_velocity_probes": [[0., 0.], [1., 0.], [-1., 0.], [0., 1.]],
        "phase_probe_pairing": "four_fixed_lab_xy_velocity_pairs_not_Cartesian_product",
        "phase_probe_evidence": "sparse_fixed_probes_not_full_space_qualification",
        "deep_time_fractions": [.0703125, .7578125], "deep_spatial_grids": [17, 33],
        "relative_fd_step": 2.**-8, "phase_chunk_size": 16,
        "early_fraction_max": .25, "late_fraction_min": .5,
        "include_velocity_L2": True, "automatic_sample_reduction": False,
        "full_macro_micro_Jacobian_decomposition": "not_implemented",
        "scientific_admission": False}


def _write(path, report):
    report = _finite(report)
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name+".partial")
        pending.write_text(text)
        pending.replace(path)
    return report


def _start(output, root_id, provenance, design):
    if output is not None and (Path(output).exists() or Path(output).is_symlink()):
        raise ValueError("fresh single-JSON diagnostic output required")
    if not isinstance(root_id, str) or not root_id:
        raise ValueError("explicit original root identity required")
    return {"schema": SCHEMA, "root_id": root_id, "provenance": provenance,
        "design": design, "execution_passed": False, "numerically_uncertain": True,
        "scientific_admission": False, "physical_regime_qualified": False,
        "stage": "validate_inputs", "new_training_steps": 0, "new_PDE_solves": 0}


def _prepare(decoder, carrier, condition, design):
    if not isinstance(decoder, MovingFrameDecoder):
        raise ValueError("dynamics cross views require the actual A decoder and complete m/z")
    if design != dynamics_design(control_count=decoder.macro_config.control_count):
        raise ValueError("measurement design differs from the fixed registered protocol")
    runtime = _Runtime(decoder, carrier, condition, design["phase_chunk_size"])
    xy = np.asarray(design["spatial_probe_fractions"])*runtime.domain
    vv = np.asarray(design["physical_velocity_probes"])
    return runtime, xy, vv


def _check(callback):
    if callback is not None:
        callback()


def _dt(values, query):
    """First derivatives only; the constant branch has exactly zero derivative."""
    flat = values.reshape(len(query), -1)
    columns = []
    for i in range(flat.shape[1]):
        if not flat[:, i].requires_grad:
            columns.append(torch.zeros_like(query[:, 0]))
        else:
            derivative = torch.autograd.grad(flat[:, i].sum(), query, retain_graph=True,
                allow_unused=True)[0]
            columns.append(torch.zeros_like(query[:, 0]) if derivative is None else derivative[:, 0])
    return torch.stack(columns, -1).reshape(values.shape)


def _phase_grid(runtime, xy, vv, count, check_deadline):
    """Exact global-U translation chain rule, evaluated at fixed lab probes.

    For F(t,x,w)=f(t,x,w+U(t)), D_t F=f_t+Udot.grad_v f at
    w=v-U(t). This does NOT hold w fixed between these saved lab-probe rows.
    The remainder includes density, local velocity, covariance and micro shape;
    it is not a measurement of pure z dynamics or a full macro/micro split.
    """
    times = np.linspace(*runtime.time_domain, count)
    points = np.column_stack((np.repeat(times, len(xy)), np.tile(xy, (count, 1)), np.tile(vv, (count, 1))))
    collected = {}
    for offset in range(0, len(points), runtime.chunk_size):
        _check(check_deadline)
        q = torch.tensor(points[offset:offset+runtime.chunk_size], device=runtime.device,
            dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            # Derive macro jets before the source helper frees its own graph.
            macros = runtime.decoder.fields(q, runtime.answer, equation_condition=runtime.condition)
            u, U, C = (macros[k] for k in ("velocity", "global_velocity", "central_covariance"))
            dU, du, dC = (_dt(v, q) for v in (U, u, C))
            source = manufacture_moving_frame_source(runtime.decoder, q, runtime.answer, runtime.condition)
        gradient = source["f_gradient"]
        dt = gradient[:, 0]
        transport = (q[:, 3:5]*gradient[:, 1:3]).sum(-1)
        acceleration = (source["electric"]*gradient[:, 3:5]).sum(-1)
        collision = runtime.nu*source["collision"]
        translation = -(dU*gradient[:, 3:5]).sum(-1)
        terms = {"rho": source["rho"], "u": u, "C": C, "U": U, "dt_U": dU,
            "dt_u": du, "dt_C": dC, "f": source["f"], "S": source["source"],
            "dt_f": dt, "transport": transport, "electric_acceleration": acceleration,
            "nu_collision": collision, "dt_plus_transport": dt+transport,
            "left_without_collision": dt+transport+acceleration,
            "source_algebraic_closure": dt+transport+acceleration-collision-source["source"],
            "global_U_translation_dt_f": translation, "U_comoving_dt_remainder": dt-translation,
            "U_damping": runtime.nu*U, "global_U_drive_proxy": dU+runtime.nu*U}
        for key, value in terms.items():
            collected.setdefault(key, []).append(value.detach().double().cpu().numpy())
    values = {k: np.concatenate(v).reshape(count, len(xy), *v[0].shape[1:]) for k, v in collected.items()}
    spectra = {k: controls._time_spectrum(v.reshape(count, -1), times) for k, v in values.items()}
    mean_squares = {k: controls._derivative_energy(v.reshape(count, -1), times) for k, v in values.items()}
    source_rms = np.sqrt(np.mean(values["S"]**2, axis=1))
    f_rms = np.sqrt(np.mean(values["f"]**2, axis=1))
    drive = np.sqrt(np.mean(np.sum(values["global_U_drive_proxy"]**2, axis=-1), axis=1))
    collision_rms = np.sqrt(np.mean(values["nu_collision"]**2, axis=1))
    fractions = (times-runtime.time_domain[0])/(runtime.time_domain[1]-runtime.time_domain[0])
    regions = {}
    for name, mask in (("early_fraction_le_0_25", fractions <= .25), ("late_fraction_ge_0_5", fractions >= .5)):
        regions[name] = {"registered_time_count": int(mask.sum()),
            "source_over_nu_f_by_time": _ratio(source_rms[mask], runtime.nu*f_rms[mask]),
            "source_over_actual_collision_probe_RMS_by_time": _ratio(source_rms[mask], collision_rms[mask]),
            "global_drive_over_nu_sqrtT_by_time": _ratio(drive[mask], runtime.nu*np.sqrt(runtime.temperature))}
    return {"count": count, "times": times, "physical_points_txyvxvy": points,
        "probe_count": len(xy), "physical_spatial_probes": xy, "physical_velocity_probes": vv,
        "point_order": "time_major_then_registered_paired_probe_index",
        "probe_geometry": "fixed_paired_sparse_lab_probes_not_Cartesian_product_or_full_space_integral",
        "values": values, "spectra": spectra, "time_mean_square_integrals": mean_squares,
        "early_late": regions,
        "global_U_is_rho_weighted_mean": "native_exact_Fourier_Parseval_construction_not_independent_grid_measurement",
        "global_drive_proxy": "Udot+nu*U; periodic mean-rho=rho0 and mean(rho E)=0 imply mean(Sj)/rho0; the global source spatial integral is not independently measured here",
        "decomposition": "f_t = -Udot dot grad_v(f) + U_comoving_dt_remainder; signed terms cancel; energies are not additive",
        "remainder_is_pure_micro": False, "fixed_w_trajectory_measured": False,
        "source_closure_is_independent_check": False, "numerically_uncertain": True}


def measure_dynamics_carrier(decoder, carrier, condition, *, output, root_id, provenance,
        design, check_deadline=None, reference_full=None, deep=False):
    """One complete cross-view carrier; caller fixes roots/views/deep membership.

    No threshold, sampling, PDE solve or model fit. Deadline callbacks are checked
    between passes/chunks; an in-flight legacy deep call needs the driver's hard
    process deadline. Completed reports remain when a later stage fails.
    """
    report = _start(output, root_id, provenance, design)
    started = time.monotonic()
    try:
        runtime, xy, vv = _prepare(decoder, carrier, condition, design)
        report.update(decoder_identity=decoder.identity(), condition=runtime.condition,
            input_carrier_sha256=controls.carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0],
            deep_requested=bool(deep), timeline=[], pending=["full_macro_micro_Jacobian_decomposition",
                "independent_spatial_mean_source_momentum", "uniform_space_time_convergence",
                "population_physical_qualification"])
        _write(output, report)
        with runtime.measuring():
            _check(check_deadline)
            report["stage"] = "native_temporal_frequency"
            _write(output, report)
            report["native_temporal_frequency"] = controls.temporal_frequency_diagnostics(
                decoder, runtime.answer, runtime.condition[None], root_ids=[root_id], provenance=provenance,
                spatial_probes=xy, velocity_probes=vv, time_counts=tuple(design["time_counts"]),
                chunk_size=runtime.chunk_size)
            _write(output, report)
            report["stage"] = "physical_terms_and_global_U_decomposition"
            for count in design["time_counts"]:
                _check(check_deadline)
                report["timeline"].append(_phase_grid(runtime, xy, vv, count, check_deadline))
                _write(output, report)
            if reference_full is not None:
                _check(check_deadline)
                report["clean_compression"] = measure_clean_compression(decoder, runtime.answer, runtime.condition,
                    output=None, root_id=root_id, provenance=provenance, design=design,
                    check_deadline=check_deadline, reference_full=reference_full)
            if deep:
                report["stage"] = "independent_full_velocity_and_coordinate_FD"
                _write(output, report)
                _check(check_deadline)
                t = runtime.time_domain[0]+np.asarray(design["deep_time_fractions"])*(runtime.time_domain[1]-runtime.time_domain[0])
                p = np.column_stack((np.repeat(t, len(xy)), np.tile(xy, (len(t), 1))))
                report["independent_physics"] = audit_independent_physics(decoder, runtime.answer, runtime.condition,
                    points_txy=p, spatial_times=t, spatial_grids=tuple(design["deep_spatial_grids"]),
                    relative_fd_step=design["relative_fd_step"], chunk_size=runtime.chunk_size,
                    include_l2=design["include_velocity_L2"])
            else:
                report["pending"].append("independent_full_velocity_and_coordinate_FD_not_selected_for_this_view")
        if controls.carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0] != report["input_carrier_sha256"]:
            raise ValueError("input carrier changed during measurement")
        complete = report["native_temporal_frequency"]["execution_passed"]
        for key in ("clean_compression", "independent_physics"):
            if key in report:
                complete = complete and report[key]["execution_passed"]
        report.update(execution_passed=bool(complete), stage="complete" if complete else "measurement_failure")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["actual_wall_seconds"] = time.monotonic()-started
    return _write(output, report)


def measure_clean_compression(decoder, carrier, condition, *, reference_full, output,
        root_id, provenance, design, check_deadline=None):
    """Actual A(real,real) versus original native B at identical phase points.

    Caller MUST mark this as a clean real/real reconstruction. Differences of
    crossed/generated views from an unrelated root are not reconstruction error.
    """
    report = _start(output, root_id, provenance, design)
    started = time.monotonic()
    try:
        if provenance.get("data_origin") != "physical_reference" or provenance.get("view") != "real_m_real_z":
            raise ValueError("clean compression requires physical_reference real_m_real_z provenance")
        runtime, xy, vv = _prepare(decoder, carrier, condition, design)
        original = _Runtime(decoder.base_decoder, reference_full, condition, runtime.chunk_size)
        if not torch.equal(extract_value_controls(original.answer, decoder.value_config, validate_slopes=True), runtime.answer["z"]):
            raise ValueError("clean A z must be the exact native values of the same stored B answer")
        report.update(grids=[], comparison="same-root A(real,real) minus stored native B, not raw-PDE truth",
            input_B_full_sha256=sha256(original.answer.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            input_carrier_sha256=controls.carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0])
        with runtime.measuring():
            for count in design["time_counts"]:
                times = np.linspace(*runtime.time_domain, count)
                p = np.column_stack((np.repeat(times, len(xy)), np.tile(xy, (count, 1)), np.tile(vv, (count, 1))))
                a, b = {}, {}
                for offset in range(0, len(p), runtime.chunk_size):
                    _check(check_deadline)
                    q = torch.tensor(p[offset:offset+runtime.chunk_size], device=runtime.device, dtype=torch.float64)
                    with torch.no_grad():
                        av = decoder.fields(q, runtime.answer, equation_condition=runtime.condition)
                        bv = decoder.base_decoder.fields(q, original.answer, equation_condition=runtime.condition)
                        bm = source_local_moments(bv, decoder.base_decoder)
                    for key, x, y in (("f", av["f"], bv["f"]), ("rho", av["rho"], bv["rho"]),
                            ("u", av["velocity"], bm["mean"]), ("C", av["central_covariance"], bm["covariance"])):
                        a.setdefault(key, []).append(x.detach().double().cpu().numpy())
                        b.setdefault(key, []).append(y.detach().double().cpu().numpy())
                fields = {}
                for key in a:
                    left, right = np.concatenate(a[key]), np.concatenate(b[key])
                    delta = left-right
                    error, scale = np.sqrt(np.mean(delta**2)), np.sqrt(np.mean(right**2))
                    fields[key] = {"A": left, "B": right, "signed_difference": delta,
                        "RMS_difference": float(error), "B_RMS": float(scale), "relative_RMS": _ratio(error, scale)}
                report["grids"].append({"count": count, "times": times, "fields": fields,
                    "physical_points_txyvxvy": p, "probe_count": len(xy),
                    "point_order": "time_major_then_registered_paired_probe_index",
                    "probe_geometry": "fixed_paired_sparse_lab_probes_not_Cartesian_product_or_full_space_integral"})
                _write(output, report)
        if sha256(original.answer.detach().cpu().contiguous().numpy().tobytes()).hexdigest() != report["input_B_full_sha256"]:
            raise ValueError("stored B input changed during clean measurement")
        if controls.carrier_hashes(decoder, runtime.answer, runtime.condition[None])[0] != report["input_carrier_sha256"]:
            raise ValueError("clean A carrier changed during measurement")
        report.update(execution_passed=True, stage="complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["actual_wall_seconds"] = time.monotonic()-started
    return _write(output, report)
