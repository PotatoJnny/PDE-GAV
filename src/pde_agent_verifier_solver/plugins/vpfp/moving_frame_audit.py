"""Direct A-carrier physics measurements in the existing v2 feature schema.

A is the complete continuous affine pushforward, including the whole floor.
Only feature REDUCTIONS and physical unit definitions are shared with B. A's
moments and derivatives come from its own fields; old Hermite moments/source
are never substituted. This is a fixed-probe development diagnostic, not
full-velocity source quadrature, physical calibration or admission.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import importlib
import json
import os

import numpy as np
import torch

from .diffusion_features import _check_conditions
from .moving_frame import MovingFrameDecoder
from .regime_fidelity_coverage import (
    BUDGET_AXES, VPFPRegimeFeaturePlan, _MACRO_NAMES, _PROBE_NAMES,
    _collision_budgets, _json_digest, _rms)
from .value_control_diffusion import extract_value_controls, lift_value_controls

SCHEMA = "vpfp_direct_moving_frame_regime_measurements_v1"


def carrier_input_sha256(full_latent, macro_controls, condition, identity):
    """Versioned A identity binds the full native B bytes AND physical m."""
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(SCHEMA.encode()+b"\0"+sha256(identity_bytes).digest()
        +np.asarray(full_latent, dtype="<f4").tobytes()
        +np.asarray(macro_controls, dtype="<f4").tobytes()
        +np.asarray(condition, dtype="<f4").tobytes()).hexdigest()


def _backend():
    # The shared A MMS implementation belongs to the A backend, not a training
    # script. Deferred import keeps plain feature-schema imports lightweight.
    return importlib.import_module(__package__+".moving_frame_backend")


def _physical_macro_chunk(decoder, txy, carrier, condition):
    parameter = next(decoder.parameters())
    query = torch.as_tensor(txy, device=parameter.device, dtype=torch.float64).detach().clone().requires_grad_(True)
    with torch.enable_grad():
        fields = _backend().evaluate_moving_frame_macros(decoder, query, carrier, condition)
        rho, momentum, kinetic = fields["rho"], fields["momentum"], fields["kinetic_energy"]
        derivatives = [torch.autograd.grad(value.sum(), query, retain_graph=True, create_graph=False)[0][:, 0]
                       for value in (momentum[:, 0], momentum[:, 1], kinetic)]
        # Use the represented C directly: raw_second/rho-u*u catastrophically
        # cancels a small thermal spread in a large coherent drift.
        covariance = fields["central_covariance"]
        trace = covariance[:, 0, 0]+covariance[:, 1, 1]
        anisotropy = torch.sqrt((covariance[:, 0, 0]-covariance[:, 1, 1]).square()
                               +4*covariance[:, 0, 1].square())/condition[1]
        values = {"rho": rho, "jx": momentum[:, 0], "jy": momentum[:, 1], "K": kinetic,
            "dt_jx": derivatives[0], "dt_jy": derivatives[1], "dt_K": derivatives[2],
            "electric": fields["electric"], "thermal_energy": .5*trace, "thermal_anisotropy": anisotropy}
    if any(not bool(torch.isfinite(value).all()) for value in values.values()) or bool((rho <= 0).any()):
        raise ValueError("A macro fields are nonfinite or density is nonpositive")
    return {name: value.detach().double().cpu().numpy() for name, value in values.items()}


def _grid_measure(decoder, carrier, condition, plan, size):
    """Same registered v2 reduction, using only actual A macro fields."""
    cfg = decoder.base_decoder.config
    bg, temperature = float(cfg.equation.background_density), float(condition[1])
    horizon, length = float(cfg.time_domain[1]-cfg.time_domain[0]), np.sqrt(np.prod(cfg.equation.x_domain))
    times = cfg.time_domain[0]+np.asarray(plan.time_fractions)*horizon
    xy = [np.arange(size)*side/size for side in cfg.equation.x_domain]
    points = np.stack(np.meshgrid(times, *xy, indexing="ij"), axis=-1).reshape(-1, 3)
    collected = {}
    for start in range(0, len(points), plan.chunk_size):
        values = _physical_macro_chunk(decoder, points[start:start+plan.chunk_size], carrier, condition)
        for key, value in values.items(): collected.setdefault(key, []).append(value)
    fields = {key: np.concatenate(chunks).reshape(len(times), size, size, *chunks[0].shape[1:])
              for key, chunks in collected.items()}
    frequency = np.fft.fftfreq(size)*size
    radius2 = frequency[:, None]**2+frequency[None]**2
    bands = ((radius2 > 0) & (radius2 <= 4), (radius2 > 4) & (radius2 <= 16), radius2 > 16)
    jscale, escale, rows = bg*np.sqrt(temperature), bg*temperature, []
    for index in range(len(times)):
        v = {key: value[index] for key, value in fields.items()}
        rho, jx, jy, energy, electric = (v[key] for key in ("rho", "jx", "jy", "K", "electric"))
        source_j = np.asarray([np.mean(v["dt_jx"]+float(condition[0])*jx-rho*electric[..., 0]),
            np.mean(v["dt_jy"]+float(condition[0])*jy-rho*electric[..., 1])])*horizon/jscale
        source_k = np.mean(v["dt_K"]-electric[..., 0]*jx-electric[..., 1]*jy
            +2*float(condition[0])*(energy-temperature*rho))*horizon/escale
        row = [_rms(rho-bg)/bg, jx.mean()/jscale, jy.mean()/jscale,
            np.hypot(jx.mean(), jy.mean())/jscale, np.sqrt(jx.var()+jy.var())/jscale,
            v["thermal_energy"].mean()/temperature, v["thermal_energy"].std()/temperature,
            np.sqrt(np.mean(np.sum(electric**2, axis=-1)))*length/temperature,
            np.sqrt(np.mean(v["dt_jx"]**2+v["dt_jy"]**2))*horizon/jscale,
            _rms(v["dt_K"])*horizon/escale, np.linalg.norm(source_j), abs(source_k)]
        for arrays, scale in (((rho,), bg), ((jx, jy), jscale), ((energy,), escale)):
            power = sum(np.abs(np.fft.fft2(array)/(size*size*scale))**2 for array in arrays)
            row.extend(float(power[band].sum()) for band in bands)
        row.append(float(v["thermal_anisotropy"].mean()))
        rows.append(row)
    return np.asarray(rows), {"grid_size": size, "minimum_density": float(fields["rho"].min()),
        "minimum_thermal_energy": float(fields["thermal_energy"].min()),
        "macro_source": "A_continuous_fields_exact_mean_covariance_not_B_Hermite_coefficients",
        "thermal_covariance": "direct_target_C_no_raw_second_minus_mean_squared_cancellation",
        "net_injection_method": "periodic_spatial_mean_macro_balance_identity_not_independent_source_quadrature",
        "spectral_bands": "radial_integer_wave_number: (0,2],(2,4],(4,grid_limit]; different high-grid support",
        "A_Hermite_tail_measured": False}


def _finite_difference_source(decoder, points, carrier, condition, steps):
    """Independent coordinate-difference source assembly from actual A f.

    The f evaluator is shared but derivative algorithm is independent. Time
    endpoints use second-order one-sided differences; space is periodic.
    This is neither exact nor a convergence certificate.
    """
    q = points.detach().double()
    cfg = decoder.base_decoder.config
    with torch.no_grad():
        base = decoder.fields(q, carrier, equation_condition=condition)
        f = base["f"]
        gradients, laplacian = [], torch.zeros_like(f)
        for axis, step in enumerate(steps):
            plus, minus = q.clone(), q.clone()
            plus[:, axis] += step; minus[:, axis] -= step
            left = q[:, 0]-step < cfg.time_domain[0] if axis == 0 else torch.zeros(len(q), dtype=torch.bool, device=q.device)
            right = q[:, 0]+step > cfg.time_domain[1] if axis == 0 else torch.zeros_like(left)
            if axis == 0:
                minus[left, axis] = q[left, axis]; plus[right, axis] = q[right, axis]
            elif axis in (1, 2):
                plus[:, axis] = plus[:, axis].remainder(cfg.equation.x_domain[axis-1])
                minus[:, axis] = minus[:, axis].remainder(cfg.equation.x_domain[axis-1])
            fp = decoder.fields(plus, carrier, equation_condition=condition)["f"]
            fm = decoder.fields(minus, carrier, equation_condition=condition)["f"]
            derivative = (fp-fm)/(2*step)
            if bool(left.any() or right.any()):
                shifted = q.clone()
                shifted[left, axis] += 2*step; shifted[right, axis] -= 2*step
                f2 = decoder.fields(shifted, carrier, equation_condition=condition)["f"]
                derivative[left] = (-3*f[left]+4*fp[left]-f2[left])/(2*step)
                derivative[right] = (3*f[right]-4*fm[right]+f2[right])/(2*step)
            gradients.append(derivative)
            if axis in (3, 4): laplacian += (fp-2*f+fm)/(step*step)
        gradient = torch.stack(gradients, -1)
        collision = 2*f+(q[:, 3:5]*gradient[:, 3:5]).sum(-1)+condition[1]*laplacian
        source = gradient[:, 0]+(q[:, 3:5]*gradient[:, 1:3]).sum(-1)
        source += (base["electric"]*gradient[:, 3:5]).sum(-1)-condition[0]*collision
    return source


def _probe_measure(decoder, carrier, condition, plan):
    cfg, parameter = decoder.base_decoder.config, next(decoder.parameters())
    bg, temperature = float(cfg.equation.background_density), float(condition[1])
    horizon, length = float(cfg.time_domain[1]-cfg.time_domain[0]), np.sqrt(np.prod(cfg.equation.x_domain))
    rng = np.random.default_rng(plan.probe_seed)
    count = plan.phase_probes_per_time
    xy = rng.uniform(size=(count, 2))*np.asarray(cfg.equation.x_domain)
    velocity = rng.normal(size=(count, 2))*np.where(np.arange(count)[:, None] % 2, 3., 1.)*np.sqrt(temperature)
    times = cfg.time_domain[0]+np.asarray(plan.time_fractions)*horizon
    points = np.concatenate([np.column_stack((np.full(count, t), xy, velocity)) for t in times])
    rows, sources, algebraic = [], [], []
    for start in range(0, len(points), plan.chunk_size):
        p = points[start:start+plan.chunk_size]
        query = torch.as_tensor(p, dtype=torch.float64, device=parameter.device)
        with torch.enable_grad():
            values = _backend().manufacture_moving_frame_source(decoder, query, carrier, condition, create_graph=False)
        f, source, gradient, electric, collision = [values[key].detach().double().cpu().numpy()
            for key in ("f", "source", "f_gradient", "electric", "collision")]
        dt, transport = gradient[:, 0], (p[:, 3:5]*gradient[:, 1:3]).sum(-1)
        acceleration, coll = (electric*gradient[:, 3:5]).sum(-1), float(condition[0])*collision
        row = np.column_stack((f*temperature/bg, source*temperature*horizon/bg,
            dt*temperature*horizon/bg, transport*temperature*horizon/bg,
            acceleration*temperature*horizon/bg, coll*temperature*horizon/bg,
            np.linalg.norm(gradient[:, 1:3], axis=-1)*length*temperature/bg,
            np.linalg.norm(gradient[:, 3:5], axis=-1)*temperature**1.5/bg))
        if not np.isfinite(row).all() or np.any(f < 0):
            raise ValueError("nonfinite A source/derivatives or negative f at fixed probes")
        rows.append(row); sources.append(source)
        algebraic.extend(((dt+transport+acceleration-coll-source)*temperature*horizon/bg).tolist())
    per_probe = np.concatenate(rows).reshape(len(times), count, 8)
    feature = np.column_stack((_rms(per_probe, axis=1), np.quantile(np.abs(per_probe[:, :, 1]), .99, axis=1)))
    independent = {"execution_passed": False, "physical_qualification": False,
        "method": "A_f_coordinate_finite_difference_h_and_h2_vs_A_autograd_source",
        "independent_derivative_algorithm_same_density_evaluator": True,
        "is_convergence_certificate": False}
    try:
        indices = np.unique(np.linspace(0, len(points)-1, min(plan.independent_source_probe_count, len(points))).astype(int))
        q = torch.as_tensor(points[indices], device=parameter.device, dtype=torch.float64)
        # Source rho/amplitude coordinate features retain native FP32; the
        # affine velocity density is evaluated in FP64. Steps follow those
        # arithmetic scales, not observed generated errors or physics gates.
        h32, h64 = np.finfo(np.float32).eps**(1/3), np.finfo(np.float64).eps**.25
        steps = np.asarray([min(h32, .25/(cfg.structured_time_control_count-1))*horizon,
            h32*cfg.equation.x_domain[0], h32*cfg.equation.x_domain[1], h64*np.sqrt(temperature), h64*np.sqrt(temperature)])
        expected = np.concatenate(sources)[indices]
        measured = [_finite_difference_source(decoder, q, carrier, condition, steps/divisor).detach().cpu().numpy()
                    for divisor in (1., 2.)]
        differences = [(value-expected)*temperature*horizon/bg for value in measured]
        if not all(np.isfinite(value).all() for value in differences):
            raise ValueError("nonfinite A finite-difference source comparison")
        independent.update(execution_passed=True, probe_indices=indices.tolist(), physical_steps_h=steps.tolist(),
            maximum_dimensionless_difference_h_h2=[float(np.abs(value).max()) for value in differences],
            rms_dimensionless_difference_h_h2=[_rms(value) for value in differences],
            maximum_dimensionless_step_sensitivity=float(np.abs(differences[1]-differences[0]).max()),
            smaller_step_is_not_assumed_more_accurate=True)
    except Exception as error:
        independent["failure"] = {"type": type(error).__name__, "message": str(error)}
    return feature, {"scope": "fixed_spatial_velocity_probes_not_full_space_L2_or_velocity_quadrature",
        "points_txyvv": points.tolist(), "algebraic_balance_maximum": float(np.max(np.abs(algebraic))),
        "algebraic_balance_is_independent_check": False, "independent_source_check": independent}


def extract_moving_frame_regime_features(decoder, full_latents, macro_controls, conditions, *, plan=None,
        progress=None, expected_identity=None):
    """Same feature schema as old native v2, but evaluate actual A(m,z).

    full_latents are the persisted FP32 B carrier. Extracting z requires an
    exact native lift replay; no projection of a malformed full answer occurs.
    All attempted rows remain in the report, including failed A decodes.
    """
    if not isinstance(decoder, MovingFrameDecoder): raise TypeError("explicit MovingFrameDecoder required")
    plan, cfg = plan or VPFPRegimeFeaturePlan(), decoder.base_decoder.config
    parameter, identity = next(decoder.parameters()), decoder.identity()
    plan.validate(cfg)
    if min(plan.grid_sizes) < 2*decoder.macro_config.max_mode+1:
        raise ValueError("A macro coordinates alias the registered spatial grid")
    if parameter.dtype != torch.float32: raise ValueError("frozen A adapter requires native FP32 source decoder")
    if parameter.device.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU extraction requires a compute allocation")
    if expected_identity is not None and expected_identity != identity: raise ValueError("A carrier decoder identity mismatch")
    raw, macros, conditions = np.asarray(full_latents), np.asarray(macro_controls), np.asarray(conditions)
    if (raw.ndim != 2 or raw.shape[1] != cfg.latent_dim or macros.shape != (len(raw), decoder.macro_config.latent_dimension)
            or conditions.shape != (len(raw), 2) or any(value.dtype != np.dtype("float32") for value in (raw, macros, conditions))):
        raise ValueError("aligned native FP32 full coefficients, physical macro controls and nu/T required")
    descriptors = [{"name": f"time{index}.{name}", "unit": "dimensionless",
        "group": "operator_fixed_probes" if name in _PROBE_NAMES else "macro_spatial", "time_fraction": fraction}
        for index, fraction in enumerate(plan.time_fractions) for name in _MACRO_NAMES+_PROBE_NAMES]
    matrix = np.zeros((len(raw), len(descriptors)), dtype=float)
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    cases, was_training = [], decoder.training
    decoder.eval()
    try:
        with analytic.ieee_float32_coordinate_matmul(decoder.base_decoder):
            for index in range(len(raw)):
                full = np.array(raw[index], dtype="<f4", copy=True)
                m, condition = np.array(macros[index], dtype="<f4", copy=True), np.array(conditions[index], dtype="<f4", copy=True)
                entry = {"row": index, "input_answer_sha256": carrier_input_sha256(full, m, condition, identity),
                    "source_B_answer_sha256": sha256(full.tobytes()+condition.tobytes()).hexdigest(),
                    "hard_valid": False, "numerically_uncertain": True, "budgets": [0., 0., 0.],
                    "budget_defined_mask": [False, False, False]}
                try:
                    _check_conditions(condition[None], cfg)
                    if not np.isfinite(full).all() or not np.isfinite(m).all(): raise ValueError("nonfinite A carrier")
                    full_tensor = torch.as_tensor(full, device=parameter.device)[None]
                    z = extract_value_controls(full_tensor, decoder.value_config)
                    relift = lift_value_controls(z, decoder.value_config)
                    if not torch.equal(relift.view(torch.int32), full_tensor.view(torch.int32)):
                        raise ValueError("A full source carrier is not bitwise native value lift; no silent projection")
                    carrier = {"m": torch.as_tensor(m, device=parameter.device)[None], "z": z}
                    theta = torch.as_tensor(condition, device=parameter.device)
                    grids = [_grid_measure(decoder, carrier, theta, plan, n) for n in plan.grid_sizes]
                    probes, probe_report = _probe_measure(decoder, carrier, theta, plan)
                    row = np.column_stack((grids[-1][0], probes)).ravel()
                    if not np.isfinite(row).all(): raise ValueError("nonfinite A dimensionless features")
                    matrix[index] = row
                    sensitivity = [{"coarse": plan.grid_sizes[j], "fine": plan.grid_sizes[j+1],
                        "maximum_absolute_dimensionless_difference": float(np.abs(grids[j+1][0]-grids[j][0]).max()),
                        "difference_by_time_and_descriptor": (grids[j+1][0]-grids[j][0]).tolist(),
                        "is_convergence_certificate": False} for j in range(len(grids)-1)]
                    horizon = float(cfg.time_domain[1]-cfg.time_domain[0])
                    budgets, defined, ledger = _collision_budgets(probes, grids[-1][0], nu=float(condition[0]), horizon=horizon,
                        minimum_normal_f_rms=np.finfo(np.float64).tiny*float(condition[1])/float(cfg.equation.background_density))
                    fractions = np.asarray(plan.time_fractions)
                    baseline = np.asarray(ledger["horizon_normalized_source_and_injection_by_time"])
                    entry.update(hard_valid=True, budgets=budgets.tolist(), budget_defined_mask=defined.tolist(),
                        budget_ledger=ledger, grids=[g[1] for g in grids], numerical_sensitivity=sensitivity,
                        source_probes=probe_report, source_B_lift_bitwise_preserved=True,
                        horizon_normalized_early_late_source_and_injection={label: baseline[mask].mean(0).tolist() for label, mask in
                            (("early_fraction_le_0_25", fractions <= .25), ("late_fraction_ge_0_5", fractions >= .5))})
                except Exception as error:
                    matrix[index] = 0.
                    entry["failure"] = {"type": type(error).__name__, "message": str(error)}
                cases.append(entry)
                if progress is not None: progress(index+1, len(raw), entry)
    finally:
        decoder.train(was_training)
    report = {"schema": SCHEMA, "plan": asdict(plan), "plan_sha256": _json_digest(asdict(plan)),
        "decoder_identity": identity, "decoder_identity_sha256": _json_digest(identity), "cases": cases,
        "complete": len(cases) == len(raw), "admitted": False, "physical_regime_qualified": False,
        "source_B_coefficients_are_A_answer": False, "B_analytic_source_substituted_for_A": False,
        "A_reported_macros_come_from_target_mean_covariance_not_B_source_moments": True,
        "source_standardization_inside_A_decoder_uses_complete_B_moments": True,
        "full_velocity_source_quadrature_completed": False, "independent_physical_calibration_completed": False,
        "numerical_uncertainty": "all_A_rows_remain_unresolved_until_FD_and_spatial_sensitivity_have_independent_error_budgets",
        "source_precision": "FP64_query_and_affine_velocity_density_with_original_FP32_source_coordinate_decoder",
        "scales": "same_v2_rho_background_sqrt_temperature_horizon_sqrt_LxLy",
        "budget_axes": list(BUDGET_AXES), "budget_time_reduction": "maximum_of_all_registered_times_requires_all_defined",
        "budget_definitions": {BUDGET_AXES[0]: "fixed_probe_RMS(S)/(nu*fixed_probe_RMS(f)); not full-space L2",
            BUDGET_AXES[1]: "norm(mean(dt_j+nu*j-rho*E))/(nu*rho_background*sqrt(temperature))",
            BUDGET_AXES[2]: "abs(mean(dt_K-E_dot_j+2*nu*(K-temperature*rho)))/(nu*rho_background*temperature)"},
        "hash_encoding": "schema_prefix_and_decoder_identity_digest_then_full_B_float32_macro_float32_nuT_float32",
        "failed_row_placeholder": "zero_storage_excluded_by_hard_valid_not_removed_from_attempts",
        "evidence_role": "development_decoder_intervention_not_blind_confirmation"}
    json.dumps(report, allow_nan=False)
    return matrix, descriptors, report


def paired_decoder_comparison(a_matrix, a_report, b_matrix, b_report, descriptors):
    """Within-draw decoder effect, not paired trajectory error against truth."""
    a, b = np.asarray(a_matrix), np.asarray(b_matrix)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(descriptors): raise ValueError("paired feature shapes differ")
    if len(a_report["cases"]) != len(a) or len(b_report["cases"]) != len(b): raise ValueError("paired case count differs")
    if any(x["source_B_answer_sha256"] != y["input_answer_sha256"] for x, y in zip(a_report["cases"], b_report["cases"])):
        raise ValueError("A/B comparison is not the same frozen draw and condition")
    if a_report["plan_sha256"] != b_report["plan_sha256"]: raise ValueError("A/B physical plans differ")
    av, bv = (np.asarray([row["hard_valid"] for row in report["cases"]], dtype=bool) for report in (a_report, b_report))
    selected = av & bv
    delta = a[selected]-b[selected]
    result = {"schema": "vpfp_same_draw_A_minus_B_physical_features_v1", "attempted_count": len(a),
        "both_valid_count": int(selected.sum()), "A_failed_count": int((~av).sum()), "B_failed_count": int((~bv).sum()),
        "paired_truth_claimed": False, "A_or_B_is_truth": False, "same_native_draw_and_condition_verified": True,
        "admitted": False, "rows": [{"name": d["name"], "unit": d["unit"],
            "signed_mean_A_minus_B": float(delta[:, i].mean()) if len(delta) else None,
            "RMS_A_minus_B": _rms(delta[:, i]) if len(delta) else None}
            for i, d in enumerate(descriptors)]}
    json.dumps(result, allow_nan=False)
    return result
