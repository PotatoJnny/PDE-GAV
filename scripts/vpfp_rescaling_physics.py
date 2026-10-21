"""Small frozen-point VPFP diagnostics; never a solver or admission gate.

Every draw gets the existing 128-point source probe and four-point AD check.
The caller selects the first root of each stratum before seeing results for
four local full-velocity audits. No spatial grid, new draw, or PDE solve is
created here. The existing audit kernels retain their original arithmetic.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import importlib
import json
import math
import os
from time import perf_counter
from types import SimpleNamespace


SCHEMA = "vpfp_rescaling_fixed_probe_physics_v1"
LOCAL_TIME_FRACTIONS = (.2, .8)
LOCAL_XY_FRACTIONS = ((.25, .25), (.75, .75))


def _require_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("VPFP numerical physics diagnostics require a compute allocation")


def _components():
    # Lazy imports leave module import and monkeypatched contract tests cheap.
    import numpy as np
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import (
        VPFPRegimeFeaturePlan, _probe_measure,
    )
    from pde_agent_verifier_solver.plugins.vpfp.independent_physics import (
        audit_independent_physics, _finite,
    )
    _legacy_api()  # Enforce the immutable physical-kernel identity first.
    return SimpleNamespace(np=np, torch=torch, Plan=VPFPRegimeFeaturePlan,
        probe=_probe_measure, local=audit_independent_physics, finite=_finite,
        analytic=importlib.import_module("agent_verifier.vpfp.analytic_source"))


def _finite_max(values):
    selected = [float(v) for v in values if v is not None]
    if not all(math.isfinite(v) for v in selected):
        raise FloatingPointError("nonfinite local diagnostic summary")
    return max(selected) if selected else None


def _local_summary(detail, np):
    points = []
    for row in detail.get("points", []):
        checks = row.get("finite_difference_checks", [])
        refinement = []
        for index in range(0, len(checks)-1, 2):
            coarse, fine = checks[index:index+2]
            a, b = [np.asarray(item["balance_residual"], dtype=float) for item in (coarse, fine)]
            refinement.append({"velocity_order": coarse["velocity_order"],
                "coarse_balance_residual_rms": float(np.sqrt(np.mean(a*a))),
                "fine_balance_residual_rms": float(np.sqrt(np.mean(b*b))),
                "fine_residual_not_larger": bool(np.linalg.norm(b) <= np.linalg.norm(a)),
                "is_convergence_certificate": False})
        points.append({"index": row["index"], "txy": row["txy"],
            "execution_passed": row.get("execution_passed") is True,
            "source_L2_relative_to_nu_f_L2": row.get("source_L2_relative_to_nu_f_L2"),
            "source_moments_collision_normalized": row.get("source_moments_collision_normalized"),
            "source_moment_fine_minus_coarse": row.get("source_moment_fine_minus_coarse"),
            "L2_squared_fine_minus_coarse": row.get("L2_squared_fine_minus_coarse"),
            "fd_refinement": refinement})
    return {"attempted_points": detail.get("attempted_point_count", len(points)),
        "completed_points": detail.get("completed_point_count", 0), "points": points,
        "maximum_local_source_L2_relative_to_nu_f_L2": _finite_max(
            row["source_L2_relative_to_nu_f_L2"] for row in points),
        "fd_flag_meaning": "residual_magnitude_nonincrease_only_no_calibrated_tolerance_or_pass_gate",
        "global_spatial_source_balance_measured": False,
        "global_spatial_mass_measured": False}


def audit_draw(draw, base_model, *, full_local_audit: bool):
    """Return finite JSON diagnostics for the supplied original full draw.

    Measurement failures remain separate from solver numerical validity. Even
    successful measurement leaves numerically_undetermined=True: the inherited
    protocol has no calibrated admission tolerance. No model/solver selection,
    artifact writing, sampling, or checkpoint mutation is performed here.
    """
    started = perf_counter()
    report = {"schema": SCHEMA, "execution_passed": False,
        "numerically_undetermined": True, "scientific_admission": False,
        "solver_validity_label": None, "full_local_audit_requested": full_local_audit,
        "input": None, "plan": None, "light": None, "local": None,
        "summary": {}, "failures": [], "new_solver_calls": 0, "new_draws": 0}
    was_training = None
    def failed(stage, exc):
        report["failures"].append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})
    try:
        if type(full_local_audit) is not bool:
            report["full_local_audit_requested"] = None
            raise TypeError("full_local_audit must be an explicit bool")
        _require_allocation()
        api = _components()
        np, torch = api.np, api.torch
        if not isinstance(draw, Mapping):
            raise TypeError("original completed draw mapping required")
        full, condition = draw["full"], draw["equation_conditions"]
        cfg = base_model.config
        parameter = next(base_model.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("retain the frozen native FP32 decoder")
        for key, value, shape in (("full", full, (1, cfg.latent_dim)),
                                  ("equation_conditions", condition, (1, 2))):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
                raise ValueError("one finite original native FP32 row required: " + key)
        full_host = full.detach().cpu().contiguous().numpy().copy()
        theta = condition.detach().cpu().contiguous().numpy()[0].copy()
        if theta[0] < 0 or theta[1] <= 0:
            raise ValueError("finite nu>=0 and bath temperature>0 required")
        start, stop = map(float, cfg.time_domain)
        lengths = np.asarray(cfg.equation.x_domain, dtype=float)
        if (not all(map(math.isfinite, (start, stop))) or stop <= start
                or lengths.shape != (2,) or not np.isfinite(lengths).all() or not (lengths > 0).all()):
            raise ValueError("physical time interval and positive spatial domain lengths required")
        if int(cfg.hermite_total_degree) != 14:
            raise ValueError("this frozen rescaling audit expects the original N14 decoder")
        plan = api.Plan()
        local_points = np.asarray([[start+(stop-start)*t, x*lengths[0], y*lengths[1]]
            for t in LOCAL_TIME_FRACTIONS for x, y in LOCAL_XY_FRACTIONS])
        report["plan"] = {"light_time_fractions": list(plan.time_fractions),
            "phase_probes_per_time": plan.phase_probes_per_time,
            "light_phase_points": len(plan.time_fractions)*plan.phase_probes_per_time,
            "independent_AD_points": plan.independent_source_probe_count,
            "probe_seed": plan.probe_seed, "light_chunk_size": plan.chunk_size,
            "local_time_fractions": list(LOCAL_TIME_FRACTIONS),
            "local_xy_fractions": [list(xy) for xy in LOCAL_XY_FRACTIONS],
            "local_points_txy": local_points.tolist(), "spatial_grids": [],
            "moment_orders": [17, 19], "L2_orders": [32, 34],
            "local_chunk_size": 512, "relative_fd_steps": [1/512, 1/1024],
            "physical_units": "native t,x,y; velocity probes scaled by sqrt(bath temperature)",
            "source_backend": "existing_pointwise_analytic_not_new_batched_source",
            "numeric_tolerance": None, "numerical_gate": False}
        digest = sha256()
        for key, array in (("full", full_host), ("condition", theta)):
            digest.update(key.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
        report["input"] = {"native_answer_sha256": digest.hexdigest(),
            "condition": theta.tolist(), "time_domain": [start, stop],
            "spatial_lengths": lengths.tolist()}
        was_training = base_model.training
        base_model.eval()
        light_start = perf_counter()
        try:
            with torch.no_grad(), api.analytic.ieee_float32_coordinate_matmul(base_model):
                features, detail = api.probe(base_model, full_host, theta, plan, api.analytic)
            features = np.asarray(features, dtype=float)
            if features.shape != (len(plan.time_fractions), 9) or not np.isfinite(features).all():
                raise FloatingPointError("finite aligned 8x9 fixed-probe features required")
            independent = detail.get("independent_source_check", {})
            if len(detail.get("points_txyvv", ())) != len(plan.time_fractions)*plan.phase_probes_per_time:
                raise ValueError("fixed light-probe inventory changed")
            if independent.get("execution_passed") is True and len(independent.get("probe_indices", ())) != 4:
                raise ValueError("four independent AD source probes required")
            report["light"] = api.finite({"execution_passed": independent.get("execution_passed") is True,
                "features_by_time": features, "detail": detail})
            background = float(cfg.equation.background_density)
            if not math.isfinite(background) or background <= 0:
                raise ValueError("positive finite background density required")
            rows = []
            for fraction, values in zip(plan.time_fractions, features):
                f_rms = float(values[0])*background/float(theta[1])
                physical = values[1:6]*background/(float(theta[1])*(stop-start))
                denominator = float(theta[0])*f_rms
                ratio = float(physical[0]/denominator) if (
                    math.isfinite(denominator) and denominator > 0 and f_rms >= 2.**-126) else None
                rows.append({"time_fraction": fraction, "time": start+(stop-start)*fraction,
                    "f_fixed_probe_rms": f_rms, "source_fixed_probe_rms": float(physical[0]),
                    "dt_f_fixed_probe_rms": float(physical[1]), "transport_fixed_probe_rms": float(physical[2]),
                    "electric_acceleration_fixed_probe_rms": float(physical[3]),
                    "collision_fixed_probe_rms": float(physical[4]),
                    "source_relative_to_nu_f_fixed_probe_rms": ratio,
                    "collision_ratio_defined": ratio is not None})
            report["summary"]["fixed_probes"] = api.finite({"by_time": rows,
                "maximum_source_relative_to_nu_f": _finite_max(
                    r["source_relative_to_nu_f_fixed_probe_rms"] for r in rows),
                "independent_AD_check_completed": independent.get("execution_passed") is True,
                "algebraic_self_balance_is_independent_validation": False,
                "scope": "fixed_probes_not_full_velocity_or_spatial_norms"})
            if independent.get("execution_passed") is not True:
                failed("light_independent_AD", RuntimeError(str(independent.get("failure", "independent source check incomplete"))))
        except Exception as exc:
            failed("light", exc)
        report["summary"]["light_wall_seconds"] = perf_counter()-light_start
        if full_local_audit:
            local_start = perf_counter()
            try:
                detail = api.local(base_model, full, condition, points_txy=local_points,
                    spatial_times=(), spatial_grids=(), chunk_size=512,
                    relative_fd_step=1/512, include_l2=True)
                report["local"] = api.finite(detail)
                report["summary"]["local"] = api.finite(_local_summary(detail, np))
                if detail.get("execution_passed") is True and (
                        detail.get("attempted_point_count") != 4 or detail.get("completed_point_count") != 4
                        or len(detail.get("points", ())) != 4):
                    raise ValueError("four completed local txy probes required")
                if detail.get("execution_passed") is not True:
                    failed("local", RuntimeError("local audit incomplete; original nested errors retained"))
            except Exception as exc:
                failed("local", exc)
            report["summary"]["local_wall_seconds"] = perf_counter()-local_start
        report["execution_passed"] = not report["failures"]
    except Exception as exc:
        failed("input_or_runtime", exc)
    finally:
        if was_training is not None:
            base_model.train(was_training)
        report["wall_seconds"] = perf_counter()-started
    json.dumps(report, allow_nan=False)
    return report
