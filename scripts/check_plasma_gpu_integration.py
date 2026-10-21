"""One-case preformal CUDA readiness, called by the integrated Campaign runner.

This is not a distribution audit, benchmark, Agent call or Guidance fit. The
accepted model is loaded by composition once; the check uses that same binding.
It retains its receipt on failure and never counts its private probe as a
formal development/evaluation case. Floating-point agreement is tolerance based.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

SCHEMA = "plasma_pde_gav_cuda_readiness_v1"
SEED = 202610020731
SOURCE_ATOL, RTOL = 2e-5, 5e-5


def _array(value, name, expected_shape=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if (not result.size or not np.isfinite(result).all()
            or expected_shape is not None and result.shape != expected_shape):
        raise ValueError(f"invalid {name}: shape={result.shape}; finite values required")
    return result


def compare(reference, actual, *, name, atol=SOURCE_ATOL, rtol=RTOL):
    left, right = _array(reference, name + ":reference"), _array(actual, name + ":actual")
    if left.shape != right.shape:
        raise ValueError(f"{name} shape changed: {left.shape} vs {right.shape}")
    delta = np.abs(left - right)
    allowance = atol + rtol * np.abs(left)
    report = {"name": name, "passed": bool(np.all(delta <= allowance)),
        "points": int(left.size), "atol": atol, "rtol": rtol,
        "max_abs_error": float(delta.max()), "reference_max_abs": float(np.abs(left).max()),
        "reference_rms": float(np.sqrt(np.mean(left * left))),
        "error_rms": float(np.sqrt(np.mean(delta * delta))),
        "max_tolerance_ratio": float((delta / allowance).max()),
        "bitwise_equality_required": False}
    if not report["passed"]:
        raise ValueError("numerical readiness mismatch: " + json.dumps(report, sort_keys=True))
    return report


def _query(record, fields, points):
    from pde_agent_verifier_solver.sdk import AnswerQuery
    answer = record.truth.answer(AnswerQuery.at_points(fields=fields, points=tuple(points)))
    return {name: _array(answer.fields[name], "truth:" + name, (len(points),)) for name in fields}


def _vpfp_points(record):
    question = record.case.public_data["question"]
    start, end = question["time_domain"]
    lx, ly = question["grid"]["x_domain"]
    phase = np.array([[start + .193 * (end - start), .137 * lx, .227 * ly, .15, -.25],
                      [start + .713 * (end - start), .431 * lx, .619 * ly, -.7, .9]], dtype=np.float64)
    # Sixteen velocities at identical t/x/y exercise the batched geometry route.
    repeated = np.repeat(phase[1:2], 16, axis=0)
    repeated[:, 3] = np.linspace(-1.1, 1.3, 16)
    repeated[:, 4] = .3 * np.sin(np.arange(16))
    return np.concatenate((phase[[0, 1, 0]], repeated)), float(start)


def _check_vpfp(binding, record, problem):
    backend = binding.plugin.generator.backend
    runtime = backend.runtime
    model = runtime.decoder if problem == "weak" else runtime.expanded_model
    numerical_device = next(model.parameters()).device
    if numerical_device.type != "cuda":
        raise ValueError("formal VPFP numerical backend is not resident on CUDA")
    phase, start = _vpfp_points(record)
    fields = ("rho", "momentum_x", "momentum_y", "kinetic_energy")
    coordinates = [{"t": float(t), "x": float(x), "y": float(y)} for t, x, y in phase[:2, :3]]
    truth = _query(record, fields, coordinates)
    session = backend.public_session_factory(record.case)
    try:
        source = _array(session.source(phase), "public source", (len(phase),))
        rho0 = _array(session.rho0(phase[:2, 1:3]), "public rho0", (2,))
        g0 = _array(session.g0(phase[:, 1:]), "public g0", (len(phase),))
        checks = [compare(source, session.source(phase), name="repeated_public_source")]
    finally:
        session.close()
    if np.any(rho0 <= 0) or np.any(g0 < 0):
        raise ValueError("initial VPFP density must remain positive/nonnegative")
    result = {"source_points": len(phase), "macro_points": len(coordinates),
        "numerical_device": str(numerical_device),
        "source_max_abs": float(np.abs(source).max()), "source_rms": float(np.sqrt(np.mean(source * source))),
        "g0_max_abs": float(np.abs(g0).max()), "rho0_min": float(rho0.min()), "checks": checks}
    if problem == "weak":
        from pde_agent_verifier_solver.plugins.vpfp.moving_frame_backend import (
            manufacture_moving_frame_source, evaluate_moving_frame_macros)
        resident = runtime._resident_for_id(record.case.case_id)
        original = manufacture_moving_frame_source(runtime.decoder, phase,
            resident.carrier, resident.condition)
        checks.append(compare(original["source"], source, name="weak_fast_vs_independent_autograd_source"))
        import torch
        with torch.no_grad():
            initial = np.column_stack((np.full(len(phase), start), phase[:, 1:]))
            actual_initial = runtime.decoder.fields(torch.as_tensor(initial, dtype=torch.float64,
                device=next(runtime.decoder.parameters()).device), resident.carrier,
                equation_condition=resident.condition)
            macro = evaluate_moving_frame_macros(runtime.decoder, phase[:2, :3],
                resident.carrier, resident.condition)
        checks.append(compare(actual_initial["g"], g0, name="weak_batched_g0_vs_native_fields", atol=2e-6))
        checks.append(compare(actual_initial["rho"][:2], rho0, name="weak_rho0_vs_native_fields", atol=2e-6))
        for name, reference in (("rho", macro["rho"]), ("momentum_x", macro["momentum"][:, 0]),
                                ("momentum_y", macro["momentum"][:, 1]),
                                ("kinetic_energy", macro["kinetic_energy"])):
            checks.append(compare(reference, truth[name], name="weak_continuous_truth:" + name, atol=2e-6))
        result["source_implementation"] = runtime.source_execution_stats()
        result["scope"] = "actual_cuda_same_carrier_source_derivation_and_public_initial_macro_interfaces"
    else:
        result["scope"] = "actual_cuda_public_source_initial_data_continuous_macro_smoke_not_new_physics_audit"
        result["oracle_kernel"] = getattr(runtime, "oracle_kernel", None)
    return result


def _check_artemis(binding, record):
    domain = record.case.public_data["equation"]["domain"]
    start, end = domain["time_interval"]
    left, right = domain["x_interval"]
    v0, v1 = domain["velocity_query_window"]
    phase = np.array([[start + .193 * (end - start), left + .137 * (right - left), v0 + .47 * (v1 - v0)],
                      [start + .713 * (end - start), left + .431 * (right - left), v0 + .56 * (v1 - v0)]])
    fields = ("M0", "M1", "M2", "M3", "E")
    _query(record, fields, [{"t": float(t), "x": float(x)} for t, x in phase[:, :2]])
    session = binding.plugin.generator.backend.public_session_factory(record.case)
    try:
        source = _array(session.query_source(phase), "Artemis source", (2,))
        initial = _array(session.query_initial_distribution(phase[:, 1:]), "Artemis initial distribution", (2,))
        checks = [compare(source, session.query_source(phase), name="repeated_artemis_public_source")]
    finally:
        session.close()
    if np.any(initial < 0):
        raise ValueError("Artemis initial distribution must remain nonnegative")
    return {"source_points": 2, "macro_points": 2, "source_max_abs": float(np.abs(source).max()),
        "initial_min": float(initial.min()), "checks": checks,
        "scope": "actual_cuda_public_source_initial_data_continuous_macro_smoke_not_new_physics_audit"}


def check_binding(problem, binding, *, output_path, seed=SEED):
    """Check one nonformal case and always release it; raise on failed readiness."""
    if problem not in ("strong", "weak", "artemis"):
        raise ValueError("expected strong, weak or artemis")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real model readiness requires a compute allocation")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("formal Plasma readiness requires CUDA")
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
    from pde_agent_verifier_solver.sdk import (
        GenerationRequest, GeneratorMode, GenerationKind, GenerationPurpose, CaseVisibility)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("readiness receipt already exists; do not silently overwrite evidence")
    started = perf_counter()
    report = {"schema": SCHEMA, "problem": problem, "status": "running", "seed": seed,
        "case_count": 0, "model_api_called": False, "guidance_trained": False,
        "performance_benchmark": False, "distribution_or_regime_reaudited": False,
        "historical_evidence_replaced": False, "formal_case_count": 0,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": torch.cuda.get_device_name(),
        "check_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    generator = binding.plugin.generator
    state, records = None, ()
    try:
        state = generator.load(binding.checkpoint, {})
        records = tuple(generator.sample(state, GenerationRequest(round_index=1, count=1, seed=seed,
            mode=GeneratorMode.ONLINE, kind=GenerationKind.BASE,
            purpose=GenerationPurpose.PROMOTION, visibility=CaseVisibility.HIDDEN)))
        if len(records) != 1:
            raise ValueError("readiness requested exactly one fresh BASE case")
        report["case_count"] = len(records)
        report["numerics"] = (_check_artemis(binding, records[0]) if problem == "artemis"
                              else _check_vpfp(binding, records[0], problem))
        torch.cuda.synchronize()
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if records:
                generator.release(state, tuple(record.case.case_id for record in records))
            report["private_probe_released"] = True
        except BaseException as cleanup_error:
            report.update(status="failed", cleanup_error=str(cleanup_error), private_probe_released=False)
            raise
        finally:
            report["wall_seconds"] = perf_counter() - started
            atomic_json(output_path, report)
