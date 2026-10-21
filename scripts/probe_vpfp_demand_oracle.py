"""Scheduled bounded real-decoder probe; no training or new diffusion draws."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
PIPELINE = Path("${PDE_GAV_RUN_ROOT}/vpfp-strong-solver-guidance-preflight-5299309/pipeline")
OLD_RESULT = PIPELINE/"rollouts/fit/fit-0002/strength-0/attempt-000/result.json"


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""): h.update(chunk)
    return h.hexdigest()


def load_saved_prediction(path, expected_sha256):
    """Read the hash-bound legacy numeric list payload without unsafe pickle."""
    import numpy as np
    import torch
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('retained prediction is missing or symlinked')
    # The historical writer stored lists of np.float64, not tensor arrays.
    # Allow only that scalar/dtype reconstruction, scoped to this verified file.
    scalar = np.core.multiarray.scalar
    allowed = [(scalar, "numpy.core.multiarray.scalar"),
        (scalar, "numpy._core.multiarray.scalar"), np.dtype, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def candidate_source(sparse):
    requests = """
    oracle = open_public_case_session(case)
    oracle.source([[.451,.971,.137,6.25,-7.], [.127,.213,.789,-6.5,.2], [.451,.971,.137,6.25,-7.]])
    oracle.rho0([[.91,.12],[.13,.82],[.91,.12],[1.17,.23],[.34,1.71],[.73,.19],[.23,.51]])
    oracle.g0([[.91,.12,6.25,-7.],[.13,.82,-6.5,.2]])
""" if sparse else ""
    return """import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView, open_public_case_session
def predict_macros(case, query, budget=None):
""" + requests + """
    a = case.public_data['arrays']
    shape = tuple(query.resolution[k] for k in ('t','x','y'))
    return AnswerView(query.kind, {k:tuple(np.zeros(np.prod(shape))) for k in query.fields},
        axes={k:tuple(a[k]) for k in ('t','x','y')}, shape=shape)
"""


def compare_batched_source(runtime, case_id, check, results):
    """Explicit test requests, separate from the solver's demand accounting."""
    import numpy as np
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.batched_source import evaluate_requested_source
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _latent_for_model
    axes = (np.asarray([0., .173, 1.]), np.linspace(.071, 5.77, 5),
        np.linspace(.137, 5.91, 7), np.linspace(-5., 5., 9), np.linspace(-5., 5., 9))
    dense = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 5)
    rng = np.random.default_rng(9820260908)
    shuffled = dense[rng.permutation(len(dense))[:8192]]
    shuffled = np.concatenate((shuffled, shuffled[[17, 2, 17]]))
    sparse = np.asarray([[.451,.971,.137,6.25,-7.], [.127,.213,.789,-6.5,.2],
        [.451,.971,.137,6.25,-7.], [0.,1.23,2.71,.13,.97], [1.,1.23,2.71,.13,.97]])
    results.update({"tolerance": {"atol": 2e-6, "rtol": 5e-5},
        "scope": "engineering_FP32_equivalence_to_existing_source_not_physical_admission",
        "request_sets": {}, "timings": {}, "chunk_checks": []})
    resident = runtime._resident_for_id(case_id)
    model = runtime.expanded_model
    latent = _latent_for_model(resident.latent, model, torch)
    for name, points in (("dense", dense), ("shuffled_partial_with_duplicates", shuffled), ("sparse", sparse)):
        check()
        reference = runtime.evaluate_source_reference(case_id, points)
        optimized = runtime.evaluate_source(case_id, points)
        with torch.no_grad(), runtime.api.ieee_float32_coordinate_matmul(model):
            diagnostic_value, accounting = evaluate_requested_source(model, points, latent,
                equation_condition=resident.equation_condition, return_diagnostics=True)
        value = diagnostic_value.detach().cpu().numpy().astype(np.float64)
        valid = bool(np.allclose(optimized, reference, atol=2e-6, rtol=5e-5))
        if not np.array_equal(optimized, value):
            raise ValueError("runtime did not use the measured batched source kernel")
        results["request_sets"][name] = {"points": len(points), "allclose": valid,
            "max_abs": float(np.max(np.abs(optimized-reference))),
            "rms": float(np.sqrt(np.mean((optimized-reference)**2))),
            "kernel_accounting": accounting}
        if not valid: raise ValueError("batched source differs from reference: "+name)
        if accounting["evaluated_pairs"] > len(points):
            raise ValueError("batching evaluated unrequested Cartesian pairs")
    # Keep the numerical comparison independent of timing warmup/order. Every
    # timed call evaluates the same actual 8192 requested points on this GPU.
    timed = dense[:8192]
    results["benchmark_requested_points_per_call"] = len(timed)
    results["timings"] = {"reference_seconds": [], "batched_seconds": []}
    for turn in range(3):
        methods = [("reference_seconds", runtime.evaluate_source_reference), ("batched_seconds", runtime.evaluate_source)]
        if turn % 2: methods.reverse()
        for label, method in methods:
            check(); torch.cuda.synchronize()
            start = time.perf_counter()
            method(case_id, timed)
            torch.cuda.synchronize()
            results["timings"][label].append(time.perf_counter()-start)
    reference_time = float(np.median(results["timings"]["reference_seconds"]))
    batched_time = float(np.median(results["timings"]["batched_seconds"]))
    results["timings"].update(reference_median_seconds=reference_time,
        batched_median_seconds=batched_time, speedup=reference_time/batched_time)
    mixed = np.concatenate((dense[[0,1,80,81,511,512,1001]], sparse, sparse[[2,0]]))
    reference = runtime.evaluate_source_reference(case_id, mixed)
    with torch.no_grad(), runtime.api.ieee_float32_coordinate_matmul(model):
        for chunk_size in (1, 7, 512, 8192):
            check()
            value = evaluate_requested_source(model, mixed, latent,
                equation_condition=resident.equation_condition, chunk_size=chunk_size)
            value = value.detach().cpu().numpy()
            close = bool(np.allclose(value, reference, atol=2e-6, rtol=5e-5))
            results["chunk_checks"].append({"chunk_size": chunk_size, "allclose": close,
                "max_abs": float(np.max(np.abs(value-reference)))})
            if not close: raise ValueError("source depends materially on requested batch partition")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-legacy", action="store_true",
        help="Explicitly let the unchanged historical solver request its entire old grid once.")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("compute allocation required")
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        raise ValueError("new absolute output directory required")
    if not args.output.is_relative_to(Path("${PDE_GAV_RUN_ROOT}")):
        raise ValueError("use the documented scratch results root")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"schema": "vpfp_demand_batched_real_decoder_probe_v1", "execution_passed": False,
        "source_revision": revision, "job_id": os.environ["SLURM_JOB_ID"],
        "new_diffusion_draws": 0, "data_generation_pde_solves": 0, "training_updates": 0,
        "historical_pde_solver_executions": 0,
        "scientific_admission": False, "guidance_effect_test": False,
        "include_legacy_full_grid_requested_by_solver": args.include_legacy,
        "checks": {}, "failures": []}
    runtime = replay = session = None
    case_id = "demand-oracle-probe"
    def save():
        report["wall_seconds"] = time.monotonic()-started
        temporary = args.output/"result.json.partial"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        temporary.replace(args.output/"result.json")
    def check():
        if time.monotonic()-started > 300:
            raise TimeoutError("bounded real-decoder probe total wall budget reached")
    save()
    try:
        import numpy as np
        import torch
        from pde_agent_verifier_solver.sdk import CaseVisibility
        from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
        from pde_agent_verifier_solver.plugins.vpfp.demand_solver_reward import run_demand_solver
        from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import load_frozen_solver, native_macro_query
        from vpfp_strong_guidance_rollouts import build_rollout_runtime
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one allocated GPU required")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        raw = OLD_RESULT.read_bytes()
        previous = json.loads(raw)
        if previous.get("valid") is not True: raise ValueError("retained successful real draw required")
        report["replayed_result"] = {"path": str(OLD_RESULT), "sha256": sha256(raw).hexdigest()}
        for name in ("private_draw", "prediction"):
            item = previous[name]
            if not Path(item["path"]).is_file():
                raise ValueError('retained ' + name + ' changed')
        draw = torch.load(previous["private_draw"]["path"], map_location="cpu", weights_only=True)
        old_prediction = (load_saved_prediction(previous["prediction"]["path"],
            previous["prediction"]["sha256"]) if args.include_legacy else None)
        # Snapshot only the sampler's public asset identity, not current labels
        # or a claimed terminal state of the independently running experiment.
        identity_snapshot = (PIPELINE/"result.json").read_bytes()
        identity = json.loads(identity_snapshot)["base_sampling_identity"]
        report["identity_source_snapshot_sha256"] = sha256(identity_snapshot).hexdigest()
        report["sampler_identity_snapshot"] = identity
        native = VPFPOnlineModelSpec.from_mapping(json.loads(
            (ROOT/"configs/vpfp_unified_live_strong_collision.json").read_text())["vpfp"])
        binding, replay = build_rollout_runtime(native, SimpleNamespace(identity=identity), args.output/"native", "cuda")
        runtime = binding.plugin.generator.backend.runtime
        replay.set_draw(draw)
        wrapped = runtime.generate_base(seeds=tuple(draw["root_seeds"]), case_ids=(case_id,),
            visibility=CaseVisibility.HIDDEN, round_index=0, cohort="demand_probe")[0]
        if wrapped.transaction is not None: wrapped.transaction.commit()
        case = wrapped.record.case
        session = runtime.public_session_factory(case)
        query = native_macro_query(case)
        batch_evidence = {}
        report["checks"]["batched_source_equivalence_and_timing"] = batch_evidence
        compare_batched_source(runtime, case_id, check, batch_evidence)
        save()
        class RecordedSession:
            def __init__(self): self.small = []
            def call(self, method, points):
                value = getattr(session, method)(points)
                if len(points) <= 16: self.small.append((method, np.asarray(points).copy(), np.asarray(value).copy()))
                return value
            def source(self, points): return self.call("source", points)
            def rho0(self, points): return self.call("rho0", points)
            def g0(self, points): return self.call("g0", points)
        recorded = RecordedSession()
        for label, sparse in (("zero_queries", False), ("sparse_queries", True)):
            path = args.output/(label+".py")
            path.write_text(candidate_source(sparse))
            asset = load_frozen_solver(path, digest(path))
            prediction, receipt = run_demand_solver(asset, case, query,
                public_session=recorded, budget_seconds=60., check=check)
            report["checks"][label] = receipt
            save()
            expected = {"source": 3, "rho0": 7, "g0": 2} if sparse else {"source": 0, "rho0": 0, "g0": 0}
            if prediction is None or not receipt["execution_passed"] or receipt["prematerialized_points"] != 0:
                raise ValueError(label+" execution failed")
            if {name: item["points"] for name, item in receipt["oracle_usage"].items()} != expected:
                raise ValueError("unrequested points evaluated")
        comparisons = []
        for method, points, returned in recorded.small:
            direct = np.asarray(getattr(session, method)(points))
            comparisons.append({"method": method, "verification_only_extra_points": len(points),
                "max_abs": float(np.max(np.abs(direct-returned))), "equal": bool(np.array_equal(direct, returned))})
        report["checks"]["direct_public_oracle_comparison"] = comparisons
        if not all(item["equal"] for item in comparisons): raise ValueError("numerical proxy changed public values")
        if args.include_legacy:
            cfg = json.loads((ROOT/"configs/vpfp_strong_solver_guidance_demand_20260908.json").read_text())
            if cfg["solver_sha256"] != previous["solver_sha256"]:
                raise ValueError("legacy solver differs from retained comparison")
            asset = load_frozen_solver(cfg["solver"], cfg["solver_sha256"])
            report["historical_pde_solver_executions"] = 1
            save()
            prediction, receipt = run_demand_solver(asset, case, query, public_session=recorded,
                budget_seconds=180., legacy_array_adapter=True, check=check)
            report["checks"]["unchanged_legacy_solver"] = receipt
            save()
            if prediction is None or not receipt["execution_passed"]: raise ValueError("real legacy solver failed")
            expected = {"source": 1146717, "rho0": 1089, "g0": 88209}
            if {k: v["points"] for k, v in receipt["oracle_usage"].items()} != expected:
                raise ValueError("legacy requested-point accounting changed")
            old = old_prediction
            if tuple(old["shape"]) != tuple(prediction.shape) or any(
                    not np.array_equal(np.asarray(old["axes"][name]), np.asarray(prediction.axes[name]))
                    for name in ("t", "x", "y")):
                raise ValueError("legacy prediction comparison changed output coordinates")
            deltas = {}
            for name, values in prediction.fields.items():
                value, target = np.asarray(values), np.asarray(old["fields"][name])
                delta = value-target
                deltas[name] = {"max_abs": float(np.max(np.abs(delta))),
                    "rms": float(np.sqrt(np.mean(delta**2))),
                    "allclose_atol_1e_7_rtol_1e_6": bool(np.allclose(value, target, atol=1e-7, rtol=1e-6))}
            report["checks"]["retained_prediction_comparison"] = deltas
            if not all(v["allclose_atol_1e_7_rtol_1e_6"] for v in deltas.values()):
                raise ValueError("legacy prediction differs from retained eager execution")
            load_frozen_solver(cfg["solver"], cfg["solver_sha256"])
        report["execution_passed"] = True
    except Exception as exc:
        if getattr(exc, "demand_solver_receipt", None) is not None:
            report["interrupted_solver_receipt"] = exc.demand_solver_receipt
        report["failures"].append({"type": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc()})
    finally:
        for cleanup in ((lambda: session.close()) if session else lambda: None,
                (lambda: runtime.release_cases((case_id,))) if runtime else lambda: None,
                replay.clear if replay else lambda: None):
            try: cleanup()
            except Exception as exc:
                report["execution_passed"] = False
                report["failures"].append({"type": type(exc).__name__, "reason": str(exc), "stage": "cleanup"})
        report["wall_seconds"] = time.monotonic()-started
        report["source_final_verified"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == revision
        if not report["source_final_verified"]: report["execution_passed"] = False
        save()
    print(json.dumps({"execution_passed": report["execution_passed"], "result": str(args.output/"result.json")}))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
