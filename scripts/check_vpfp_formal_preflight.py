#!/usr/bin/env python3
"""Compare Candidate throughput on sixteen fixed, private VPFP BASE roots.

This engineering check does not change the formal scientific configuration,
train Guidance, invoke an Agent or admit a scientific model. All requested
concurrency groups must complete consistently before any worker count is chosen.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_canonical_vpfp_campaign import Writer, file_sha, jsonable

SCHEMA = "vpfp_formal_fixed_case_concurrency_preflight_v1"
ROOT_COUNT = 16
FIELDS = ("rho", "momentum_x", "momentum_y", "kinetic_energy")
ARTIFACT_ID = "vpfp-formal-preflight-s0"
REQUIRED_METRICS = {"rho_relative_l2", "momentum_relative_l2", "kinetic_energy_relative_l2"}
ABS_TOL = 1e-12
REL_TOL = 1e-10


def parse_worker_counts(text):
    try:
        counts = tuple(int(value) for value in text.split(","))
    except (TypeError, ValueError) as error:
        raise ValueError("worker counts must be comma-separated integers") from error
    if counts != (4, 8, 16):
        raise ValueError("this bounded preflight requires exactly worker counts 4,8,16")
    return counts


def _finite_nonnegative(value):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0)


def verify_measurement(measurement, inventory, *, solver_sha, point_count):
    """Require all sixteen results, never silently select a valid subset."""
    if len(inventory) != ROOT_COUNT or len({r["root_seed"] for r in inventory}) != ROOT_COUNT:
        raise ValueError("preflight requires sixteen independent physical roots")
    case_ids = [r["case_id"] for r in inventory]
    if len(set(case_ids)) != ROOT_COUNT:
        raise ValueError("preflight requires sixteen unique Cases")
    if measurement["status"] != "measured" or measurement["solver_sha256"] != solver_sha:
        raise ValueError("measurement failed or evaluated a different Solver")
    rows, audits = measurement["rows"], measurement["audits"]
    if len(rows) != ROOT_COUNT or [r["case_id"] for r in rows] != case_ids:
        raise ValueError("Candidate results do not contain the complete fixed Case order")
    if len(audits) != ROOT_COUNT or len({(r["execution_id"], r["case_index"]) for r in audits}) != ROOT_COUNT:
        raise ValueError("missing or repeated Candidate execution attempts")
    if {r["case_index"] for r in audits} != set(range(ROOT_COUNT)):
        raise ValueError("Candidate audit indices do not cover every fixed Case")
    for audit in audits:
        if (audit["status"] != "completed" or audit["return_code"] != 0
                or audit["error_code"] is not None or audit["phase"] != "official"
                or audit["process_group_cleaned"] is not True
                or audit["candidate_process_isolated"] is not True
                or audit["controller_truth_shared"] is not False
                or audit["artifact_id"] != ARTIFACT_ID):
            raise ValueError("Candidate execution, isolation or cleanup failed")
    for row, expected in zip(rows, inventory, strict=True):
        if row["root_seed"] != expected["root_seed"]:
            raise ValueError("Candidate score belongs to a different physical root")
        if (row["point_count"] != point_count or row["fields"] != list(FIELDS)
                or row["candidate_query_point_count"] != point_count):
            raise ValueError("private macro evaluation point or field contract changed")
        digest = row["query_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing actual private query fingerprint")
        if not _finite_nonnegative(row["primary_error"]):
            raise ValueError("invalid macro score")
        if (not REQUIRED_METRICS.issubset(row["macro_metrics"])
                or any(not _finite_nonnegative(v) for v in row["macro_metrics"].values())):
            raise ValueError("invalid or absent per-channel macro metrics")
    if not _finite_nonnegative(measurement["wall_seconds"]) or measurement["wall_seconds"] == 0:
        raise ValueError("throughput timing must be finite and positive")


def choose_worker_count(measurements, inventory, *, solver_sha, point_count, worker_counts=(4, 8, 16)):
    """Fail the entire check if even one requested concurrency group fails."""
    if tuple(row["worker_count"] for row in measurements) != tuple(worker_counts):
        raise ValueError("a requested concurrency group is missing or reordered")
    for measurement in measurements:
        verify_measurement(measurement, inventory, solver_sha=solver_sha, point_count=point_count)
    reference = measurements[0]["rows"]
    for measurement in measurements[1:]:
        for baseline, row in zip(reference, measurement["rows"], strict=True):
            if (baseline["query_sha256"] != row["query_sha256"]
                    or baseline["macro_metrics"].keys() != row["macro_metrics"].keys()):
                raise ValueError("private points or macro metric identities changed with concurrency")
            left = {"primary_error": baseline["primary_error"], **baseline["macro_metrics"]}
            right = {"primary_error": row["primary_error"], **row["macro_metrics"]}
            if any(not math.isclose(left[k], right[k], rel_tol=REL_TOL, abs_tol=ABS_TOL) for k in left):
                raise ValueError("macro scores are inconsistent across worker counts")
    return min(measurements, key=lambda row: (row["wall_seconds"], row["worker_count"]))["worker_count"]


def _source_snapshot():
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    return revision, dirty


def _query_receipt(plan):
    payload = {"fields": list(plan.query.fields), "points": [dict(point) for point in plan.query.points]}
    return {"fields": payload["fields"], "point_count": len(payload["points"]),
            "query_sha256": sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "query": payload}


def run_preflight(config_path, output, worker_counts=(4, 8, 16)):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("VPFP formal numerical preflight requires a Slurm compute allocation")
    worker_counts = parse_worker_counts(",".join(map(str, worker_counts)))
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())  # Deliberately not the historical runner's load_config.
    if config["task"].get("require_prepared_solution") is not True:
        raise ValueError("preflight requires query-blind prepare_solution")
    point_count = config["task"]["official_point_count"]
    if type(point_count) is not int or point_count < 1:
        raise ValueError("private official point_count must be a positive integer")
    revision, dirty = _source_snapshot()
    if dirty:
        raise ValueError("preflight requires a clean committed source checkout")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    writer = Writer(output)
    started = time.monotonic()
    report = {"schema": SCHEMA, "status": "running", "execution_passed": False,
        "preflight_passed": False, "selected_worker_count": None, "source_revision": revision,
        "config_sha256": file_sha(config_path), "worker_counts": list(worker_counts),
        "independent_roots": ROOT_COUNT, "private_macro_points": point_count,
        "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
        "llm_invoked": False, "guidance_training_performed": False,
        "scientific_admission": False, "formal_config_modified": False,
        "online_assets_replaced": False, "measurements": [], "failures": [],
        "selection_rule": "minimum elapsed complete consistent group; all three groups must pass"}
    binding = state = base = initial = None
    guard, inventory = {}, []
    try:
        from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
        from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
        from pde_agent_verifier_solver.sdk import (AnswerQuery, CaseVisibility, GenerationKind, GeneratorMode,
            GenerationPurpose, GenerationRequest, TaskKind, inspect_candidate_bundle)
        from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import _module_sha, _json_sha
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        binding = build_vpfp_strong_iteration_binding(**config["backend"], output_root=output / "private_backend")
        generator, backend = binding.plugin.generator, binding.plugin.generator.backend
        base = backend.capture.base
        before = (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
        state = generator.load(binding.checkpoint, {})
        with _forbid_online_vpfp_generation(guard):
            records = tuple(generator.sample(state, GenerationRequest(round_index=0, count=ROOT_COUNT,
                mode=GeneratorMode.ONLINE, kind=GenerationKind.BASE, visibility=CaseVisibility.HIDDEN,
                seed=config["master_seed"] + 12700000000, purpose=GenerationPurpose.GUIDANCE_FEEDBACK)))
            inventory = [{"case_id": record.case.case_id, "root_seed": backend._draws[record.case.case_id]["root_seed"]}
                         for record in records]
            if len(records) != ROOT_COUNT or len({row["root_seed"] for row in inventory}) != ROOT_COUNT:
                raise ValueError("generation did not return sixteen independent actual BASE roots")
            report["inventory"] = inventory
            report["sampling_audits"] = jsonable(backend.sampling_audits)
            codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
            initial = codec.bind_existing((config_path.parent / config["candidate"]["initial_bundle"]).resolve(),
                storage_root=output / "controller_candidates", artifact_id=ARTIFACT_ID,
                name="Frozen formal S0", summary="Public-only continuous strong VPFP incumbent",
                usage="prepare_solution then evaluate private points")
            report["solver_sha256"] = initial.program.tree_sha256
            task = binding.plugin.task("observable_model")
            query = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": 0., "y": 0.},))
            settings = config["execution"]
            writer.write("summary.json", report)
            for count in worker_counts:
                (output / f"workers-{count:02d}").mkdir(mode=0o700)
                measurement = {"worker_count": count, "status": "running", "rows": [], "audits": [],
                    "solver_sha256": initial.program.tree_sha256, "failures": []}
                service = None
                measured_started = time.monotonic()
                try:
                    service = SubprocessCandidateExecutionService(output / f"workers-{count:02d}" / "candidate_execution",
                        task=PythonTaskWorkerSpec.from_task_type(type(task)),
                        evaluation_settings={**config["task"], "default_query": query}, max_workers=count,
                        case_hard_wall_seconds=settings["candidate_case_hard_wall_seconds"],
                        development_batch_hard_wall_seconds=settings["development_candidate_batch_hard_wall_seconds"],
                        python_executable=sys.executable)
                    verifier = HiddenVerifier(task, query, settings=config["task"],
                        seed=config["master_seed"] + 12800000000, generator=generator, generator_state=state,
                        candidate_execution=service)
                    plans = [verifier._plan_for(record, index) for index, record in enumerate(records)]
                    query_rows = [_query_receipt(plan) for plan in plans]
                    writer.write(f"workers-{count:02d}/private-queries.json", query_rows)
                    for record, plan in zip(records, plans, strict=True):
                        verifier._truth_for(record, plan)
                    measurement["truth_and_plan_wall_seconds"] = time.monotonic() - measured_started
                    measured_started = time.monotonic()
                    scored = verifier.evaluate(initial, records)
                    measurement["wall_seconds"] = time.monotonic() - measured_started
                    measurement["cases_per_second"] = ROOT_COUNT / measurement["wall_seconds"]
                    for result, root, receipt in zip(scored.cases, inventory, query_rows, strict=True):
                        metrics = dict(result.metrics)
                        measurement["rows"].append({**root, "case_id": result.case_id,
                            **{k: v for k, v in receipt.items() if k != "query"},
                            "primary_error": result.primary_error,
                            "candidate_query_point_count": metrics.get("candidate_query_point_count"),
                            "macro_metrics": {k: v for k, v in metrics.items()
                                if "relative_l2" in k or k == "macro_match_error"},
                            "timing_metrics": {k: v for k, v in metrics.items() if "wall_seconds" in k}})
                    measurement["status"] = "measured"
                except BaseException as error:
                    measurement.update(status="failed", error_type=type(error).__name__, error=str(error),
                        wall_seconds=time.monotonic() - measured_started)
                    measurement["failures"].append({"type": type(error).__name__, "message": str(error)})
                    writer.write(f"workers-{count:02d}/failure.json", {"traceback": traceback.format_exc()})
                finally:
                    measurement["audits"] = [] if service is None else jsonable(service.audits)
                    report["measurements"].append(measurement)
                    writer.write(f"workers-{count:02d}/measurement.json", measurement)
                    writer.write("summary.json", report)
                if measurement["status"] == "failed":
                    # Preserve the failed attempt and stop; no lower-worker fallback.
                    raise RuntimeError(f"worker group {count} failed; formal launch is blocked")
            report["selected_worker_count"] = choose_worker_count(report["measurements"], inventory,
                solver_sha=initial.program.tree_sha256, point_count=point_count, worker_counts=worker_counts)
            report["base_final_verified"] = before == (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
            report["solver_final_verified"] = (inspect_candidate_bundle(initial.program.bundle_locator).tree_sha256
                                               == initial.program.tree_sha256)
            if not report["base_final_verified"] or not report["solver_final_verified"]:
                raise ValueError("preflight mutated the frozen model or Solver")
            report["status"] = "measured"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["failures"].append({"type": type(error).__name__, "message": str(error)})
        writer.write("failure.json", {"traceback": traceback.format_exc()})
    finally:
        try:
            if binding is not None:
                backend = binding.plugin.generator.backend
                if state is not None:
                    case_ids = tuple(set(backend._draws) | set(backend.runtime._resident_cases) | set(backend.backend._anchors))
                    binding.plugin.generator.release(state, case_ids)
                retention = {"draws": len(backend._draws), "pending_draws": len(backend.capture.pending),
                    "residents": len(backend.runtime._resident_cases), "anchors": len(backend.backend._anchors)}
                report["resident_counts_after_run"] = retention
                if any(retention.values()) or backend.training_audits:
                    raise ValueError("preflight leaked Case state or unexpectedly trained Guidance")
            final_revision, final_dirty = _source_snapshot()
            report["source_final_verified"] = final_revision == revision and not final_dirty
            report["config_final_verified"] = file_sha(config_path) == report["config_sha256"]
            if not report["source_final_verified"] or not report["config_final_verified"]:
                raise ValueError("source or formal configuration changed during preflight")
            if report["status"] == "measured":
                report.update(status="passed", execution_passed=True, preflight_passed=True)
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            report["failures"].append({"type": type(error).__name__, "message": str(error)})
        if report["status"] != "passed":
            report.update(execution_passed=False, preflight_passed=False, selected_worker_count=None)
        report["wall_seconds"] = time.monotonic() - started
        report["online_coarse_solver_guard"] = guard
        writer.write("summary.json", report)
        writer.write("status.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-counts", default="4,8,16")
    args = parser.parse_args(argv)
    report = run_preflight(args.config, args.output, parse_worker_counts(args.worker_counts))
    print(json.dumps({k: report.get(k) for k in ("status", "execution_passed", "selected_worker_count", "wall_seconds")}))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
