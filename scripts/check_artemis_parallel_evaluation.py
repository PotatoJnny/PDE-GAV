#!/usr/bin/env python3
"""Bounded real CPU Candidate -> HTTP -> one/three GPU Artemis D5 smoke.

Two frozen R1 Candidates share four fresh engineering Cases and one frozen
Task-owned hidden plan set. This is not a Campaign, Official score replacement,
Guidance fit, full physics qualification, or a general performance benchmark.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "artemis_parallel_evaluation_smoke_v1"
EXPECTED_PHYSICAL = "e0cccc649a9ccf48ad69ac2d161c4a6b46010675"
MASTER_SEED = 2026091001
CASE_COUNT, CANDIDATE_COUNT, WORKERS = 4, 2, 4


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocated_devices(raw):
    devices = tuple(value.strip() for value in raw.split(","))
    if len(devices) != 3 or any(not value for value in devices) or len(set(devices)) != 3:
        raise ValueError("parallel smoke requires exactly three distinct allocated GPU tokens")
    return devices


def validate_pool(stats, devices, *, closed):
    lanes = stats.get("lanes", [])
    if (stats.get("started") is not True or stats.get("closed") is not closed
            or len(lanes) != len(devices)
            or tuple(str(row.get("device")) for row in lanes) != tuple(devices)):
        raise ValueError("parallel pool did not bind the exact allocated GPU lanes")
    pids = [row.get("pid") for row in lanes]
    if len(set(pids)) != len(devices) or any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 for pid in pids):
        raise ValueError("each GPU lane must have a distinct real worker process")
    for row in lanes:
        identity = row.get("device_metadata", {})
        if (identity.get("cuda_visible_devices") != str(row["device"])
                or identity.get("device_count") != 1 or identity.get("current_device") != 0
                or identity.get("cpu_threads") != 1 or not identity.get("name")):
            raise ValueError("GPU worker did not confirm its single-device CUDA mask at startup")
        for key in ("request_count", "source_calls", "distribution_calls"):
            if not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] <= 0:
                raise ValueError(f"every GPU lane must receive real source/initial requests: {key}")
        if row.get("alive") is not (not closed):
            raise ValueError("GPU lane process liveness disagrees with scope lifecycle")
    uuids = [row["device_metadata"].get("uuid") for row in lanes]
    available_uuids = [value for value in uuids if value]
    if len(available_uuids) != len(set(available_uuids)):
        raise ValueError("different GPU lanes reported the same physical-device UUID")
    if closed and stats.get("cleanup_complete") is not True:
        raise ValueError("GPU pool process cleanup was not confirmed")


def public_workload(report):
    """Compare per-Case call order/coordinates, not scheduling/timing order."""
    rows = []
    for row in report.get("cases", []):
        sessions = []
        for session in row.get("public_sessions", []):
            digest = session.get("request_identity_sha256")
            methods = {name: {key: values.get(key) for key in ("calls", "points", "failures")}
                       for name, values in session.get("methods", {}).items()}
            if (not isinstance(digest, str) or len(digest) != 64
                    or methods.get("query_source", {}).get("calls", 0) <= 0
                    or methods.get("query_initial_distribution", {}).get("calls", 0) <= 0
                    or any(values.get("failures") != 0 for values in methods.values())):
                raise ValueError("missing or failed exact public query workload")
            sessions.append({"request_identity_sha256": digest, "methods": methods})
        if not sessions:
            raise ValueError("every Candidate/Case must use its real public source session")
        rows.append({"candidate": row["candidate"], "case_key": row["case_key"],
                     "plan_seed": row["plan_seed"], "sessions": sessions})
    if len(rows) != CASE_COUNT * CANDIDATE_COUNT:
        raise ValueError("parallel smoke requires eight completed Candidate/Case results per arm")
    return rows


def compare_runs(benchmark, single, parallel, root):
    comparison = benchmark._comparison_payload(single, parallel,
        left_root=root, right_root=root, rtol=1e-10, atol=1e-12)
    # Only a connectivity/correctness smoke, not a robust speedup estimate.
    comparison.pop("evaluation_wall_speedup", None)
    for row in comparison.get("cases", []):
        row.pop("wall_speedup", None)
    if comparison.get("status") != "equivalent":
        raise ValueError(f"one/three GPU scientific comparison failed: {comparison.get('problems')}")
    if public_workload(single) != public_workload(parallel):
        raise ValueError("GPU routing changed Case order, query coordinates/order, counts or failures")
    comparison.update(public_workload="identical", multi_gpu_speedup_claimed=False)
    return comparison


def validate_solve_metrics(metrics):
    keys = ("candidate_solve_wall_seconds", "candidate_solve_active_seconds",
            "candidate_platform_wait_seconds", "candidate_oracle_queue_wait_seconds",
            "candidate_oracle_lock_wait_seconds")
    values = [metrics.get(key) for key in keys]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("missing real solve/wait budget metrics")
    wall, active, wait, queue, lock = values
    if active > 600.0 or abs(wall - active - wait) > .02 or abs(wait - queue - lock) > 1e-6:
        raise ValueError("parallel execution broke charged solve-budget accounting")


def _metered_generator(benchmark, generator):
    class RoutedMeter(benchmark._MeteredGenerator):
        def public_session_operation_lock(self, state, case):
            method = getattr(self.generator, "public_session_operation_lock", None)
            return method(state, case) if callable(method) else None
    return RoutedMeter(generator)


def _hidden_cohort(benchmark, binding, root):
    """Use normal issued hidden Cases, never relabel Development records.

    This is the split benchmark's two-parent paired sampling recipe with the
    SDK's hidden/promotion issuance contract required by the parallel pool.
    It creates no Campaign and does not use any saved formal Official suite.
    """
    import numpy as np
    from pde_agent_verifier_solver.sdk import (
        CaseVisibility, GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode,
    )
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    records, descriptors, owned = [], [], []
    try:
        for pair_index in range(2):
            parent = None
            for distribution, kind in (("base", GenerationKind.BASE), ("base_perturbed", GenerationKind.PERTURBED)):
                seed = benchmark._seed(MASTER_SEED, f"generation/{distribution}", pair_index)
                request = GenerationRequest(round_index=1, count=1, mode=GeneratorMode.ONLINE,
                    kind=kind, visibility=CaseVisibility.HIDDEN, seed=seed,
                    purpose=GenerationPurpose.PROMOTION, anchors=() if parent is None else (parent.case,))
                began = perf_counter()
                batch = tuple(generator.sample(state, request))
                owned.extend(record.case.case_id for record in batch)
                if len(batch) != 1 or batch[0].case.visibility is not CaseVisibility.HIDDEN:
                    raise ValueError("parallel smoke requires normally issued hidden Cases")
                record = batch[0]
                if parent is not None and record.case.parent_case_id != parent.case.case_id:
                    raise ValueError("parallel smoke lost BASE/PERT pairing")
                if parent is None:
                    parent = record
                key = f"pair-{pair_index:03d}-{distribution}"
                arrays = {name: np.asarray(value) for name, value in record.case.public_data["arrays"].items()}
                archive = root / "case_data" / f"{key}.npz"
                archive.parent.mkdir(exist_ok=True)
                np.savez(archive, **arrays)
                os.chmod(archive, 0o600)
                descriptor = {"key": key, "pair_index": pair_index, "distribution": distribution,
                    "case_id": record.case.case_id, "parent_case_id": record.case.parent_case_id,
                    "visibility": "hidden", "generation_seed": seed,
                    "generation_wall_seconds": perf_counter() - began,
                    "public_arrays_path": str(archive.relative_to(root)),
                    "public_array_sha256": {name: sha256(value.tobytes(order="C")).hexdigest()
                                            for name, value in arrays.items()}}
                records.append(record)
                descriptors.append(descriptor)
                benchmark._write_json(root / "generation" / f"{key}.json", descriptor)
                print(json.dumps({"stage": "parallel_smoke_case_generated", "case_key": key}), flush=True)
        return state, tuple(records), descriptors
    except BaseException:
        generator.release(state, tuple(owned))
        raise


def _run_arm(*, benchmark, split, binding, state, records, descriptors, candidates,
             evaluation, settings, plans, truths, service, root, label, config):
    from pde_agent_verifier_solver.engine.evaluator import _CaseSessionBindings, _evaluate_records_with_truths

    meter = _metered_generator(benchmark, binding.plugin.generator)
    sessions = _CaseSessionBindings(meter, state, allow_development_solution=False, operation_lock=Lock())
    report = {"schema": benchmark.SCHEMA, "label": label, "status": "running",
        "master_seed": MASTER_SEED, "parents": 2, "point_count": 32,
        "candidate_compute_budget": 600.0, "evaluation_phase": "official",
        "horizon_policy": "private-random", "backend_configuration": config,
        "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cohort": descriptors, "candidates": [], "cases": [], "candidate_device": "cpu",
        "fresh_engineering_cases_only": True, "formal_official_replacement": False}
    arm_audit_start = len(service.audits)
    arm_started = perf_counter()
    try:
        for candidate, artifact in candidates:
            report["candidates"].append({"name": candidate["name"], "tree_sha256": artifact.program.tree_sha256,
                                         "source_path": candidate["path"]})
            session_start, audit_start = len(meter.sessions), len(service.audits)
            began = perf_counter()
            result = _evaluate_records_with_truths(artifact, records, evaluation, plans, truths,
                case_sessions=sessions, candidate_execution=service)
            audits = {audit.case_index: audit for audit in service.audits[audit_start:]}
            for index, (record, descriptor, plan, truth, case) in enumerate(zip(records, descriptors, plans, truths, result.cases, strict=True)):
                validate_solve_metrics(case.metrics)
                row = {"candidate": candidate["name"], "case_key": descriptor["key"], "status": "completed",
                    "plan_seed": plan.seed, "truth": benchmark._answer_payload(truth),
                    "prediction": benchmark._answer_payload(case.prediction), "primary_error": case.primary_error,
                    "metrics": dict(case.metrics), "evaluation_wall_seconds": audits[index].wall_seconds,
                    "public_sessions": [deepcopy(session) for session in meter.sessions[session_start:]
                                        if session["case_id"] == record.case.case_id]}
                report["cases"].append(row)
            benchmark._write_json(root / label / "partial-result.json", report)
            print(json.dumps({"stage": "parallel_smoke_candidate", "arm": label,
                "candidate": candidate["name"], "cases": len(result.cases), "wall_seconds": perf_counter() - began}), flush=True)
        report.update(status="completed", evaluation_wall_seconds=perf_counter() - arm_started,
            candidate_execution_audits=[asdict(audit) for audit in service.audits[arm_audit_start:]])
        split.check_audits(report["candidate_execution_audits"])
        public_workload(report)
        return report
    finally:
        benchmark._write_json(root / label / "candidate-audits.json", [asdict(audit) for audit in service.audits[arm_audit_start:]])
        benchmark._write_json(root / label / "result.json", report)


def run_smoke(args, benchmark, previous, split, root, summary):
    # Import the framework only through the pinned real factory first.
    binding, config = benchmark._load_binding(ROOT, args.backend_config)
    from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.plugins.vp_1d1v.evaluation_gpu_reuse import ArtemisHiddenEvaluationScope
    from pde_agent_verifier_solver.sdk import AnswerQuery, CaseAccessKind, EvaluationPhase, TaskKind
    import numpy as np
    import torch

    devices = allocated_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise ValueError("real smoke needs three visible CUDA GPUs")
    summary.update(devices=list(devices), backend_configuration=config, backend_report=dict(binding.report))
    state, records, descriptors = _hidden_cohort(benchmark, binding, root)
    service = None
    try:
        arrays = records[0].case.public_data["arrays"]
        query = AnswerQuery.at_points(fields=benchmark.FIELDS,
            points=({"t": float(arrays["t"][1]), "x": float(arrays["x"][0])},))
        settings = {"default_query": query, "official_point_count": 32,
            "causal_horizon": True, "require_prepared_solution": True, "candidate_compute_budget": 600.0}
        task = binding.plugin.task("observable_model")
        evaluation = task.create_evaluation(settings)
        plans, truths = [], []
        for record, descriptor in zip(records, descriptors, strict=True):
            seed = benchmark._seed(MASTER_SEED, "plan", descriptor["pair_index"])
            plan = evaluation.make_plan(record.case, EvaluationPhase.OFFICIAL, settings, seed)
            plans.append(plan)
            truths.append(record.truth.answer(plan.query))
            descriptor["plan"] = {"phase": plan.phase.value, "seed": plan.seed,
                                  "query": benchmark._answer_payload(truths[-1])["points"]}
        summary["cohort"] = descriptors
        codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
        candidates = []
        for candidate in previous.frozen_candidates(args.source_campaign):
            artifact = codec.bind_existing(Path(candidate["path"]), storage_root=root / "candidates" / candidate["name"],
                artifact_id=candidate["artifact_id"], name=candidate["name"],
                summary="Frozen R1 multi-GPU connectivity smoke", usage="prepare_solution then evaluate")
            if artifact.program.tree_sha256 != candidate["tree_sha256"]:
                raise ValueError("frozen R1 Candidate bundle content changed")
            candidates.append((candidate, artifact))
        service = SubprocessCandidateExecutionService(root / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)), evaluation_settings=settings,
            max_workers=WORKERS, case_hard_wall_seconds=900,
            development_batch_hard_wall_seconds=6900, python_executable=sys.executable)
        common = dict(benchmark=benchmark, split=split, binding=binding, state=state,
            records=records, descriptors=descriptors, candidates=candidates, evaluation=evaluation,
            settings=settings, plans=plans, truths=truths, service=service, root=root, config=config)
        single = _run_arm(**common, label="single_gpu")
        summary["completed_evaluations"] = len(single["cases"])
        scope = ArtemisHiddenEvaluationScope(binding, service, backend_settings=config["factory_settings"], devices=devices)
        with scope(cases=tuple(record.case for record in records)) as pool:
            if service.oracle_host_count != 3:
                raise ValueError("multi-GPU scope did not enable three hidden oracle hosts")
            parallel = _run_arm(**common, label="three_gpu")
            summary["pool_active"] = dict(pool.stats_snapshot)
            validate_pool(summary["pool_active"], devices, closed=False)
        summary["scope_report"] = scope.last_report
        summary["pool_closed"] = dict(pool.stats_snapshot)
        validate_pool(summary["pool_closed"], devices, closed=True)
        summary["worker_device_identity"] = ("distinct_cuda_uuid" if all(
            lane["device_metadata"].get("uuid") for lane in summary["pool_closed"]["lanes"])
            else "distinct_requested_masks_and_single_visible_cuda_device")
        if service.oracle_host_count != 1:
            raise ValueError("hidden scope did not restore the single-host Development service")
        generator = binding.plugin.generator
        if getattr(generator._backend(), "_parallel_oracle_pool", None) is not None:
            raise ValueError("hidden scope left a multi-GPU pool attached in Development")
        if generator.public_session_operation_lock(state, records[0].case) is not None:
            raise ValueError("Development still resolves a GPU-lane operation lock after scope exit")
        # One tiny ordinary public call confirms resident primary-GPU reuse;
        # the closed worker pool must receive no more requests.
        domain = records[0].case.public_data["equation"]["domain"]
        point = np.array([[float(domain["time_interval"][0]), float(domain["x_interval"][0]), 0.0]], dtype=np.float64)
        session = generator.open_case_session(state, records[0].case, CaseAccessKind.PUBLIC_PROBLEM)
        try:
            value = np.asarray(session.query_source(point))
            if value.shape != (1,) or not np.isfinite(value).all():
                raise ValueError("post-scope primary-GPU Development source failed")
        finally:
            session.close()
        if pool.stats_snapshot != summary["pool_closed"]:
            raise ValueError("Development issued a request to the closed hidden GPU pool")
        comparison = compare_runs(benchmark, single, parallel, root)
        benchmark._write_json(root / "comparison.json", comparison)
        summary.update(status="passed", completed_evaluations=16, scientific_comparison="equivalent",
            public_workload="identical", development_single_gpu_restored=True,
            candidate_workers_cleaned=True, gpu_workers_cleaned=True)
    finally:
        try:
            if service is not None:
                service._executor.shutdown(wait=True)
                benchmark._write_json(root / "all-candidate-audits.json", [asdict(audit) for audit in service.audits])
                if service._processes or service._unhealthy_reason is not None:
                    raise RuntimeError("Candidate service did not finish healthy with all processes removed")
        finally:
            binding.plugin.generator.release(state, tuple(record.case.case_id for record in records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, default=ROOT / "configs/plasma_artemis_opt5.json")
    parser.add_argument("--source-campaign", type=Path,
        default=Path("${PDE_GAV_RUN_ROOT}/artemis_canonical_d2_guided_5round_20260908/job-5309385/campaign"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        parser.error("real multi-GPU smoke requires its own Slurm compute allocation")
    previous = load_module(ROOT / "scripts/benchmark_artemis_opt3.py", "parallel_smoke_previous")
    previous.configure_threads(1)
    benchmark = previous.harness()
    previous.install_request_meter(benchmark)
    split = load_module(ROOT / "scripts/check_artemis_split_execution.py", "parallel_smoke_split")
    root = benchmark._fresh_root(args.output_root)
    summary = {"schema": SCHEMA, "status": "failed", "completed_evaluations": 0,
        "agent_invoked": False, "guidance_trained": False, "formal_official_replacement": False,
        "formal_scientific_admission": False, "multi_gpu_speedup_claimed": False,
        "candidate_count": CANDIDATE_COUNT, "case_count": CASE_COUNT, "candidate_case_workers": WORKERS}
    try:
        summary["driver_source"] = benchmark._version(ROOT)
        config = json.loads(args.backend_config.read_text())
        summary["backend_config_sha256"] = sha256(args.backend_config.read_bytes()).hexdigest()
        physical = benchmark._version(Path(config["factory_settings"]["physical_checkout"]))
        summary["physical_source"] = physical
        if summary["driver_source"]["status_porcelain"] or physical["status_porcelain"]:
            raise ValueError("smoke requires clean committed driver and physical source")
        if config["factory_settings"].get("expected_physical_revision") != EXPECTED_PHYSICAL or physical["commit"] != EXPECTED_PHYSICAL:
            raise ValueError("parallel smoke must use the pinned D5 physical revision")
        run_smoke(args, benchmark, previous, split, root, summary)
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    benchmark._write_json(root / "acceptance-summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("schema", "status", "completed_evaluations")}
        | {"summary_path": str(root / "acceptance-summary.json"),
           "error": summary.get("error")}, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
