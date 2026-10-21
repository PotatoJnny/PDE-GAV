#!/usr/bin/env python3
"""Four fresh engineering Cases: exact off-grid truth and public three-GPU ABI.

No LLM, old campaign, numerical retraining, speed comparison, or repeated solve.
The first formal allocation runs this once in a separate process.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from threading import Barrier, Lock
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def check_local_ad_concurrency(binding, records):
    """Exercise the R1 failure path, not only separate hidden GPU processes.

    Reuse the smoke's existing parent/child fields. Two private truth clients
    and one public source client submit distinct small GPU batches from the
    same process, as concurrent Development calls do. No model-quality gate.
    """
    import numpy as np
    from pde_agent_verifier_solver.sdk import AnswerQuery

    barrier = Barrier(3)
    batches, points = 4, 16
    fields = ("M0", "M1", "M2", "M3", "E")

    def coordinates(record, index):
        domain = record.case.public_data["equation"]["domain"]
        start, end = domain["time_interval"]
        left, right = domain["x_interval"]
        time = start + (end - start) * (index + 1) / (batches + 1)
        positions = left + (right - left) * (np.arange(points) + .25 + .01 * index) / points
        return time, positions

    def truth_client(record):
        barrier.wait(timeout=30)
        for index in range(batches):
            time, positions = coordinates(record, index)
            query = AnswerQuery.at_points(fields=fields, points=tuple(
                {"t": float(time), "x": float(position)} for position in positions))
            answer = record.truth.answer(query)
            if any(not np.isfinite(answer.fields[name]).all() for name in fields):
                raise ValueError("concurrent private truth returned nonfinite macros")
        return batches

    def source_client(record):
        session = binding.plugin.generator.backend.public_session_factory(record.case)
        try:
            barrier.wait(timeout=30)
            domain = record.case.public_data["equation"]["domain"]
            velocity = sum(domain["velocity_query_window"]) / 2
            for index in range(batches):
                time, positions = coordinates(record, index)
                query = np.column_stack((np.full(points, time), positions, np.full(points, velocity)))
                answer = np.asarray(session.query_source(query))
                if answer.shape != (points,) or not np.isfinite(answer).all():
                    raise ValueError("concurrent public Oracle returned an invalid source batch")
        finally:
            session.close()
        return batches

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = (executor.submit(truth_client, records[0]),
                   executor.submit(truth_client, records[-1]),
                   executor.submit(source_client, records[0]))
        counts = [future.result() for future in futures]
    return {"status": "passed", "threads": 3, "private_truth_batches": sum(counts[:2]),
            "public_source_batches": counts[2], "points_per_batch": points,
            "wall_seconds": perf_counter() - started}


def run(args, report):
    import numpy as np
    from pde_agent_verifier_solver.plugins.vp_1d1v.product import load_artemis_backend
    from pde_agent_verifier_solver.plugins.vp_1d1v.evaluation_gpu_reuse import ArtemisHiddenEvaluationScope
    from pde_agent_verifier_solver.engine import PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.engine.evaluator import _CaseSessionBindings, _evaluate_records_with_truths
    from pde_agent_verifier_solver.sdk import (
        AnswerQuery, GenerationRequest, GeneratorMode, GenerationKind,
        GenerationPurpose, CaseVisibility, EvaluationPhase,
    )
    from scripts.check_artemis_guidance_execution import bind_candidates
    from scripts.check_artemis_parallel_evaluation import validate_pool

    config = json.loads(args.backend_config.read_text())
    settings = config["factory_settings"]
    binding = load_artemis_backend(**settings)
    generator = binding.plugin.generator
    generator.backend.continuous_macro_truth = True
    state = generator.load(binding.checkpoint, {})
    records = []
    service = None
    started = perf_counter()
    try:
        parents = generator.sample(state, GenerationRequest(
            round_index=1, count=3, seed=2026092601, mode=GeneratorMode.ONLINE,
            kind=GenerationKind.BASE, purpose=GenerationPurpose.PROMOTION,
            visibility=CaseVisibility.HIDDEN))
        records.extend(parents)
        child = generator.sample(state, GenerationRequest(
            round_index=1, count=1, seed=2026092602, mode=GeneratorMode.ONLINE,
            kind=GenerationKind.PERTURBED, purpose=GenerationPurpose.PROMOTION,
            visibility=CaseVisibility.HIDDEN, anchors=(parents[0].case,)))
        records.extend(child)
        if len(records) != 4 or child[0].case.parent_case_id != parents[0].case.case_id:
            raise ValueError("smoke requires three distinct parents and one paired child")
        _, candidate = bind_candidates(args.output_root)
        task = binding.plugin.task("observable_model")
        default = AnswerQuery.at_points(fields=("M0", "M1", "M2", "M3", "E"),
                                        points=({"t": .5, "x": 0.0},))
        evaluation_settings = {
            "default_query": default, "causal_horizon": True,
            "require_prepared_solution": True, "candidate_budget_mode": "ns_remaining_case_v1",
            "development_point_count": 128, "official_point_count": 256,
            "terminal_point_count": 256, "scoring_coordinate_mode": "continuous_x_v1",
        }
        evaluation = task.create_evaluation(evaluation_settings)
        plans = tuple(evaluation.make_plan(record.case, EvaluationPhase.OFFICIAL,
                      evaluation_settings, 2026092603 + (index if index < 3 else 0))
                      for index, record in enumerate(records))
        truths = tuple(record.truth.answer(plan.query) for record, plan in zip(records, plans))
        if any(len(plan.query.points) != 256 for plan in plans) or plans[0].query != plans[3].query:
            raise ValueError("off-grid count or parent/child query pairing differs")
        if any(not all(math.isfinite(v) for v in values)
               for truth in truths for values in truth.fields.values()):
            raise ValueError("off-grid exact truth contains nonfinite values")
        if not all(np.allclose(truths[0].fields[name], truths[3].fields[name],
                               rtol=1e-12, atol=1e-12) for name in ("M0", "E")):
            raise ValueError("paired exact truth did not retain parent density/electric field")
        service = SubprocessCandidateExecutionService(
            args.output_root / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)),
            evaluation_settings=evaluation_settings, max_workers=4,
            case_hard_wall_seconds=900, development_batch_hard_wall_seconds=6900,
            python_executable=sys.executable)
        devices = (os.environ["PDE_CONTROLLER_GPU_DEVICE"],
                   *os.environ["PDE_AGENT_WORKER_GPU_DEVICES"].split(","))
        if len(devices) != 3 or len(set(devices)) != 3:
            raise ValueError("expected three distinct allocated devices")
        scope = ArtemisHiddenEvaluationScope(
            binding, service, backend_settings=settings, devices=devices)
        sessions = _CaseSessionBindings(generator, state,
                    allow_development_solution=False, operation_lock=Lock())
        with scope(cases=tuple(record.case for record in records)) as pool:
            results = _evaluate_records_with_truths(
                candidate, tuple(records), evaluation, plans, truths,
                case_sessions=sessions, candidate_execution=service)
            report["pool_active"] = dict(pool.stats_snapshot)
            validate_pool(report["pool_active"], devices, closed=False)
        report["pool_closed"] = dict(pool.stats_snapshot)
        validate_pool(report["pool_closed"], devices, closed=True)
        if len(results.cases) != 4 or any(
                case.macro_match is not None and case.macro_match.failures
                for case in results.cases):
            raise ValueError("one or more short public-ABI evaluations failed")
        report["local_ad_concurrency"] = check_local_ad_concurrency(binding, records)
        report.update(status="passed", case_count=4, point_count=256,
                      parent_child_plan_shared=True, unchanged_m0_e=True,
                      elapsed_seconds=perf_counter() - started,
                      primary_errors=[row.primary_error for row in results.cases],
                      candidate_execution_audits=[asdict(row) for row in service.audits],
                      assets=binding.report)
    finally:
        if service is not None:
            service._executor.shutdown(wait=True, cancel_futures=True)
        generator.release(state, tuple(record.case.case_id for record in records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real assets and numerical smoke require a compute allocation")
    args.output_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = {"schema": "artemis_vpfp_aligned_backend_smoke_v1", "status": "running",
              "scientific_requalification": False, "model_api_called": False,
              "formal_state_imported": False}
    try:
        run(args, report)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output_root / "report.json"),
                      "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
