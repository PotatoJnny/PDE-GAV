#!/usr/bin/env python3
"""Verify real Artemis source RPC through the current CPU Candidate workers.

Reuses saved direct-D2 predictions and the unchanged benchmark's fresh cohort,
queries, serialization and scientific comparator. No Agent or hidden suite is
invoked. Controller-side oracle computation retains the allocated GPU.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import monotonic, perf_counter, time
import traceback


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parent / "artemis-eval-benchmark-20260907/scripts/benchmark_canonical_artemis_oracle.py"
SELECTED = ("R4_winner", "R5_explorer_2")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_subset(direct, direct_root):
    reference = deepcopy(direct)
    reference["candidates"] = [row for row in reference["candidates"] if row["name"] in SELECTED]
    reference["cases"] = [row for row in reference["cases"] if row["candidate"] in SELECTED]
    if (reference["status"] != "completed"
            or [row["name"] for row in reference["candidates"]] != list(SELECTED)
            or len(reference["cases"]) != 8 or len(reference["cohort"]) != 4
            or any(row["status"] != "completed" for row in reference["cases"])):
        raise ValueError("direct D2 reference must contain both frozen candidates on four completed cases")
    for row in reference["cohort"]:
        row["public_arrays_path"] = str((direct_root / row["public_arrays_path"]).resolve(strict=True))
    reference["evaluation_wall_seconds"] = sum(row["evaluation_wall_seconds"] for row in reference["cases"])
    return reference


def check_audits(audits):
    if len(audits) != 8 or len({(row["execution_id"], row["case_index"]) for row in audits}) != 8:
        raise ValueError("split execution must contain eight unique worker audits")
    for row in audits:
        if (row["status"] != "completed" or row["return_code"] != 0
                or row["process_group_cleaned"] is not True
                or row["controller_truth_shared"] is not False
                or row["candidate_process_isolated"] is not True
                or row["error_code"] is not None or row["timeout_seconds"] != 900.0):
            raise ValueError("split Candidate worker failed, leaked truth or did not confirm cleanup")


def run_split(benchmark, binding, config, reference, root):
    from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.engine.evaluator import _CaseSessionBindings, _evaluate_records_with_truths
    from pde_agent_verifier_solver.sdk import AnswerQuery, EvaluationPhase, TaskKind

    device = str(config["factory_settings"]["device"])
    state, records, descriptors = benchmark._cohort(binding, parents=2, master_seed=2026090801, root=root, device=device)
    service = None
    report = {key: deepcopy(reference[key]) for key in (
        "schema", "master_seed", "parents", "point_count", "candidate_compute_budget", "evaluation_phase", "horizon_policy")}
    report.update(label="split_cpu_workers", status="running", started_at_unix=time(),
                  canonical_source=benchmark._version(ROOT), backend_configuration=config,
                  backend_config_sha256=sha256((ROOT / "configs/plasma_canonical_artemis.json").read_bytes()).hexdigest(),
                  physical_source=benchmark._version(Path(config["factory_settings"]["physical_checkout"])),
                  environment={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  cohort=descriptors, candidates=[], cases=[], agent_invoked=False,
                  official_or_terminal_cases_used=False, formal_iteration_resume=False)
    try:
        arrays = records[0].case.public_data["arrays"]
        query = AnswerQuery.at_points(fields=benchmark.FIELDS,
            points=({"t": float(arrays["t"][1]), "x": float(arrays["x"][0])},))
        settings = {"default_query": query, "development_point_count": 32, "causal_horizon": True,
                    "require_prepared_solution": True, "candidate_compute_budget": 600.0}
        task = binding.plugin.task("observable_model")
        evaluation = task.create_evaluation(settings)
        plans, truths = [], []
        for record, descriptor in zip(records, descriptors):
            seed = benchmark._seed(2026090801, "plan", descriptor["pair_index"])
            private = evaluation.make_private_development_plan(record.case, settings, seed)
            query = benchmark._query_at_benchmark_horizon(private.query, record.case, horizon_policy="native-end")
            plan = evaluation.make_plan(record.case, EvaluationPhase.DEVELOPMENT, settings, seed, request=query)
            plans.append(plan)
            truths.append(record.truth.answer(plan.query))
            descriptor["plan"] = {"phase": plan.phase.value, "seed": plan.seed,
                                  "query": benchmark._answer_payload(truths[-1])["points"]}
            descriptor["oracle_probe"] = benchmark._probe_public_oracle(binding.plugin.generator, state, record,
                plan, benchmark._seed(2026090801, "public-oracle-probe", descriptor["pair_index"]), device)
            benchmark._write_json(root / "generation" / f"{descriptor['key']}.json", descriptor)
        service = SubprocessCandidateExecutionService(root / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)), evaluation_settings=settings,
            max_workers=2, case_hard_wall_seconds=900, development_batch_hard_wall_seconds=6900,
            python_executable=sys.executable)
        codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
        meter = benchmark._MeteredGenerator(binding.plugin.generator)
        sessions = _CaseSessionBindings(meter, state, allow_development_solution=False, operation_lock=Lock())
        for candidate in reference["candidates"]:
            name = candidate["name"]
            artifact = codec.bind_existing(Path(candidate["source_path"]), storage_root=root / "candidates" / name,
                artifact_id=f"split-{name}", name=name, summary="Frozen source-RPC split acceptance", usage="prepare_solution then evaluate POINTS")
            report["candidates"].append({"name": name, "tree_sha256": artifact.program.tree_sha256,
                                         "source_path": candidate["source_path"]})
            session_start, audit_start = len(meter.sessions), len(service.audits)
            benchmark._synchronize(device)
            began = perf_counter()
            result = _evaluate_records_with_truths(artifact, records, evaluation, plans, truths,
                case_sessions=sessions, candidate_execution=service,
                development_batch_deadline=monotonic() + 6900)
            benchmark._synchronize(device)
            elapsed = perf_counter() - began
            audits = {audit.case_index: audit for audit in service.audits[audit_start:]}
            for index, (record, descriptor, plan, truth, case) in enumerate(zip(records, descriptors, plans, truths, result.cases)):
                row = {"candidate": name, "case_key": descriptor["key"], "status": "completed",
                       "plan_seed": plan.seed, "truth": benchmark._answer_payload(truth),
                       "prediction": benchmark._answer_payload(case.prediction), "primary_error": case.primary_error,
                       "metrics": dict(case.metrics), "evaluation_wall_seconds": audits[index].wall_seconds,
                       "public_sessions": [session for session in meter.sessions[session_start:]
                                           if session["case_id"] == record.case.case_id]}
                report["cases"].append(row)
                benchmark._write_json(root / "case_results" / f"{name}-{descriptor['key']}.json", row)
                print(json.dumps({"stage": "split_case_evaluated", "candidate": name,
                                  "case_key": descriptor["key"], "worker_wall_seconds": audits[index].wall_seconds}), flush=True)
            benchmark._write_json(root / "partial_result.json", report)
            print(json.dumps({"stage": "split_batch_complete", "candidate": name, "wall_seconds": elapsed}), flush=True)
        report.update(status="completed", evaluation_wall_seconds=sum(row["evaluation_wall_seconds"] for row in report["cases"]),
                      finished_at_unix=time(), candidate_execution_audits=[asdict(audit) for audit in service.audits])
        check_audits(report["candidate_execution_audits"])
        benchmark._write_json(root / "result.json", report)
        return report
    finally:
        if service is not None:
            benchmark._write_json(root / "candidate-audits.json", [asdict(audit) for audit in service.audits])
        binding.plugin.generator.release(state, tuple(record.case.case_id for record in records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        parser.error("real-asset split acceptance requires a Slurm compute allocation")
    benchmark = load_module(BENCHMARK, "artemis_frozen_benchmark")
    helper = load_module(ROOT / "scripts/artemis_backend_cold_acceptance.py", "artemis_cold_acceptance")
    root = benchmark._fresh_root(args.output_root)
    summary = {"schema": "artemis_split_execution_acceptance_v1", "status": "failed",
               "completed_evaluations": 0, "agent_invoked": False, "formal_scientific_admission": False,
               "candidate_case_workers": 2, "candidate_device": "cpu", "case_hard_wall_seconds": 900,
               "development_batch_hard_wall_seconds": 6900, "performance_comparison_claimed": False}
    try:
        reference = reference_subset(json.loads(args.direct_result.read_text()), args.direct_result.parent)
        current = benchmark._version(ROOT)
        config_path = ROOT / "configs/plasma_canonical_artemis.json"
        digest = sha256(config_path.read_bytes()).hexdigest()
        summary.update(canonical_commit=current["commit"], backend_config_sha256=digest)
        if (current["status_porcelain"] or reference["canonical_source"] != current
                or reference["backend_config_sha256"] != digest):
            raise ValueError("split acceptance requires the same clean D2 source as the direct reference")
        expected = {name: tree for name, _, tree in helper.CANDIDATES if name in SELECTED}
        if {row["name"]: row["tree_sha256"] for row in reference["candidates"]} != expected:
            raise ValueError("unexpected direct reference candidate hashes")
        for name, value in {"master_seed": 2026090801, "parents": 2, "point_count": 32,
                            "candidate_compute_budget": 600.0, "horizon_policy": "native-end",
                            "evaluation_phase": "development"}.items():
            if reference[name] != value:
                raise ValueError(f"unexpected direct reference setting: {name}")
        benchmark._write_json(root / "direct-reference.json", reference)
        binding, config = benchmark._load_binding(ROOT, config_path)
        split = run_split(benchmark, binding, config, reference, root)
        comparison = benchmark._comparison_payload(reference, split,
            left_root=args.direct_result.parent, right_root=root, rtol=1e-10, atol=1e-12)
        # The existing comparator inspects only scientific equality. Its timing
        # ratios are deliberately omitted: serial and parallel clocks differ.
        comparison.pop("evaluation_wall_speedup", None)
        for row in comparison.get("cases", []):
            row.pop("wall_speedup", None)
        benchmark._write_json(root / "comparison.json", comparison)
        if comparison["status"] != "equivalent":
            raise ValueError("direct versus split scientific comparison failed")
        if helper.workload(reference) != helper.workload(split):
            raise ValueError("split execution changed public calls, points or failures")
        if any(not any(session["methods"].get("query_source", {}).get("calls", 0) > 0
                       for session in case["public_sessions"]) for case in split["cases"]):
            raise ValueError("a split candidate did not exercise the public source RPC")
        summary.update(status="passed", completed_evaluations=len(split["cases"]),
                       clean_worker_audits=len(split["candidate_execution_audits"]),
                       scientific_comparison="equivalent", public_workload="identical")
    except BaseException as error:
        summary.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    benchmark._write_json(root / "acceptance-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
